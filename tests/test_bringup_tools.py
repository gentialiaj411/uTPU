import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

build = load("build_bitstream", "tools/board/build_bitstream.py")
mcs = load("make_flash_image", "tools/board/make_flash_image.py")

def test_flash_byte_map_and_overlap(tmp_path):
    bit = tmp_path / "x.bit"; weights = tmp_path / "image.bin"
    bit.write_bytes(b"B" * 1024); weights.write_bytes(b"W" * 4096)
    assert mcs.byte_map(bit, weights) == [("bitstream", 0, 1024), ("weights", 0x280000, 0x280000 + 4096)]
    bit.write_bytes(b"B" * (0x280000 + 1))
    with pytest.raises(ValueError, match="overlaps"):
        mcs.byte_map(bit, weights)

def test_flash_tcl_has_required_offsets_and_interface(tmp_path):
    tcl = mcs.make_tcl(tmp_path / "boolean_top.bit", tmp_path / "image.bin", tmp_path / "boolean_top.mcs")
    assert "-interface SPIx4" in tcl and "-loadbit [list up 0x0 {" in tcl and "-loaddata [list up 0x280000 {" in tcl

@pytest.mark.parametrize("ddr", [False, True])
def test_bitstream_dry_run_tcl_variants(tmp_path, ddr):
    tcl = build.bitstream_tcl(tmp_path / ("ddr.tcl" if ddr else "sdr.tcl"), tmp_path / "boolean_top.bit")
    assert "open_checkpoint {" in tcl and "write_bitstream -force" in tcl
    assert "source" not in tcl and "route_design" not in tcl   # never re-implements
    assert "SPI_BUSWIDTH 4" in tcl and "SPI_32BIT_ADDR No" in tcl and "CONFIGRATE 26" in tcl

def test_refuses_failed_timing(tmp_path):
    (tmp_path / "timing.rpt").write_text("WNS(ns) TNS(ns) WHS(ns) THS(ns)\nfoo\n -0.100 -1.000 0 0 0.100 0.000 0 0\n")
    (tmp_path / "drc.rpt").write_text("Checks found: 1\n")
    with pytest.raises(RuntimeError, match="refusing"):
        build.parse_reports(tmp_path)

def test_refuses_drc_error(tmp_path):
    (tmp_path / "timing.rpt").write_text("WNS(ns) TNS(ns) WHS(ns) THS(ns)\nfoo\n 0.100 0.000 0 0 0.100 0.000 0 0\n")
    (tmp_path / "drc.rpt").write_text("Checks found: 1\n| DRC-1 | Error | bad |\n")
    with pytest.raises(RuntimeError, match="refusing"):
        build.parse_reports(tmp_path)

def _write_hex(path, mem):
    lines, base = [], None
    for a in sorted(mem):
        if a >> 16 != base:
            base = a >> 16
            r = bytes([2, 0, 0, 4]) + base.to_bytes(2, "big"); lines.append(":" + (r + bytes([-sum(r) & 0xFF])).hex().upper())
        r = bytes([1]) + (a & 0xFFFF).to_bytes(2, "big") + bytes([0, mem[a]]); lines.append(":" + (r + bytes([-sum(r) & 0xFF])).hex().upper())
    lines.append(":00000001FF"); path.write_text("\n".join(lines) + "\n")

def test_verify_mcs_accepts_exact_and_catches_corruption(tmp_path):
    bitdata = b"\xff" * 16 + bytes.fromhex("AA995566") + bytes(range(200))
    bit = tmp_path / "x.bit"; bit.write_bytes(b"header" + bitdata)
    w = bytes((i * 7) & 0xFF for i in range(300)); weights = tmp_path / "image.bin"; weights.write_bytes(w)
    mem = {i: b for i, b in enumerate(bitdata)} | {mcs.WEIGHT_OFFSET + i: b for i, b in enumerate(w)}
    good = tmp_path / "good.mcs"; _write_hex(good, mem)
    assert mcs.verify_mcs(good, bit, weights)["weights_bytes_verified"] == 300
    bad = dict(mem); bad[mcs.WEIGHT_OFFSET + 123] ^= 0x01
    _write_hex(tmp_path / "bad.mcs", bad)
    with pytest.raises(ValueError, match="weights differ"):
        mcs.verify_mcs(tmp_path / "bad.mcs", bit, weights)
    swapped = {a: int(f"{b:08b}"[::-1], 2) if a >= mcs.WEIGHT_OFFSET else b for a, b in mem.items()}   # bit-reversed data
    _write_hex(tmp_path / "swap.mcs", swapped)
    with pytest.raises(ValueError, match="weights differ"):
        mcs.verify_mcs(tmp_path / "swap.mcs", bit, weights)
    nobit = {a: b for a, b in mem.items() if a >= mcs.WEIGHT_OFFSET}
    _write_hex(tmp_path / "nobit.mcs", nobit)
    with pytest.raises(ValueError, match="bitstream"):
        mcs.verify_mcs(tmp_path / "nobit.mcs", bit, weights)
