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
    assert "-interface SPIx4" in tcl and "-loadbit \"up 0x0" in tcl and "-loaddata \"up 0x280000" in tcl

@pytest.mark.parametrize("ddr", [False, True])
def test_bitstream_dry_run_tcl_variants(tmp_path, ddr):
    tcl = build.bitstream_tcl(tmp_path / ("ddr.tcl" if ddr else "sdr.tcl"), tmp_path / "boolean_top.bit")
    assert "source {" in tcl and "write_bitstream -force" in tcl
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
