#!/usr/bin/env python3
"""P0 onboard bring-up: flash shipping bitstream, UART upload MNIST, bit-exact check.

Emits ``bench/results/onboard_execution.json`` with full provenance.
Does **not** claim hardware success unless captured UART bytes match ISA sim
and the replay test can re-validate the staged transcript without a board.

Diagnostic failures (baud, framing, fit, missing bitstream, no COM port) are
recorded explicitly — they are evidence, not silent retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HOST = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST))

from board_config import BoardConfig  # noqa: E402
from isa_encoder import IsaConfig  # noqa: E402
from isa_simulator import simulate_program_bytes  # noqa: E402
from lowering_fused_mlp_utpu import lower_fused_mlp_program_utpu  # noqa: E402
from program_loader import MAGIC_START, MAGIC_UPLOAD, ProgramLoader  # noqa: E402
from uart_replay import serialize_program_upload  # noqa: E402

OUT_JSON = REPO / "bench" / "results" / "onboard_execution.json"
EVIDENCE_DIR = REPO / "docs" / "evidence" / "fpga_onboard"
SHIPPING_REPORT_PREFIX = "dss_n8_cdw8_mb48_clk12_pd3_prog65536"
SHIPPING_DCP = (
    REPO
    / "build"
    / "vivado_uTPU_dss_n8_cdw8_mb48_clk12"
    / "uTPU_dss_n8_cdw8_mb48_clk12.runs"
    / "impl_1"
    / "top_routed.dcp"
)
SHIPPING_BIT = REPO / "build" / "reports" / f"{SHIPPING_REPORT_PREFIX}.bit"
UPLOAD_LEN_MAX = 65535
DEFAULT_BAUD = 115200
MAGIC_REPEAT = 0xA6
BUFFER_RESIDENT_ARTIFACT_REQUIRED = {
    "schema_version", "generated_at_utc", "status", "claim_rule", "scope",
    "deployment", "float_accuracy", "quantized_accuracy",
    "isa_vs_integer_reference", "device_cycles_per_inference",
    "board_perf_program_cycles_per_inference", "simulation_seconds",
    "board_host_elapsed_seconds", "board_host_seconds_per_image",
    "board_host_latency_scope", "bitstream", "flash", "uart",
}


def validate_buffer_resident_artifact(artifact: Dict[str, Any]) -> None:
    missing = BUFFER_RESIDENT_ARTIFACT_REQUIRED.difference(artifact)
    if missing:
        raise ValueError(f"buffer-resident artifact missing schema fields: {sorted(missing)}")
    if artifact["schema_version"] != 1:
        raise ValueError("unsupported buffer-resident artifact schema_version")
    uart = artifact["uart"]
    rows = uart.get("per_image", [])
    if int(artifact["scope"]["images_requested"]) < 1:
        raise ValueError("artifact images_requested must be positive")
    if artifact["status"] == "hardware_bit_exact":
        if not uart.get("attempted") or len(rows) != int(artifact["scope"]["images_requested"]):
            raise ValueError("hardware_bit_exact requires a UART result row for every requested image")
        if not all(row.get("output_bytes_bit_exact") is True for row in rows):
            raise ValueError("hardware_bit_exact requires every output byte comparison to pass")

# Board-specific defaults are deliberately explicit: UART protocol and the
# generated program stay common, while the RTL build provenance cannot.
BOARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "arty_a7_100t": {
        "label": "Arty A7-100T",
        "part": "xc7a100tcsg324-1",
        "board_config": BoardConfig.artix_a7100t_bram_max(65536),
        "report_prefix": SHIPPING_REPORT_PREFIX,
        "dcp": SHIPPING_DCP,
        "bit": SHIPPING_BIT,
        "clock_period_ns": 12.0,
        "frequency_mhz": 83.33333333333333,
        "wns_ns": 0.271,
        "max_batch_count": 48,
    },
    "boolean_xc7s50": {
        "label": "Real Digital Boolean",
        "part": "xc7s50csga324-1",
        "board_config": BoardConfig.boolean_xc7s50(),
        "report_prefix": "boolean_n8_cdw8_mb4_clk12_pd3_prog65536",
        "dcp": REPO / "build" / "vivado_uTPU_boolean_xc7s50" / "uTPU_boolean_xc7s50.runs" / "impl_1" / "top_routed.dcp",
        "bit": REPO / "bench" / "results" / "boolean_n8_cdw8_mb4_clk12_pd3_prog65536.bit",
        "clock_period_ns": 12.0,
        "frequency_mhz": 83.33333333333333,
        "wns_ns": None,
        "max_batch_count": 4,
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_sha() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except OSError:
        pass
    return None


def _host_env() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": str(REPO),
        "git_sha": _git_sha(),
        "machine": platform.machine(),
        "node": platform.node(),
    }


def _load_shipping_point() -> Dict[str, Any]:
    path = REPO / "bench" / "results" / "design_space_sweep.json"
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    return dict(blob.get("shipping_point") or {})


def _list_serial_ports() -> List[Dict[str, str]]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return [{"error": "pyserial_not_installed"}]
    return [
        {
            "device": p.device,
            "description": p.description or "",
            "hwid": p.hwid or "",
        }
        for p in list_ports.comports()
    ]


def _path_for_artifact(path: Path) -> str:
    try:
        return str(path.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path)


def _ensure_bitstream(
    *, profile: Dict[str, Any], bitstream: Optional[Path], write_bit: bool
) -> Dict[str, Any]:
    """Locate a board bitstream, optionally materializing the default DCP."""
    dcp = Path(profile["dcp"])
    bit = bitstream or Path(profile["bit"])
    info: Dict[str, Any] = {
        "report_prefix": profile["report_prefix"],
        "expected_clock_period_ns": profile["clock_period_ns"],
        "expected_frequency_mhz": profile["frequency_mhz"],
        "expected_wns_ns": profile["wns_ns"],
        "dcp_path": _path_for_artifact(dcp),
        "dcp_exists": dcp.is_file(),
        "dcp_sha256": _sha256_file(dcp),
        "bit_path": _path_for_artifact(bit),
        "bit_exists": bit.is_file(),
        "bit_sha256": _sha256_file(bit),
        "write_bitstream_attempted": False,
        "write_bitstream_ok": None,
        "write_bitstream_log_tail": None,
        "diagnostic": None,
    }
    if bit.is_file():
        info["diagnostic"] = "bitstream_present"
        return info
    if not dcp.is_file():
        info["diagnostic"] = (
        "missing_bitstream_and_dcp: design-space flow is route-only "
            "(scripts/synth_design_space_point.tcl); no top_routed.dcp found"
        )
        return info
    if not write_bit:
        info["diagnostic"] = (
            "missing_bitstream: routed DCP exists but write_bitstream not requested "
            "(pass --write-bitstream). Design-space points do not emit .bit by default."
        )
        return info

    vivado = shutil.which("vivado") or os.environ.get("VIVADO_BAT")
    if not vivado:
        candidates = [
            Path(r"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"),
            Path(r"C:\Xilinx\Vivado\2025.2\bin\vivado.bat"),
        ]
        for c in candidates:
            if c.is_file():
                vivado = str(c)
                break
    if not vivado:
        info["diagnostic"] = "missing_bitstream: Vivado not found to write_bitstream from DCP"
        return info

    info["write_bitstream_attempted"] = True
    tcl = REPO / "tools" / "write_bitstream_from_dcp.tcl"
    cmd = [
        vivado,
        "-mode",
        "batch",
        "-source",
        str(tcl),
        "-tclargs",
        "dcp",
        str(dcp),
        "bit_out",
        str(bit),
    ]
    log_path = REPO / "build" / "logs" / f"{profile['report_prefix']}_write_bitstream.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    info["write_bitstream_ok"] = proc.returncode == 0 and bit.is_file()
    info["bit_exists"] = bit.is_file()
    info["bit_sha256"] = _sha256_file(bit)
    info["write_bitstream_log_tail"] = (proc.stdout or proc.stderr or "")[-2000:]
    if info["write_bitstream_ok"]:
        info["diagnostic"] = "bitstream_written_from_routed_dcp"
    else:
        info["diagnostic"] = f"write_bitstream_failed_rc={proc.returncode}"
    return info


def _pack_mem_words(program: bytes) -> List[str]:
    words = []
    for i in range(0, len(program), 2):
        w = int.from_bytes(program[i : i + 2], "little")
        words.append(f"{w:04x}")
    return words


def build_onboard_mnist_program(
    *,
    array_size: int,
    buffer_size: int,
    prog_depth: int,
    compute_data_width: int,
    sample_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Fused MNIST 64→64→10 INT8 program matched to shipping array generics.

    Uses a deterministic synthetic INT8 weight/activation set derived from the
    same 8×8-downsampled MNIST topology as ``run_mnist_utpu_demo`` so the
    program fits the Artix shipping bitstream (N=8 INT8), not the historical
    N=16 INT4 sim vectors.
    """
    # Address width must cover BUFFER_SIZE.
    address_width = int(np.ceil(np.log2(max(buffer_size, 2))))
    cfg = IsaConfig(address_width=address_width, compute_data_width=compute_data_width)

    label = None
    if sample_index is None:
        rng = np.random.default_rng(20260805)
        q1 = rng.integers(-7, 8, size=(64, 64), dtype=np.int8)
        q2 = rng.integers(-7, 8, size=(10, 64), dtype=np.int8)
        xq = rng.integers(-7, 8, size=(1, 64), dtype=np.int8)
    else:
        xte8, yte, q1, q2 = _boolean_mnist_assets()
        if not 0 <= int(sample_index) < len(xte8):
            raise IndexError(f"MNIST test sample index out of range: {sample_index}")
        from run_mnist_utpu_demo import _quantize_input_vector
        xq, _ = _quantize_input_vector(xte8[int(sample_index)])
        xq = xq.astype(np.int8).reshape(1, -1)
        label = int(yte[int(sample_index)])

    lowered = lower_fused_mlp_program_utpu(
        fc1_weights_int4=q1,
        fc2_weights_int4=q2,
        input_activations_int4=xq,
        residual_input_int4=None,
        array_size=array_size,
        fc1_apply_relu=True,
        fc2_apply_relu=False,
        apply_quant=True,
        result_addr=ProgramLoader.BUFFER_SECTION_C,
        residual_addr=ProgramLoader.BUFFER_SECTION_D,
        num_pe=1,
        cfg=cfg,
    )
    program = bytes(lowered["program"])
    program_words = len(program) // 2
    sim = simulate_program_bytes(
        program,
        array_size=array_size,
        buffer_size=buffer_size,
        cfg=cfg,
        accumulator_data_width=32,
    )
    expected = bytes(int(v) & 0xFF for v in sim.fetch_bytes)
    output_values = [int(v) - 256 if int(v) >= 128 else int(v) for v in expected[:10]]
    upload = serialize_program_upload(program)
    fits = program_words <= prog_depth and program_words <= UPLOAD_LEN_MAX
    return {
        "name": "fused_mnist_64x64x10_int8_shipping",
        "sample_index": sample_index,
        "label": label,
        "predicted_class": int(np.argmax(output_values)),
        "topology": {"in": 64, "hidden": 64, "out": 10},
        "array_size": array_size,
        "buffer_size": buffer_size,
        "prog_depth": prog_depth,
        "compute_data_width": compute_data_width,
        "address_width": address_width,
        "program_words": program_words,
        "fits_prog_depth": bool(program_words <= prog_depth),
        "fits_upload_len_max": bool(program_words <= UPLOAD_LEN_MAX),
        "fits_shipping": bool(fits),
        "program_sha256": _sha256_bytes(program),
        "upload_sha256": _sha256_bytes(upload),
        "expected_uart_bytes": list(expected),
        "expected_uart_hex": expected.hex(),
        "program": program,
        "upload_bytes": upload,
        "cfg": {
            "address_width": address_width,
            "compute_data_width": compute_data_width,
        },
    }


@lru_cache(maxsize=1)
def _boolean_mnist_assets() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the committed deterministic MNIST MLP and return test inputs/INT8 weights."""
    from run_mnist_utpu_demo import (
        _load_and_downsample,
        _quantize_weights_per_row,
        _train_float_model,
    )

    xtr, ytr, xte, yte = _load_and_downsample()
    model = _train_float_model(xtr, ytr, xte, yte)
    q1, _ = _quantize_weights_per_row(model.fc1.weight.detach().cpu().numpy())
    q2, _ = _quantize_weights_per_row(model.fc2.weight.detach().cpu().numpy())
    # Keep the 8-bit output tensor in range while retaining the trained
    # classifier's sign/relative structure; the exact scaled weights are what
    # both the ISA oracle and board runner upload and compare.
    q2 = np.rint(q2.astype(np.float32) / 4.0).astype(np.int8)
    return xte, yte, q1, q2


def build_boolean_mnist_sample_programs(
    *, array_size: int, buffer_size: int, prog_depth: int,
    compute_data_width: int, sample_count: int = 10, candidate_limit: int = 512,
) -> List[Dict[str, Any]]:
    """Choose the first real test examples with at least 6/10 non-clipped logits."""
    selected: List[Dict[str, Any]] = []
    for index in range(candidate_limit):
        program = build_onboard_mnist_program(
            array_size=array_size,
            buffer_size=buffer_size,
            prog_depth=prog_depth,
            compute_data_width=compute_data_width,
            sample_index=index,
        )
        logits = program["expected_uart_bytes"][:10]
        non_saturated = sum(int(v) not in (0x7F, 0x80) for v in logits)
        if non_saturated >= 6:
            program["non_saturated_output_count"] = non_saturated
            selected.append(program)
            if len(selected) == sample_count:
                return selected
    raise RuntimeError(
        f"found only {len(selected)}/{sample_count} MNIST test samples with >=6/10 "
        f"non-saturated outputs in the first {candidate_limit} cases"
    )


def _stage_evidence(
    *,
    program_info: Dict[str, Any],
    bitstream_info: Dict[str, Any],
    transcript: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Dict[str, str]:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    mem_path = EVIDENCE_DIR / "onboard_mnist_program.mem"
    mem_path.write_text(
        "\n".join(_pack_mem_words(program_info["program"])) + "\n",
        encoding="utf-8",
    )
    paths["program_mem"] = str(mem_path.relative_to(REPO)).replace("\\", "/")

    prog_bin = EVIDENCE_DIR / "onboard_mnist_program.bin"
    prog_bin.write_bytes(program_info["program"])
    paths["program_bin"] = str(prog_bin.relative_to(REPO)).replace("\\", "/")

    upload_bin = EVIDENCE_DIR / "onboard_uart_upload.bin"
    upload_bin.write_bytes(program_info["upload_bytes"])
    paths["upload_bin"] = str(upload_bin.relative_to(REPO)).replace("\\", "/")

    expected_path = EVIDENCE_DIR / "onboard_expected_uart.json"
    expected_path.write_text(
        json.dumps(
            {
                "expected_uart_bytes": program_info["expected_uart_bytes"],
                "program_sha256": program_info["program_sha256"],
                "program_words": program_info["program_words"],
                "samples": program_info.get("sample_summaries", []),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["expected_uart_json"] = str(expected_path.relative_to(REPO)).replace("\\", "/")

    transcript_path = EVIDENCE_DIR / "uart_transcript.json"
    transcript_path.write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")
    paths["uart_transcript"] = str(transcript_path.relative_to(REPO)).replace("\\", "/")

    env_path = EVIDENCE_DIR / "host_environment.json"
    env_path.write_text(json.dumps(artifact.get("host_environment"), indent=2) + "\n", encoding="utf-8")
    paths["host_environment"] = str(env_path.relative_to(REPO)).replace("\\", "/")

    bit_meta = EVIDENCE_DIR / "bitstream_provenance.json"
    bit_meta.write_text(json.dumps(bitstream_info, indent=2) + "\n", encoding="utf-8")
    paths["bitstream_provenance"] = str(bit_meta.relative_to(REPO)).replace("\\", "/")

    art_copy = EVIDENCE_DIR / "onboard_execution.json"
    # Strip raw program bytes from the evidence copy if present
    slim = dict(artifact)
    slim.pop("_program_bytes", None)
    art_copy.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    paths["onboard_execution_copy"] = str(art_copy.relative_to(REPO)).replace("\\", "/")
    return paths


def _try_flash(bitstream: Path, do_flash: bool, board_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "attempted": False,
        "ok": None,
        "method": "vivado_hw_manager",
        "board": board_name,
        "diagnostic": None,
        "log_tail": None,
    }
    if not do_flash:
        out["diagnostic"] = "flash_not_requested (pass --flash)"
        return out
    if not bitstream.is_file():
        out["diagnostic"] = "flash_skipped_missing_bitstream"
        return out

    vivado = shutil.which("vivado")
    if not vivado:
        for c in (
            Path(r"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"),
            Path(r"C:\Xilinx\Vivado\2025.2\bin\vivado.bat"),
        ):
            if c.is_file():
                vivado = str(c)
                break
    if not vivado:
        out["diagnostic"] = "vivado_not_found_for_flash"
        return out

    # Minimal program-only Tcl using the prebuilt .bit.
    tcl_body = f"""
open_hw_manager
connect_hw_server
open_hw_target
set hw_device [lindex [get_hw_devices] 0]
current_hw_device $hw_device
refresh_hw_device $hw_device
set_property PROGRAM.FILE {{{bitstream.as_posix()}}} $hw_device
program_hw_devices $hw_device
puts "PROGRAM_OK"
"""
    tcl_path = REPO / "build" / "logs" / "onboard_program_device.tcl"
    tcl_path.parent.mkdir(parents=True, exist_ok=True)
    tcl_path.write_text(tcl_body, encoding="utf-8")
    out["attempted"] = True
    proc = subprocess.run(
        [vivado, "-mode", "batch", "-source", str(tcl_path)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (REPO / "build" / "logs" / "onboard_program_device.log").write_text(log, encoding="utf-8")
    out["ok"] = proc.returncode == 0 and "PROGRAM_OK" in log
    out["log_tail"] = log[-2000:]
    out["diagnostic"] = "flash_ok" if out["ok"] else f"flash_failed_rc={proc.returncode}"
    return out


def _try_uart_run(
    *,
    port: Optional[str],
    baud: int,
    program: bytes,
    expected: bytes,
    timeout_s: float,
    board_label: str,
) -> Dict[str, Any]:
    transcript: Dict[str, Any] = {
        "attempted": False,
        "port": port,
        "baud": baud,
        "protocol": {
            "MAGIC_UPLOAD": MAGIC_UPLOAD,
            "MAGIC_START": MAGIC_START,
            "length_endianness": "little",
            "length_field_bytes": 2,
            "upload_len_max": UPLOAD_LEN_MAX,
        },
        "tx_upload_hex": None,
        "tx_start_hex": f"{MAGIC_START:02x}",
        "rx_bytes": [],
        "rx_hex": "",
        "rx_byte_count": 0,
        "expected_byte_count": len(expected),
        "bit_exact_vs_isa_sim": False,
        "diagnostic": None,
        "error": None,
    }
    ports = _list_serial_ports()
    transcript["ports_seen"] = ports
    if not port:
        if len(ports) == 1 and "device" in ports[0]:
            port = ports[0]["device"]
            transcript["port"] = port
        else:
            transcript["diagnostic"] = (
                "no_serial_port: pyserial saw no COM devices. "
                f"Connect {board_label} USB-UART and re-run with --port COMx."
            )
            return transcript

    try:
        from uart_driver import UARTDriver
        from program_loader import ProgramLoader as PL
    except ImportError as exc:
        transcript["diagnostic"] = f"import_failed: {exc}"
        return transcript

    transcript["attempted"] = True
    upload = serialize_program_upload(program)
    transcript["tx_upload_hex"] = upload.hex()
    try:
        with UARTDriver(port, baud=baud, timeout=timeout_s, write_timeout=timeout_s) as uart:
            loader = PL(uart=uart, verbose=True)
            uart.flush_input()
            # Brief settle after potential bitstream config
            time.sleep(0.2)
            loader.uploadProgram(program)
            time.sleep(0.05)
            loader.start()
            # Capture expected output bytes (FETCH stream on TX)
            captured = uart.receive_exact(len(expected), timeout=timeout_s)
            transcript["rx_bytes"] = list(captured)
            transcript["rx_hex"] = captured.hex()
            transcript["rx_byte_count"] = len(captured)
            if len(captured) != len(expected):
                transcript["diagnostic"] = (
                    f"uart_short_read: got {len(captured)}/{len(expected)} bytes "
                    "(baud mismatch, reset sequencing, MAGIC framing, or program not running)"
                )
            elif captured == expected:
                transcript["bit_exact_vs_isa_sim"] = True
                transcript["diagnostic"] = "bit_exact_pass"
            else:
                transcript["diagnostic"] = (
                    "uart_bytes_mismatch: length matched but payload differs from ISA sim "
                    "(endianness / datapath / wrong bitstream)"
                )
                # first differing index
                for i, (a, b) in enumerate(zip(captured, expected)):
                    if a != b:
                        transcript["first_mismatch_index"] = i
                        transcript["first_mismatch"] = {"got": a, "expected": b}
                        break
    except Exception as exc:  # noqa: BLE001 — board bring-up must record any failure
        transcript["error"] = f"{type(exc).__name__}: {exc}"
        transcript["diagnostic"] = f"uart_exception: {transcript['error']}"
    return transcript


def build_artifact(args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "model", None):
        return build_buffer_resident_model_artifact(args)
    profile = BOARD_PROFILES[args.board]
    board = profile["board_config"]
    shipping = _load_shipping_point() if args.board == "arty_a7_100t" else {}
    # Boolean is deliberately fixed to its mb4 synthesis point.  Legacy Arty
    # behavior keeps using the saved shipping-point fields when present.
    buffer_size = int(shipping.get("BUFFER_SIZE") or board.buffer_size)
    array_size = int(shipping.get("ARRAY_SIZE") or board.array_size)
    prog_depth = int(shipping.get("PROG_DEPTH") or board.prog_depth)
    compute_dw = int(shipping.get("COMPUTE_DATA_WIDTH") or board.data_width_bits)

    diagnostics: List[str] = []
    requested_bitstream = Path(args.bitstream).expanduser().resolve() if args.bitstream else None
    bitstream_info = _ensure_bitstream(
        profile=profile,
        bitstream=requested_bitstream,
        write_bit=bool(args.write_bitstream),
    )
    if bitstream_info.get("diagnostic"):
        diagnostics.append(str(bitstream_info["diagnostic"]))

    program_info = build_onboard_mnist_program(
        array_size=array_size,
        buffer_size=buffer_size,
        prog_depth=prog_depth,
        compute_data_width=compute_dw,
    )
    sample_programs: List[Dict[str, Any]] = []
    if args.board == "boolean_xc7s50":
        sample_programs = build_boolean_mnist_sample_programs(
            array_size=array_size,
            buffer_size=buffer_size,
            prog_depth=prog_depth,
            compute_data_width=compute_dw,
        )
        program_info = sample_programs[0]
        program_info["sample_summaries"] = [
            {
                "sample_index": p["sample_index"],
                "label": p["label"],
                "predicted_class": p["predicted_class"],
                "non_saturated_output_count": p["non_saturated_output_count"],
                "expected_uart_bytes": p["expected_uart_bytes"],
                "program_sha256": p["program_sha256"],
            }
            for p in sample_programs
        ]
        program_info["expected_uart_bytes"] = [
            byte for p in sample_programs for byte in p["expected_uart_bytes"]
        ]
        program_info["expected_uart_hex"] = bytes(program_info["expected_uart_bytes"]).hex()
    if not program_info["fits_shipping"]:
        diagnostics.append(
            f"program_does_not_fit: words={program_info['program_words']} "
            f"prog_depth={prog_depth} upload_len_max={UPLOAD_LEN_MAX}"
        )

    flash_info = _try_flash(
        requested_bitstream or Path(profile["bit"]),
        do_flash=bool(args.flash),
        board_name=profile["label"],
    )
    if flash_info.get("diagnostic"):
        diagnostics.append(str(flash_info["diagnostic"]))

    uart_info: Dict[str, Any] = {
        "attempted": False,
        "diagnostic": "uart_skipped_pending_prereqs",
    }
    can_uart = (
        program_info["fits_shipping"]
        and (bitstream_info.get("bit_exists") or not args.require_bitstream_for_uart)
    )
    if args.uart and can_uart:
        if sample_programs:
            sample_results = []
            for sample in sample_programs:
                one = _try_uart_run(
                    port=args.port,
                    baud=int(args.baud),
                    program=sample["program"],
                    expected=bytes(sample["expected_uart_bytes"]),
                    timeout_s=float(args.timeout),
                    board_label=str(profile["label"]),
                )
                sample_results.append({
                    "sample_index": sample["sample_index"],
                    "label": sample["label"],
                    "predicted_class": sample["predicted_class"],
                    "non_saturated_output_count": sample["non_saturated_output_count"],
                    "expected_uart_bytes": sample["expected_uart_bytes"],
                    "captured_uart_bytes": one.get("rx_bytes", []),
                    "bit_exact_vs_isa_sim": one.get("bit_exact_vs_isa_sim") is True,
                    "diagnostic": one.get("diagnostic"),
                    "first_mismatch": one.get("first_mismatch"),
                })
                if not args.port and one.get("port"):
                    args.port = one["port"]
            all_pass = all(row["bit_exact_vs_isa_sim"] for row in sample_results)
            uart_info = {
                "attempted": any(row.get("diagnostic") != "no_serial_port" for row in sample_results),
                "port": args.port,
                "baud": int(args.baud),
                "ports_seen": _list_serial_ports(),
                "per_sample": sample_results,
                "expected_bytes": sum(len(row["expected_uart_bytes"]) for row in sample_results),
                "rx_bytes": [v for row in sample_results for v in row["captured_uart_bytes"]],
                "rx_byte_count": sum(len(row["captured_uart_bytes"]) for row in sample_results),
                "bit_exact_vs_isa_sim": all_pass,
                "diagnostic": "all_10_samples_bit_exact" if all_pass else "one_or_more_sample_runs_failed",
            }
        else:
            uart_info = _try_uart_run(
                port=args.port,
                baud=int(args.baud),
                program=program_info["program"],
                expected=bytes(program_info["expected_uart_bytes"]),
                timeout_s=float(args.timeout),
                board_label=str(profile["label"]),
            )
    elif args.uart and not program_info["fits_shipping"]:
        uart_info = {
            "attempted": False,
            "diagnostic": "uart_skipped_program_does_not_fit",
        }
    elif args.uart and not bitstream_info.get("bit_exists"):
        uart_info = {
            "attempted": False,
            "diagnostic": "uart_skipped_missing_bitstream",
            "ports_seen": _list_serial_ports(),
        }
    else:
        uart_info = {
            "attempted": False,
            "diagnostic": "uart_not_requested (pass --uart [--port COMx])",
            "ports_seen": _list_serial_ports(),
        }
    if uart_info.get("diagnostic"):
        diagnostics.append(str(uart_info["diagnostic"]))

    bit_exact = bool(uart_info.get("bit_exact_vs_isa_sim"))
    if bit_exact:
        status = "hardware_bit_exact"
        on_silicon = "hardware"
    elif uart_info.get("attempted"):
        status = "hardware_attempt_failed"
        on_silicon = "attempted"
    elif not bitstream_info.get("bit_exists"):
        status = "blocked_missing_bitstream"
        on_silicon = "not_attempted"
    elif not any(p.get("device") for p in (uart_info.get("ports_seen") or []) if isinstance(p, dict)):
        status = "blocked_no_serial_port"
        on_silicon = "not_attempted"
    else:
        status = "scaffolded_not_executed"
        on_silicon = "not_attempted"

    program_public = {
        k: v
        for k, v in program_info.items()
        if k not in ("program", "upload_bytes")
    }

    artifact: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": status,
        "on_silicon": {
            "status": on_silicon,
            "bit_exact_vs_isa_sim": bit_exact,
            "claim_rule": (
                "Do not claim board execution until status==hardware_bit_exact "
                "and test_onboard_execution_replay.py passes on the staged transcript."
            ),
        },
        "scope": {
            "workload": "fused_mnist_64x64x10_int8",
            "part": profile["part"],
            "board": profile["label"],
            "board_key": args.board,
            "shipping_report_prefix": profile["report_prefix"],
            "clock_period_ns": shipping.get("clock_period_ns", profile["clock_period_ns"]),
            "frequency_mhz": shipping.get("frequency_mhz_constraint", profile["frequency_mhz"]),
            "wns_ns": (shipping.get("timing") or {}).get("wns_ns", profile["wns_ns"]),
            "ARRAY_SIZE": array_size,
            "COMPUTE_DATA_WIDTH": compute_dw,
            "BUFFER_SIZE": buffer_size,
            "PROG_DEPTH": prog_depth,
            "MAX_BATCH_COUNT": profile["max_batch_count"],
            "UART_BAUD_host": int(args.baud),
            "UART_BAUD_rtl_default": 115200,
        },
        "host_environment": _host_env(),
        "bitstream": bitstream_info,
        "program": program_public,
        "flash": flash_info,
        "uart": {
            k: v
            for k, v in uart_info.items()
            if k != "tx_upload_hex" or True
        },
        "diagnostics": diagnostics,
        "evidence_dir": str(EVIDENCE_DIR.relative_to(REPO)).replace("\\", "/"),
    }

    transcript = {
        "generated_at_utc": artifact["generated_at_utc"],
        "status": status,
        "program_sha256": program_info["program_sha256"],
        "bitstream_sha256": bitstream_info.get("bit_sha256"),
        "expected_uart_bytes": program_info["expected_uart_bytes"],
        "captured_uart_bytes": uart_info.get("rx_bytes") or [],
        "bit_exact_vs_isa_sim": bit_exact,
        "uart": uart_info,
        "host_environment": artifact["host_environment"],
    }
    paths = _stage_evidence(
        program_info=program_info,
        bitstream_info=bitstream_info,
        transcript=transcript,
        artifact=artifact,
    )
    artifact["evidence_files"] = paths
    return artifact


def build_buffer_resident_model_artifact(args: argparse.Namespace) -> Dict[str, Any]:
    """Build an A5 resident-weight run report for one requested MLP model.

    On silicon this uploads the weight image once. The current RTL returns to
    UPLOAD_HEADER after A3 rearm, so it reloads the control program for each
    image before filling that image and issuing A2. No hardware status is
    granted unless every fetched byte for every requested image matched ISA.
    """
    import torch
    from board_config import BoardConfig
    from buffer_resident_deployment import (
        deploy_model, encode_buffer_fill, integer_reference, quantize_input,
        run_isa_inference,
    )
    from isa_simulator import UTPUISASimulator
    from run_buffer_resident_models import (
        FashionMLP, _load_dataset, load_fashion, train_fashion,
    )
    from run_real_model_accelerator import FloatMLP, _train_float_model

    if args.board != "boolean_xc7s50":
        raise ValueError("--model deployments are currently supported only on --board boolean_xc7s50")
    if args.model == "mnist_mlp":
        xtr, ytr, xte, yte = _load_dataset()
        model = _train_float_model(xtr, ytr)
    else:
        xtr, ytr, xte, yte = load_fashion(Path(args.fashion_data))
        model = train_fashion(xtr, ytr)
    n = min(len(xte), int(args.images))
    if n <= 0:
        raise ValueError("--images must be positive")
    board = BoardConfig.boolean_xc7s50()
    deployment = deploy_model(model, board, torch.from_numpy(xtr))
    sim = UTPUISASimulator(array_size=board.array_size, buffer_size=board.buffer_size,
                           cfg=deployment.config, accumulator_data_width=32)
    for address, word in enumerate(deployment.weight_image_words):
        sim.buffer[address] = int(word)

    predictions: List[int] = []
    per_image: List[Dict[str, Any]] = []
    cycle_counts: List[int] = []
    expected_by_image: List[Tuple[bytes, int]] = []
    sim_start = time.perf_counter()
    for i in range(n):
        image_q = quantize_input(xte[i], deployment.input_scale)
        integer_expected = integer_reference(image_q, deployment)
        isa_values, cycles = run_isa_inference(image_q, deployment, sim)
        if not np.array_equal(isa_values, integer_expected):
            raise RuntimeError(f"ISA vs independent integer mismatch before board run at image {i}")
        padded_bytes = bytearray(max(16, ((deployment.output_features + 7) // 8) * 8))
        padded_bytes[:deployment.output_features] = bytes(int(x) & 0xff for x in integer_expected)
        expected_by_image.append((bytes(padded_bytes), cycles))
        predictions.append(int(np.argmax(integer_expected)))
        cycle_counts.append(cycles)
    sim_seconds = time.perf_counter() - sim_start
    model.eval()
    with torch.no_grad():
        float_predictions = np.concatenate([
            model(torch.from_numpy(xte[start:min(n, start + 512)])).argmax(1).cpu().numpy()
            for start in range(0, n, 512)
        ])

    uart_rows: List[Dict[str, Any]] = []
    device_program_cycles: List[int] = []
    uart_attempted = False
    host_started = time.perf_counter()
    port = args.port
    if args.uart:
        bitstream_path = Path(args.bitstream).expanduser().resolve() if args.bitstream else None
        if bitstream_path is None or not bitstream_path.is_file():
            uart_rows.append({"passed": False, "diagnostic": "missing bitstream path; pass --bitstream for the configured Boolean image"})
        else:
            ports = _list_serial_ports()
            if not port and len(ports) == 1:
                port = ports[0].get("device")
            if not port:
                uart_rows.append({"passed": False, "diagnostic": "no serial port selected; pass --port COMx"})
            else:
              from uart_driver import UARTDriver
              import struct
              try:
                uart_attempted = True
                with UARTDriver(port, baud=int(args.baud), timeout=float(args.timeout),
                                write_timeout=float(args.timeout)) as uart:
                    loader = ProgramLoader(uart=uart, verbose=False, cfg=deployment.config)
                    uart.flush_input()
                    time.sleep(0.2)
                    _send_uart_chunks(uart, encode_buffer_fill(0, deployment.weight_image_words))
                    program_header = bytes((MAGIC_UPLOAD,)) + struct.pack("<H", deployment.control_words)
                    _send_uart_chunks(uart, program_header + deployment.control_program)
                    time.sleep(0.02)
                    for i in range(n):
                        if i:
                            # A6 returns from HALT to WAIT_START while retaining
                            # instruction BRAM, avoiding a retransmit per image.
                            _send_uart_chunks(uart, bytes((MAGIC_REPEAT,)))
                            time.sleep(0.02)
                        image_q = quantize_input(xte[i], deployment.input_scale)
                        _send_uart_chunks(uart, deployment.input_fill_command(image_q))
                        _send_uart_chunks(uart, bytes((MAGIC_START,)))
                        expected, cycles = expected_by_image[i]
                        captured = uart.receive_exact(len(expected), timeout=float(args.timeout))
                        passed = captured == expected
                        counters = loader.readPerfCounters(timeout=float(args.timeout))
                        device_program_cycles.append(int(counters["program_cycles"]))
                        uart_rows.append({
                            "image_index": i,
                            "label": int(yte[i]),
                            "predicted_class": predictions[i],
                            "expected_bytes_hex": expected.hex(),
                            "actual_bytes_hex": captured.hex(),
                            "output_bytes_bit_exact": passed,
                            "device_cycles_from_isa_sim": cycles,
                            "device_program_cycles_from_perf_counter": int(counters["program_cycles"]),
                            "diagnostic": "bit_exact_pass" if passed else "uart_bytes_mismatch_or_short_read",
                        })
                        if not passed:
                            break
              except Exception as exc:  # record board/driver errors in result artifact
                uart_rows.append({"passed": False, "diagnostic": f"{type(exc).__name__}: {exc}"})
    host_seconds = time.perf_counter() - host_started if args.uart else None

    result_bit_exact = bool(args.uart and len(uart_rows) == n and all(row.get("output_bytes_bit_exact") for row in uart_rows))
    selected_bitstream = Path(args.bitstream).expanduser().resolve() if args.bitstream else Path(BOARD_PROFILES[args.board]["bit"])
    bitstream_record = {"path": str(selected_bitstream), "exists": selected_bitstream.is_file(),
                        "sha256": _sha256_file(selected_bitstream) if selected_bitstream.is_file() else None}
    flash_record = _try_flash(selected_bitstream, bool(args.flash), BOARD_PROFILES[args.board]["label"])
    artifact = {
        "schema_version": 1,
        "generated_at_utc": _utc(),
        "status": ("hardware_bit_exact" if result_bit_exact else
                   ("hardware_attempt_failed" if uart_attempted else
                    ("blocked_missing_bitstream" if args.uart and not bitstream_record["exists"] else "scaffolded_not_executed"))),
        "claim_rule": "hardware_bit_exact requires every fetched output byte for every requested image to match the ISA simulator; replay test required before any claim",
        "scope": {"model": args.model, "dataset": "MNIST" if args.model == "mnist_mlp" else "Fashion-MNIST",
                  "images_requested": n, "board": args.board, "part": BOARD_PROFILES[args.board]["part"],
                  "baud": int(args.baud), "weight_fill_once": True, "control_program_uploaded_once": True,
                  "repeat_magic": f"0x{MAGIC_REPEAT:02X}"},
        "bitstream": bitstream_record,
        "flash": flash_record,
        "deployment": {"board_config": board.as_dict(), "weight_words": len(deployment.weight_image_words),
                       "control_words": deployment.control_words,
                       "layer_shapes": [{"name": l.name, "input_features": l.input_features,
                                         "output_features": l.output_features, "weight_words": l.weight_words}
                                        for l in deployment.layers],
                       "fit_report": deployment.fit_report},
        "float_accuracy": float(np.mean(float_predictions == yte[:n])),
        "quantized_accuracy": float(np.mean(np.asarray(predictions) == yte[:n])),
        "isa_vs_integer_reference": {"all_images_bit_exact": True, "images": n,
                                      "output_bytes_checked": n * deployment.output_features},
        "device_cycles_per_inference": {"mean": float(np.mean(cycle_counts)),
                                        "min": int(min(cycle_counts)), "max": int(max(cycle_counts)),
                                        "source": "ISA simulator perf/cycle result; device counters require board readback"},
        "board_perf_program_cycles_per_inference": {
            "mean": float(np.mean(device_program_cycles)) if device_program_cycles else None,
            "per_image": device_program_cycles,
            "source": "RTL MAGIC_READ_PERF program_cycles; populated only during --uart board runs",
        },
        "simulation_seconds": sim_seconds,
        "board_host_elapsed_seconds": host_seconds,
        "board_host_seconds_per_image": (float(host_seconds / n) if host_seconds is not None else None),
        "board_host_latency_scope": "includes one weight fill, one program upload, per-image input fill/start/readback, perf-counter reads, and UART chunk pacing",
        "uart": {"attempted": uart_attempted, "port": port, "baud": int(args.baud),
                 "matched_images": sum(bool(row.get("output_bytes_bit_exact")) for row in uart_rows),
                 "per_image": uart_rows},
    }
    validate_buffer_resident_artifact(artifact)
    out = REPO / "bench" / "results" / f"onboard_execution_{args.model}.json"
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    artifact["artifact_path"] = str(out.relative_to(REPO)).replace("\\", "/")
    return artifact


def _send_uart_chunks(uart: Any, data: bytes, chunk_size: int = 128) -> None:
    for start in range(0, len(data), chunk_size):
        chunk = data[start:start + chunk_size]
        written = uart.ser.write(chunk)
        if written != len(chunk):
            raise IOError(f"short UART write: {written}/{len(chunk)} bytes")
        if start + chunk_size < len(data):
            time.sleep(0.015)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--board",
        choices=sorted(BOARD_PROFILES),
        default="arty_a7_100t",
        help="bitstream/build profile (default: arty_a7_100t)",
    )
    parser.add_argument("--model", choices=("mnist_mlp", "fashion_mlp"), default=None,
                        help="run the buffer-resident general MLP deployment path")
    parser.add_argument("--images", type=int, default=10000,
                        help="number of test images for --model (default: full test set)")
    parser.add_argument("--fashion-data", default=str(REPO / "software" / "data" / "fashion-mnist"),
                        help="directory containing official Fashion-MNIST IDX gzip files")
    parser.add_argument(
        "--bitstream",
        default=None,
        help="explicit .bit file; overrides the selected board profile default",
    )
    parser.add_argument("--write-bitstream", action="store_true", help="write .bit from routed DCP")
    parser.add_argument("--flash", action="store_true", help="program the selected board via Vivado hw_server")
    parser.add_argument("--uart", action="store_true", help="upload/run/capture over UART")
    parser.add_argument("--port", default=None, help="COM port (e.g. COM6)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--require-bitstream-for-uart",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-uart-without-bitstream",
        action="store_true",
        help="attempt UART even if shipping .bit is missing (debug only)",
    )
    args = parser.parse_args()
    if args.allow_uart_without_bitstream:
        args.require_bitstream_for_uart = False

    artifact = build_artifact(args)
    out_path = Path(artifact.get("artifact_path", OUT_JSON))
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop any non-serializable leftovers
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "diagnostics": artifact["diagnostics"]}, indent=2))
    print(f"Wrote {out_path}")
    return 0 if artifact["status"] == "hardware_bit_exact" else 2


if __name__ == "__main__":
    raise SystemExit(main())
