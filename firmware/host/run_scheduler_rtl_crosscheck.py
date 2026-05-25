"""Phase 7 remediation P4.1 — scheduler RTL cycle cross-check runner.

Pipeline:

1. Regenerate ``build/test_vectors/scheduler_{naive,sched}.mem`` and
   ``scheduler_expected.svh`` from the Phase 5 simulator's locked
   artifact via :mod:`generate_scheduler_rtl_test_vectors`.
2. Compile ``rtl/tb/tb_scheduler_cycles.sv`` with iverilog if a binary
   is available locally (Windows: ``C:\\iverilog\\bin\\iverilog.exe``;
   Linux/WSL: ``iverilog`` on PATH).
3. Run vvp, parse the test log, and emit
   ``bench/results/scheduler_rtl_crosscheck.json``.

The artifact has two top-level modes:

* ``status="ok"`` — iverilog ran, test produced a TB_RESULT line, and
  all asserted invariants (positive RTL cycle savings, RTL_reduction
  within ``±SCHED_TOL_PERMILLE`` of sim, RTL naive_bytes ===
  RTL scheduled_bytes) passed. Emits the parsed permille / cycle
  counts plus the per-byte naive/scheduled vectors.

* ``status="iverilog_unavailable"`` — no iverilog binary was located,
  so the testbench was not executed. The artifact still records the
  simulator-side test vectors (sim cycles, sim permille, expected
  bytes) so consumers can detect when the file is a stub vs a real run.

The non-headline byte cross-check (RTL vs simulator's expected bytes)
is logged but does *not* gate ``status``. The scheduler's correctness
invariant at the RTL level is RTL_naive === RTL_scheduled (the gating
check); per-shape disagreement between the simulator's ideal-pipeline
bytes and the RTL FSM's bytes is a separate concern tracked outside
the scheduler claim. See ``rtl/tb/tb_scheduler_cycles.sv`` for the
testbench rationale.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_DIR = REPO_ROOT / "firmware" / "host"
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

# Reused so the artifact's `expected_*` block matches what the testbench
# header `define`s exactly (avoids drift between generator + runner).
from generate_scheduler_rtl_test_vectors import (  # noqa: E402
    ARRAY_SIZE,
    INPUT_ADDR,
    IN_FEATURES,
    OUT_FEATURES,
    RESULT_ADDR,
    WEIGHT_ADDR,
    main as _regenerate_vectors,
)
from isa_simulator import simulate_program_bytes  # noqa: E402
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu  # noqa: E402
from scheduler_allocator import lower_blocked_fc_program_scheduled  # noqa: E402

BUILD_DIR = REPO_ROOT / "build"
SIM_OUT_DIR = BUILD_DIR / "sim_iverilog"
TEST_VECTOR_DIR = BUILD_DIR / "test_vectors"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
OUTPUT_JSON = RESULTS_DIR / "scheduler_rtl_crosscheck.json"

DESIGN_FILES = [
    "rtl/tb/xpm_memory_sdpram_stub.sv",
    "rtl/top/top.sv",
    "rtl/memory/instr_bram.sv",
    "rtl/PEArray/pe_controller.sv",
    "rtl/PEArray/pe_array.sv",
    "rtl/PEArray/pe.sv",
    "rtl/quantizer/quantizer.sv",
    "rtl/quantizer/quantizer_array.sv",
    "rtl/LeakyReLU/leaky_relu.sv",
    "rtl/LeakyReLU/leaky_relu_array.sv",
    "rtl/unified_buffer/unified_buffer.sv",
    "rtl/fifo/fifo_rx.sv",
    "rtl/fifo/fifo_tx.sv",
    "rtl/UART/uart.sv",
    "rtl/UART/uart_receiver.sv",
    "rtl/UART/uart_transmitter.sv",
    "rtl/UART/clk_divider.sv",
]
TB_FILE = "rtl/tb/tb_scheduler_cycles.sv"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _resolve_iverilog() -> Tuple[Optional[str], Optional[str]]:
    """Return (iverilog, vvp) absolute paths or (None, None)."""
    candidates = [
        (r"C:\iverilog\bin\iverilog.exe", r"C:\iverilog\bin\vvp.exe"),
        (
            r"C:\Program Files\Icarus Verilog\bin\iverilog.exe",
            r"C:\Program Files\Icarus Verilog\bin\vvp.exe",
        ),
    ]
    for iv, vv in candidates:
        if os.path.exists(iv) and os.path.exists(vv):
            return iv, vv
    iv_path = shutil.which("iverilog")
    vv_path = shutil.which("vvp")
    if iv_path and vv_path:
        return iv_path, vv_path
    return None, None


def _simulator_reference() -> Dict[str, object]:
    """Recompute the simulator-side reference so the artifact never
    drifts from what the generator wrote into the .svh header.
    """
    import numpy as np

    seed = 0xC0DE + OUT_FEATURES * 31 + IN_FEATURES
    rng = np.random.default_rng(seed)
    w = rng.integers(low=-8, high=8, size=(OUT_FEATURES, IN_FEATURES), dtype=np.int8)
    x = rng.integers(low=-8, high=8, size=IN_FEATURES, dtype=np.int8)
    naive = lower_blocked_fc_program_utpu(
        w, x, OUT_FEATURES, IN_FEATURES, ARRAY_SIZE, False, True,
        WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR,
        prog_depth=1024,
    )
    sched = lower_blocked_fc_program_scheduled(
        w, x, OUT_FEATURES, IN_FEATURES, ARRAY_SIZE, False, True,
        WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR,
    )
    rn = simulate_program_bytes(naive["program"], array_size=ARRAY_SIZE)
    rs = simulate_program_bytes(sched["program"], array_size=ARRAY_SIZE)
    naive_cycles = int(rn.cycle_count_sequential)
    sched_cycles = int(rs.cycle_count_sequential)
    cycles_saved = naive_cycles - sched_cycles
    permille = (
        int(round(1000.0 * cycles_saved / naive_cycles)) if naive_cycles else 0
    )
    return {
        "shape": {"out_features": OUT_FEATURES, "in_features": IN_FEATURES},
        "array_size": ARRAY_SIZE,
        "weight_addr": WEIGHT_ADDR,
        "input_addr": INPUT_ADDR,
        "result_addr": RESULT_ADDR,
        "naive_words": int(naive["program_instruction_words"]),
        "sched_words": int(sched["program_instruction_words"]),
        "naive_cycles": naive_cycles,
        "sched_cycles": sched_cycles,
        "cycles_saved": cycles_saved,
        "reduction_permille": permille,
        "fetch_bytes_n": len(rn.fetch_bytes),
        "expected_fetch_bytes": [int(b) for b in rn.fetch_bytes],
        "fetch_bytes_invariant_simulator": list(rn.fetch_bytes) == list(rs.fetch_bytes),
    }


_LOG_RE_RTL_NAIVE = re.compile(
    r"RTL_NAIVE_CYCLES=(\d+) \(sim=(\d+)\) NAIVE_FETCH_N=(\d+) \(sim=(\d+)\)"
)
_LOG_RE_RTL_SCHED = re.compile(
    r"RTL_SCHED_CYCLES=(\d+) \(sim=(\d+)\) SCHED_FETCH_N=(\d+) \(sim=(\d+)\)"
)
_LOG_RE_RTL_REDUCTION = re.compile(
    r"RTL_REDUCTION_PERMILLE=(\d+)\s+SIM_REDUCTION_PERMILLE=(\d+)\s+TOL=(\d+)\s+DIFF=(\d+)"
)
_LOG_RE_ADV_NAIVE = re.compile(
    r"ADVISORY: RTL_naive matches sim on (\d+) / (\d+) bytes"
)
_LOG_RE_ADV_SCHED = re.compile(
    r"ADVISORY: RTL_sched matches sim on (\d+) / (\d+) bytes"
)
_LOG_RE_DONE = re.compile(r"DONE tests=(\d+) errors=(\d+)")


def _parse_log(text: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {
        "raw_log": text,
        "tb_result_pass": "TB_RESULT: PASS" in text,
    }
    m = _LOG_RE_RTL_NAIVE.search(text)
    if m:
        parsed["rtl_naive_cycles"] = int(m.group(1))
        parsed["sim_naive_cycles_in_header"] = int(m.group(2))
        parsed["rtl_naive_fetch_n"] = int(m.group(3))
        parsed["sim_fetch_n_in_header"] = int(m.group(4))
    m = _LOG_RE_RTL_SCHED.search(text)
    if m:
        parsed["rtl_sched_cycles"] = int(m.group(1))
        parsed["sim_sched_cycles_in_header"] = int(m.group(2))
        parsed["rtl_sched_fetch_n"] = int(m.group(3))
    m = _LOG_RE_RTL_REDUCTION.search(text)
    if m:
        parsed["rtl_reduction_permille"] = int(m.group(1))
        parsed["sim_reduction_permille_in_header"] = int(m.group(2))
        parsed["tol_permille"] = int(m.group(3))
        parsed["diff_permille"] = int(m.group(4))
    m = _LOG_RE_ADV_NAIVE.search(text)
    if m:
        parsed["rtl_naive_bytes_matching_sim"] = int(m.group(1))
        parsed["rtl_naive_bytes_total"] = int(m.group(2))
    m = _LOG_RE_ADV_SCHED.search(text)
    if m:
        parsed["rtl_sched_bytes_matching_sim"] = int(m.group(1))
        parsed["rtl_sched_bytes_total"] = int(m.group(2))
    m = _LOG_RE_DONE.search(text)
    if m:
        parsed["tests"] = int(m.group(1))
        parsed["errors"] = int(m.group(2))
    return parsed


def _run_iverilog_flow(iv_bin: str, vv_bin: str) -> Dict[str, object]:
    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_vvp = SIM_OUT_DIR / "tb_scheduler_cycles.out"
    log_path = SIM_OUT_DIR / "tb_scheduler_cycles.log"
    if out_vvp.exists():
        out_vvp.unlink()
    if log_path.exists():
        log_path.unlink()

    sources = [TB_FILE] + DESIGN_FILES
    compile_cmd = [
        iv_bin, "-g2012", "-DICARUS",
        "-o", str(out_vvp), *sources,
    ]
    compile_proc = subprocess.run(
        compile_cmd, cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )
    if compile_proc.returncode != 0:
        return {
            "iverilog_compile_ok": False,
            "iverilog_compile_stderr": compile_proc.stdout,
        }

    run_proc = subprocess.run(
        [vv_bin, str(out_vvp)], cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )
    log_text = run_proc.stdout
    log_path.write_text(log_text, encoding="utf-8")
    parsed = _parse_log(log_text)
    parsed["iverilog_compile_ok"] = True
    parsed["iverilog_run_returncode"] = int(run_proc.returncode)
    parsed["log_path"] = str(log_path.relative_to(REPO_ROOT))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(OUTPUT_JSON),
        help="Path to write the JSON artifact (default: bench/results/scheduler_rtl_crosscheck.json)",
    )
    parser.add_argument(
        "--skip-iverilog", action="store_true",
        help="Emit a stub artifact (status=iverilog_unavailable) regardless of binary presence; useful for hostless CI",
    )
    args = parser.parse_args()

    OUTPUT_JSON_PATH = Path(args.output)
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    rc = _regenerate_vectors()
    if rc != 0:
        raise RuntimeError(f"generate_scheduler_rtl_test_vectors failed with rc={rc}")

    sim_ref = _simulator_reference()

    artifact: Dict[str, object] = {
        "version": 1,
        "generated_at_utc": _now_iso(),
        "git_sha": _git_sha(),
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        },
        "tolerance_permille": 20,
        "expected": sim_ref,
        "methodology": {
            "summary": (
                "RTL cycle cross-check of the Phase 5 scheduler at "
                f"(M={OUT_FEATURES}, K={IN_FEATURES}). The simulator's "
                "1-cycle-per-op model and the RTL FSM's multi-cycle "
                "STORE/FETCH paths produce different absolute cycle counts; "
                "we cross-check the *percentage* cycle reduction (within "
                "±20 permille = ±2.0%) and verify the scheduler's "
                "RTL-side bit-exactness invariant (RTL_naive === "
                "RTL_scheduled fetch_bytes). The simulator-vs-RTL byte "
                "agreement is recorded as an advisory metric only."
            ),
            "headline_assertions": [
                "RTL scheduled cycles strictly less than naive",
                "|RTL_reduction_permille - sim_reduction_permille| <= 20",
                "RTL_naive fetch_bytes === RTL_scheduled fetch_bytes",
                "RTL_naive fetch_n == sim fetch_n; RTL_sched fetch_n == sim fetch_n",
            ],
            "advisory_metrics": [
                "RTL_naive byte agreement with simulator (out of sim fetch_n)",
                "RTL_sched byte agreement with simulator (out of sim fetch_n)",
            ],
            "tools": {
                "rtl_generator": "firmware/host/generate_scheduler_rtl_test_vectors.py",
                "testbench": "rtl/tb/tb_scheduler_cycles.sv",
                "simulator": "firmware/host/isa_simulator.py (1 cycle/op; STORE/BSTORE 2+N)",
                "scheduler": "firmware/host/scheduler_allocator.py (lower_blocked_fc_program_scheduled)",
                "naive_baseline": "firmware/host/lowering_blocked_fc_utpu.py (lower_blocked_fc_program_utpu)",
            },
        },
    }

    iv_bin, vv_bin = _resolve_iverilog()
    if args.skip_iverilog or iv_bin is None or vv_bin is None:
        artifact["status"] = "iverilog_unavailable"
        artifact["iverilog_resolved"] = {
            "iverilog": iv_bin,
            "vvp": vv_bin,
        }
        artifact["instructions"] = (
            "Install iverilog locally (Windows: bleyer.org/icarus; Linux: "
            "apt-get install -y iverilog) and rerun this script. The "
            "test vectors under build/test_vectors/scheduler_*.mem and "
            "the testbench at rtl/tb/tb_scheduler_cycles.sv are already "
            "in place; no other inputs change between stub and populated "
            "modes."
        )
        OUTPUT_JSON_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[run_scheduler_rtl_crosscheck] iverilog unavailable; wrote stub to {OUTPUT_JSON_PATH}")
        return 0

    iverilog_result = _run_iverilog_flow(iv_bin, vv_bin)
    artifact["iverilog"] = {
        "iverilog_bin": iv_bin,
        "vvp_bin": vv_bin,
    }
    artifact["iverilog_result"] = iverilog_result

    headline_ok = (
        bool(iverilog_result.get("iverilog_compile_ok"))
        and bool(iverilog_result.get("tb_result_pass"))
        and int(iverilog_result.get("errors", 1)) == 0
    )
    artifact["status"] = "ok" if headline_ok else "failed"

    if headline_ok:
        artifact["headline"] = {
            "rtl_naive_cycles": iverilog_result.get("rtl_naive_cycles"),
            "rtl_sched_cycles": iverilog_result.get("rtl_sched_cycles"),
            "rtl_cycles_saved": (
                int(iverilog_result.get("rtl_naive_cycles", 0))
                - int(iverilog_result.get("rtl_sched_cycles", 0))
            ),
            "rtl_reduction_permille": iverilog_result.get("rtl_reduction_permille"),
            "sim_reduction_permille": iverilog_result.get("sim_reduction_permille_in_header"),
            "diff_permille": iverilog_result.get("diff_permille"),
            "tol_permille": iverilog_result.get("tol_permille"),
            "scheduler_invariant_holds": True,
        }

    OUTPUT_JSON_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[run_scheduler_rtl_crosscheck] status={artifact['status']} -> {OUTPUT_JSON_PATH}"
    )
    return 0 if headline_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
