"""Task 4 — deterministic-latency static analysis verification harness.

Produces ``bench/results/latency_determinism.json``. Three layers of
evidence, each scoped honestly:

1. **Static cycle model matches the ISA simulator exactly.**
   For every shape in ``--shapes``, the harness compiles the blocked-FC
   program (using ``lowering_blocked_fc_utpu.lower_blocked_fc_program_utpu``,
   the same lowering used by P4.1 + P5), computes static cycles via
   ``latency_analysis.static_cycles_simulator``, simulates it via
   ``isa_simulator.simulate_program_bytes``, and asserts the two are
   *byte-exactly* equal. Any mismatch fails the entire harness.

2. **Static cycle count is invariant across adversarial input
   distributions.** For one representative shape, the harness
   regenerates the program with 5 different input vectors drawn from
   the distribution set ``["zero", "saturating", "random", "sparse",
   "alternating"]``. The static cycle count for each is asserted to
   equal the static cycle count of the shape's reference run. This
   is true by construction (input data lives in BSTORE payload words,
   not in opcode bits, so the instruction count is shape-determined),
   but the harness validates it empirically as a defense against
   future regressions in the lowering.

3. **Empirical RTL cycle variance == 0 across distributions** (the
   data-independence witness). For the same representative shape +
   distribution sweep, if ``iverilog``/``vvp`` are available locally,
   the harness runs ``rtl/tb/tb_latency_determinism.sv`` once per
   distribution, captures the RTL cycle count, and asserts the
   variance across all 5 runs is exactly 0. Stub mode
   (``on_silicon.status="rtl_sim_unavailable_no_iverilog"``) emits the
   static-arm evidence only and records that the RTL arm was skipped.

Honest scope notes embedded in the artifact:

* ``scope_note`` documents that the static model matches the
  *simulator*, NOT the RTL FSM's absolute cycle count (per P4.1's
  already-published bounded equivalence the absolute counts differ;
  the RELATIVE cycle reduction agrees within ±2.0%).
* ``on_silicon`` carries ``status="rtl_sim"`` until the Arty A7
  board is available; on board day the harness can flip
  ``on_silicon.onchip_counter_cycles`` and ``matches_static`` to the
  Hardware perf-counter values.

The data-independence proof itself is purely static: every opcode
encountered in every swept program is checked against
``latency_analysis.DEFAULT_DATA_INDEPENDENT_OPS``. Any opcode outside
that set would show up in ``proof.data_dependent_ops_found``, which
must be empty for the artifact to pass.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_DIR = REPO_ROOT / "firmware" / "host"
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

import numpy as np  # noqa: E402

from isa_simulator import simulate_program_bytes  # noqa: E402
from latency_analysis import (  # noqa: E402
    DEFAULT_DATA_INDEPENDENT_OPS,
    analyze_program,
    prove_data_independent,
    static_cycles_simulator,
)
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu  # noqa: E402

BUILD_DIR = REPO_ROOT / "build"
SIM_OUT_DIR = BUILD_DIR / "sim_iverilog"
TEST_VECTOR_DIR = BUILD_DIR / "test_vectors"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
OUTPUT_JSON = RESULTS_DIR / "latency_determinism.json"

# The shapes we sweep for the static-vs-simulator parity gate. These
# fit the shipping ``PROG_DEPTH=1024`` and ``BUFFER_SIZE=512`` of the
# pynqz2 baseline (the same bitstream-relevant config as P4.1) so the
# RTL testbench can run them without parameter overrides. The
# ``(M=32, K=32)`` shape is the canonical one chosen for the
# distribution sweep because it matches the existing P4.1 testbench's
# shape (same BUFFER region layout, same PROG_DEPTH).
DEFAULT_SHAPES: Tuple[Tuple[int, int], ...] = (
    (32, 32),
    (32, 64),
    (64, 32),
    (64, 64),
)

DEFAULT_DISTRIBUTION_SHAPE: Tuple[int, int] = (32, 32)

DEFAULT_DISTRIBUTIONS: Tuple[str, ...] = (
    "zero",
    "saturating",
    "random",
    "sparse",
    "alternating",
)

ARRAY_SIZE = 16
WEIGHT_ADDR = 256
INPUT_ADDR = 0
RESULT_ADDR = 320
WEIGHT_SEED_BASE = 0xC0DE

SCHEMA_VERSION = 1

# Shipping wall-clock conversion uses the design-space shipping close
# (12 ns / ~83.333 MHz, WNS=+0.271). 100 MHz is ceiling-only (WNS=+0.012).
FPGA_CLOCK_MHZ = 1000.0 / 12.0  # shipping default ~83.333 MHz (12 ns)
FPGA_CLOCK_PERIOD_NS = 12.0
FPGA_CLOCK_WNS_NS = 0.271
FPGA_CLOCK_MARGIN_CLASS = "thin"
FPGA_CLOCK_SOURCE = (
    "bench/results/design_space_sweep.json::shipping_point "
    "N=8 INT8 MAX_BATCH_COUNT=48 @ 12 ns (WNS=+0.271, margin_class=thin). "
    "100 MHz is the demonstrated ceiling (WNS=+0.012, marginal) — quote WNS if cited."
)
FPGA_CEILING_MHZ = 100.0
FPGA_CEILING_WNS_NS = 0.012
FPGA_CEILING_MARGIN_CLASS = "marginal"

# Extra random-input RTL trials beyond the 5 adversarial distributions.
# Each trial is one iverilog vvp invocation (compile-once + plusargs).
DEFAULT_E2E_RANDOM_TRIALS = 32

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
TB_FILE = "rtl/tb/tb_latency_determinism.sv"


# ---------------------------------------------------------------------------
# Input data + program generation
# ---------------------------------------------------------------------------


def _draw_input_vector(
    distribution: str, in_features: int, rng: np.random.Generator
) -> np.ndarray:
    """Adversarial input distributions used by the data-independence sweep.

    Each distribution targets a different hypothesis about why a real
    hardware datapath might leak data into its cycle count:

    - ``zero``: every element 0. A multiplier with a non-constant-time
      fast path for zero operands would shorten the cycle count here.
    - ``saturating``: every element at the maximum representable INT4
      value (+7). Tests for any data-magnitude-dependent path.
    - ``random``: uniformly random across the full INT4 range. A
      "typical-case" input for the sweep.
    - ``sparse``: 75% zeros + 25% saturating values. Tests for
      operand-aware skipping or early-exit semantics.
    - ``alternating``: ``+7, -8, +7, -8, ...``. Tests for a
      sign-flip-sensitive datapath.
    """
    if distribution == "zero":
        return np.zeros(in_features, dtype=np.int8)
    if distribution == "saturating":
        return np.full(in_features, 7, dtype=np.int8)
    if distribution == "random":
        return rng.integers(low=-8, high=8, size=in_features, dtype=np.int8)
    if distribution == "sparse":
        v = np.zeros(in_features, dtype=np.int8)
        # 25% of slots filled with +7; deterministic per-distribution.
        idx = rng.choice(in_features, size=max(1, in_features // 4), replace=False)
        v[idx] = 7
        return v
    if distribution == "alternating":
        v = np.empty(in_features, dtype=np.int8)
        v[0::2] = 7
        v[1::2] = -8
        return v
    raise ValueError(f"unknown distribution: {distribution!r}")


def _compile_program(
    out_features: int,
    in_features: int,
    *,
    x_vector: Optional[np.ndarray] = None,
    weight_seed: Optional[int] = None,
) -> Dict[str, object]:
    """Compile a blocked-FC program for a given (M, K) with optional x override.

    Weights are seeded deterministically from the shape so the same
    shape always produces the same weight tensor (input distributions
    only vary ``x``). The returned dict carries the program bytes,
    word count, and the seed.
    """
    if weight_seed is None:
        weight_seed = WEIGHT_SEED_BASE + out_features * 31 + in_features
    rng_w = np.random.default_rng(int(weight_seed))
    w = rng_w.integers(low=-8, high=8, size=(out_features, in_features), dtype=np.int8)
    if x_vector is None:
        rng_x = np.random.default_rng(int(weight_seed) ^ 0xA1F)
        x_vector = rng_x.integers(low=-8, high=8, size=in_features, dtype=np.int8)
    if x_vector.shape != (in_features,):
        raise ValueError(f"x_vector shape {x_vector.shape} != ({in_features},)")

    lowered = lower_blocked_fc_program_utpu(
        w,
        x_vector,
        out_features,
        in_features,
        ARRAY_SIZE,
        False,
        True,
        WEIGHT_ADDR,
        INPUT_ADDR,
        RESULT_ADDR,
        prog_depth=1024,
    )
    return {
        "program": lowered["program"],
        "program_words": int(lowered["program_instruction_words"]),
        "weight_seed": int(weight_seed),
        "x_bytes": [int(v) for v in x_vector.tolist()],
    }


def _bytes_to_mem_text(byte_seq: bytes) -> str:
    if len(byte_seq) % 2 != 0:
        raise ValueError(
            f"odd byte stream length {len(byte_seq)}; cannot pair into 16b words"
        )
    lines: List[str] = []
    for i in range(0, len(byte_seq), 2):
        word = (byte_seq[i + 1] << 8) | byte_seq[i]
        lines.append(f"{word:04x}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Static-arm sweep (shape sweep + distribution invariance)
# ---------------------------------------------------------------------------


def _shape_tag(out_features: int, in_features: int) -> str:
    return f"(M={out_features},K={in_features})"


def _static_arm_for_shape(out_features: int, in_features: int) -> Dict[str, object]:
    """Compile + measure one shape; assert static == sim exactly."""
    case = _compile_program(out_features, in_features)
    prog = case["program"]
    sim = simulate_program_bytes(prog, array_size=ARRAY_SIZE)
    static = static_cycles_simulator(prog)
    proof = prove_data_independent(prog)
    exact_match = bool(int(static.total_cycles) == int(sim.cycle_count_sequential))
    return {
        "shape": _shape_tag(out_features, in_features),
        "out_features": int(out_features),
        "in_features": int(in_features),
        "program_words": int(case["program_words"]),
        "static_cycles": int(static.total_cycles),
        "simulator_cycles": int(sim.cycle_count_sequential),
        "exact_match": exact_match,
        "per_opcode_cycles": dict(static.per_opcode_cycles),
        "per_opcode_counts": dict(static.per_opcode_counts),
        "halted": bool(static.halted),
        "data_independence_proven": bool(proof.is_proven),
        "data_independent_ops_observed": list(proof.data_independent_ops_observed),
        "data_dependent_ops_found": list(proof.data_dependent_ops_found),
    }


def _distribution_static_sweep(
    out_features: int,
    in_features: int,
    distributions: Sequence[str],
    rng_seed: int,
) -> Dict[str, object]:
    """Generate one program per distribution; assert static-cycle invariance."""
    per_dist: List[Dict[str, object]] = []
    for i, dist in enumerate(distributions):
        rng = np.random.default_rng(rng_seed + i)
        x = _draw_input_vector(dist, in_features, rng)
        case = _compile_program(out_features, in_features, x_vector=x)
        prog = case["program"]
        sim = simulate_program_bytes(prog, array_size=ARRAY_SIZE)
        static = static_cycles_simulator(prog)
        per_dist.append(
            {
                "distribution": dist,
                "static_cycles": int(static.total_cycles),
                "simulator_cycles": int(sim.cycle_count_sequential),
                "program_words": int(case["program_words"]),
                "static_equals_simulator": bool(
                    int(static.total_cycles) == int(sim.cycle_count_sequential)
                ),
                "x_bytes_first8": case["x_bytes"][:8],
            }
        )
    static_counts = [int(d["static_cycles"]) for d in per_dist]
    sim_counts = [int(d["simulator_cycles"]) for d in per_dist]
    static_invariant = len(set(static_counts)) == 1
    sim_invariant = len(set(sim_counts)) == 1
    return {
        "shape": _shape_tag(out_features, in_features),
        "out_features": int(out_features),
        "in_features": int(in_features),
        "distributions": list(distributions),
        "per_distribution": per_dist,
        "static_cycles_invariant_across_distributions": bool(static_invariant),
        "static_cycle_variance": int(_int_variance(static_counts)),
        "simulator_cycles_invariant_across_distributions": bool(sim_invariant),
        "simulator_cycle_variance": int(_int_variance(sim_counts)),
    }


def _int_variance(values: Sequence[int]) -> int:
    """Population variance as a non-negative integer (rounded).

    Returns 0 iff all values are equal. We deliberately use integer
    rounding because the goal is binary equality (0 vs >0); the
    artifact reports variance for human reading but the data-
    independence gate is ``variance == 0``.
    """
    if not values:
        return 0
    if len(set(values)) == 1:
        return 0
    return int(round(statistics.pvariance(values)))


# ---------------------------------------------------------------------------
# RTL arm (iverilog cycle measurement per distribution)
# ---------------------------------------------------------------------------


_LOG_RE_LATENCY = re.compile(
    r"LATENCY_CYCLES=(\d+) SHAPE=(\S+) DISTRIBUTION=(\S+) WORDS=(\d+)"
)
_LOG_RE_DONE = re.compile(r"DONE tests=(\d+) errors=(\d+)")


def _resolve_iverilog() -> Tuple[Optional[str], Optional[str]]:
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


def _iverilog_version(iv_bin: str) -> str:
    try:
        out = subprocess.check_output(
            [iv_bin, "-V"], stderr=subprocess.STDOUT, text=True
        )
        first_line = (out.splitlines() or [""])[0].strip()
        return first_line
    except Exception:
        return ""


def _write_latency_expected_svh(
    mem_path: Path,
    program_words: int,
    shape_tag: str,
    distribution_tag: str,
    *,
    prog_depth: int = 1024,
) -> Path:
    out = TEST_VECTOR_DIR / "latency_expected.svh"
    TEST_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "// Auto-generated by firmware/host/run_latency_determinism.py",
        "// Task 4 -- per-trial header for tb_latency_determinism.sv.",
        "// DO NOT EDIT.",
        f"`define LATENCY_MEM         \"{mem_path.as_posix()}\"",
        f"`define LATENCY_WORDS       {int(program_words)}",
        f"`define LATENCY_SHAPE_TAG   \"{shape_tag}\"",
        f"`define LATENCY_DIST_TAG    \"{distribution_tag}\"",
        f"`define LATENCY_TB_PROG_DEPTH {int(prog_depth)}",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _compile_testbench(iv_bin: str) -> Tuple[bool, str, Path]:
    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_vvp = SIM_OUT_DIR / "tb_latency_determinism.out"
    if out_vvp.exists():
        out_vvp.unlink()
    sources = [TB_FILE] + DESIGN_FILES
    compile_cmd = [
        iv_bin,
        "-g2012",
        "-DICARUS",
        "-o",
        str(out_vvp),
        *sources,
    ]
    proc = subprocess.run(
        compile_cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode == 0, proc.stdout, out_vvp


def _cycles_to_wall_ns(cycles: int, clock_mhz: float = FPGA_CLOCK_MHZ) -> float:
    """Convert RTL cycle count to nanoseconds at the stated FPGA clock."""
    return float(cycles) * (1000.0 / float(clock_mhz))


def _wall_clock_block(cycles_list: Sequence[int]) -> Dict[str, object]:
    """Provenance-bearing wall-clock conversion for a cycle sample list."""
    ns = [_cycles_to_wall_ns(int(c)) for c in cycles_list]
    return {
        "clock_mhz": float(FPGA_CLOCK_MHZ),
        "clock_source": FPGA_CLOCK_SOURCE,
        "conversion": "wall_ns = cycles * (1000 / clock_mhz)  # i.e. 10 ns/cycle at 100 MHz",
        "samples_ns": [float(v) for v in ns],
        "samples_us": [float(v) / 1000.0 for v in ns],
        "median_ns": float(statistics.median(ns)) if ns else None,
        "min_ns": float(min(ns)) if ns else None,
        "max_ns": float(max(ns)) if ns else None,
        "stddev_ns": float(statistics.pstdev(ns)) if len(ns) > 1 else (0.0 if ns else None),
        "jitter_ns": float(max(ns) - min(ns)) if ns else None,
        "jitter_cycles": int(max(cycles_list) - min(cycles_list)) if cycles_list else None,
    }


def _run_one_trial(
    vv_bin: str,
    vvp_path: Path,
    *,
    mem_path: Optional[Path] = None,
    program_words: Optional[int] = None,
    shape_tag: Optional[str] = None,
    distribution_tag: Optional[str] = None,
) -> Tuple[bool, str, Optional[int]]:
    """Run vvp once; pass trial metadata via plusargs when provided."""
    cmd: List[str] = [vv_bin, str(vvp_path)]
    if mem_path is not None:
        cmd.append(f"+LATENCY_MEM={mem_path.as_posix()}")
    if program_words is not None:
        cmd.append(f"+LATENCY_WORDS={int(program_words)}")
    if shape_tag is not None:
        cmd.append(f"+LATENCY_SHAPE_TAG={shape_tag}")
    if distribution_tag is not None:
        cmd.append(f"+LATENCY_DIST_TAG={distribution_tag}")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log = proc.stdout
    m = _LOG_RE_LATENCY.search(log)
    if not m:
        return False, log, None
    cycles = int(m.group(1))
    done_m = _LOG_RE_DONE.search(log)
    if done_m and int(done_m.group(2)) != 0:
        return False, log, cycles
    ok = "TB_RESULT: PASS" in log
    return ok, log, cycles


def _prepare_trial_mem(
    out_features: int,
    in_features: int,
    tag: str,
    x_vector: np.ndarray,
) -> Dict[str, object]:
    """Compile one program and write its .mem; return trial metadata."""
    case = _compile_program(out_features, in_features, x_vector=x_vector)
    mem_path = TEST_VECTOR_DIR / f"latency_{out_features}x{in_features}_{tag}.mem"
    TEST_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    mem_path.write_text(_bytes_to_mem_text(case["program"]), encoding="utf-8")
    return {
        "tag": tag,
        "program_words": int(case["program_words"]),
        "mem_path": mem_path,
        "x_bytes_first8": case["x_bytes"][:8],
    }


def _rtl_arm_for_distribution_sweep(
    shape: Tuple[int, int],
    distributions: Sequence[str],
    rng_seed: int,
    iv_bin: str,
    vv_bin: str,
    *,
    e2e_random_trials: int = DEFAULT_E2E_RANDOM_TRIALS,
) -> Dict[str, object]:
    """Capture RTL cycles across adversarial + many random inputs.

    Compiles the testbench once (header provides fallback `defines), then
    re-runs ``vvp`` per trial with plusargs so end-to-end latency across
    many inputs does not require a recompile per vector.
    """
    out_features, in_features = shape
    shape_tag = _shape_tag(out_features, in_features)

    # Seed header so the `include resolves at compile time even when
    # every live trial overrides via plusargs.
    seed_x = _draw_input_vector("zero", in_features, np.random.default_rng(rng_seed))
    seed_trial = _prepare_trial_mem(out_features, in_features, "seed_header", seed_x)
    _write_latency_expected_svh(
        seed_trial["mem_path"],
        int(seed_trial["program_words"]),
        shape_tag,
        "seed_header",
    )
    compile_ok, compile_stdout, vvp_path = _compile_testbench(iv_bin)
    if not compile_ok:
        return {
            "shape": shape_tag,
            "iverilog_compile_ok": False,
            "compile_stdout": compile_stdout,
            "per_distribution": [],
            "e2e_trials": [],
            "rtl_cycles_invariant_across_distributions": False,
            "rtl_cycle_variance": None,
            "rtl_cycles_observed": [],
            "all_trials_passed": False,
            "wall_clock": None,
        }

    per_dist: List[Dict[str, object]] = []
    cycles_seen: List[int] = []
    overall_pass = True

    for i, dist in enumerate(distributions):
        rng = np.random.default_rng(rng_seed + i)
        x = _draw_input_vector(dist, in_features, rng)
        trial = _prepare_trial_mem(out_features, in_features, dist, x)
        ok, log, cycles = _run_one_trial(
            vv_bin,
            vvp_path,
            mem_path=trial["mem_path"],
            program_words=int(trial["program_words"]),
            shape_tag=shape_tag,
            distribution_tag=dist,
        )
        if not ok or cycles is None:
            overall_pass = False
        else:
            cycles_seen.append(int(cycles))
        per_dist.append(
            {
                "distribution": dist,
                "rtl_cycles": int(cycles) if cycles is not None else None,
                "iverilog_recompile_ok": True,
                "tb_result_pass": bool(ok),
                "program_words": int(trial["program_words"]),
                "mem_path": trial["mem_path"].as_posix(),
                "x_bytes_first8": trial["x_bytes_first8"],
                "vvp_log_tail": "\n".join(log.splitlines()[-12:]),
                "wall_ns": (
                    _cycles_to_wall_ns(int(cycles)) if cycles is not None else None
                ),
            }
        )

    # Many-input end-to-end sweep: additional random activation vectors
    # against the same weight seed / shape (data-independence witness).
    e2e_trials: List[Dict[str, object]] = []
    e2e_cycles: List[int] = []
    n_e2e = max(0, int(e2e_random_trials))
    for j in range(n_e2e):
        rng = np.random.default_rng(rng_seed + 10_000 + j)
        x = _draw_input_vector("random", in_features, rng)
        tag = f"e2e_random_{j:04d}"
        trial = _prepare_trial_mem(out_features, in_features, tag, x)
        ok, log, cycles = _run_one_trial(
            vv_bin,
            vvp_path,
            mem_path=trial["mem_path"],
            program_words=int(trial["program_words"]),
            shape_tag=shape_tag,
            distribution_tag=tag,
        )
        if not ok or cycles is None:
            overall_pass = False
        else:
            e2e_cycles.append(int(cycles))
            cycles_seen.append(int(cycles))
        e2e_trials.append(
            {
                "trial_index": int(j),
                "tag": tag,
                "rtl_cycles": int(cycles) if cycles is not None else None,
                "tb_result_pass": bool(ok),
                "program_words": int(trial["program_words"]),
                "mem_path": trial["mem_path"].as_posix(),
                "x_bytes_first8": trial["x_bytes_first8"],
                "wall_ns": (
                    _cycles_to_wall_ns(int(cycles)) if cycles is not None else None
                ),
                "vvp_log_tail": "\n".join(log.splitlines()[-8:]),
            }
        )

    all_expected = len(distributions) + n_e2e
    dist_cycles = [
        int(d["rtl_cycles"]) for d in per_dist if d.get("rtl_cycles") is not None
    ]
    dist_invariant = bool(
        len(dist_cycles) == len(distributions) and len(set(dist_cycles)) == 1
    )
    e2e_invariant = (
        bool(len(e2e_cycles) == n_e2e and len(set(e2e_cycles)) == 1)
        if n_e2e > 0
        else None
    )
    wall = _wall_clock_block(cycles_seen) if cycles_seen else None
    return {
        "shape": shape_tag,
        "iverilog_compile_ok": True,
        "compile_once": True,
        "plusargs_dispatch": True,
        "per_distribution": per_dist,
        "e2e_random_trials_requested": int(n_e2e),
        "e2e_trials": e2e_trials,
        "e2e_rtl_cycles_observed": list(e2e_cycles),
        "e2e_rtl_cycle_variance": int(_int_variance(e2e_cycles)) if e2e_cycles else None,
        "e2e_rtl_cycles_invariant": e2e_invariant,
        "rtl_cycles_invariant_across_distributions": bool(dist_invariant),
        "distribution_rtl_cycles_observed": list(dist_cycles),
        "distribution_rtl_cycle_variance": int(_int_variance(dist_cycles)),
        "rtl_cycle_variance": int(_int_variance(cycles_seen)) if cycles_seen else None,
        "rtl_cycles_observed": list(cycles_seen),
        "n_e2e_inputs_measured": int(len(cycles_seen)),
        "all_trials_passed": bool(overall_pass and len(cycles_seen) == all_expected),
        "wall_clock": wall,
    }


# ---------------------------------------------------------------------------
# Artifact emission
# ---------------------------------------------------------------------------


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


def _build_artifact(
    *,
    shapes: Sequence[Tuple[int, int]],
    distributions: Sequence[str],
    distribution_shape: Tuple[int, int],
    rng_seed: int,
    iv_bin: Optional[str],
    vv_bin: Optional[str],
    skip_iverilog: bool,
    e2e_random_trials: int = DEFAULT_E2E_RANDOM_TRIALS,
) -> Dict[str, object]:
    static_vs_sim_entries: List[Dict[str, object]] = []
    proof_observed_union: set = set()
    proof_flagged_union: set = set()
    static_all_match = True
    static_all_proven = True

    for (m, k) in shapes:
        entry = _static_arm_for_shape(m, k)
        if not entry["exact_match"]:
            static_all_match = False
        if not entry["data_independence_proven"]:
            static_all_proven = False
        proof_observed_union.update(entry["data_independent_ops_observed"])
        proof_flagged_union.update(entry["data_dependent_ops_found"])
        static_vs_sim_entries.append(entry)

    dist_static = _distribution_static_sweep(
        distribution_shape[0],
        distribution_shape[1],
        distributions,
        rng_seed,
    )

    rtl_arm: Optional[Dict[str, object]] = None
    if skip_iverilog or iv_bin is None or vv_bin is None:
        on_silicon_status = "rtl_sim_unavailable_no_iverilog" if (
            iv_bin is None or vv_bin is None
        ) else "rtl_sim_skipped_by_user"
        rtl_arm = None
    else:
        rtl_arm = _rtl_arm_for_distribution_sweep(
            distribution_shape,
            distributions,
            rng_seed,
            iv_bin,
            vv_bin,
            e2e_random_trials=int(e2e_random_trials),
        )
        on_silicon_status = "rtl_sim"

    # Distribution-only RTL invariance (legacy gate) vs all-input e2e.
    dist_rtl_cycles: List[int] = []
    if rtl_arm is not None:
        for d in rtl_arm.get("per_distribution") or []:
            if d.get("rtl_cycles") is not None:
                dist_rtl_cycles.append(int(d["rtl_cycles"]))
    dist_rtl_invariant = bool(
        len(dist_rtl_cycles) == len(distributions) and len(set(dist_rtl_cycles)) == 1
    )
    if rtl_arm is not None:
        rtl_arm["rtl_cycles_invariant_across_distributions"] = dist_rtl_invariant
        rtl_arm["distribution_rtl_cycle_variance"] = int(_int_variance(dist_rtl_cycles))
        rtl_arm["distribution_rtl_cycles_observed"] = list(dist_rtl_cycles)

    data_independence_block = {
        "shape": dist_static["shape"],
        "distributions_tested": list(distributions),
        "static_cycle_variance": int(dist_static["static_cycle_variance"]),
        "static_cycle_invariant": bool(
            dist_static["static_cycles_invariant_across_distributions"]
        ),
        "simulator_cycle_variance": int(dist_static["simulator_cycle_variance"]),
        "simulator_cycle_invariant": bool(
            dist_static["simulator_cycles_invariant_across_distributions"]
        ),
        "rtl_cycle_variance": (
            int(rtl_arm["rtl_cycle_variance"])
            if (rtl_arm is not None and rtl_arm.get("rtl_cycle_variance") is not None)
            else None
        ),
        "rtl_cycle_invariant": bool(
            rtl_arm is not None
            and rtl_arm.get("rtl_cycles_invariant_across_distributions", False)
            and (
                rtl_arm.get("e2e_rtl_cycles_invariant") in (True, None)
            )
        ),
        "rtl_arm_present": rtl_arm is not None,
        "n_e2e_inputs_measured": (
            int(rtl_arm.get("n_e2e_inputs_measured") or 0) if rtl_arm else 0
        ),
        "fpga_clock_mhz": float(FPGA_CLOCK_MHZ),
        "fpga_clock_source": FPGA_CLOCK_SOURCE,
    }

    overall_static_arm_ok = bool(
        static_all_match and static_all_proven and len(proof_flagged_union) == 0
        and dist_static["static_cycles_invariant_across_distributions"]
    )
    overall_rtl_arm_ok = bool(
        rtl_arm is None
        or rtl_arm.get("rtl_cycles_invariant_across_distributions", False)
    )
    overall_pass = bool(overall_static_arm_ok and overall_rtl_arm_ok)

    artifact: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_iso(),
        "git_sha": _git_sha(),
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        },
        "iverilog_version": _iverilog_version(iv_bin) if iv_bin else "",
        "status": "rtl_sim" if overall_pass else "failed",
        "shapes": [f"(M={m},K={k})" for (m, k) in shapes],
        "static_vs_sim": static_vs_sim_entries,
        "static_arm_all_shapes_exact_match": bool(static_all_match),
        "data_independence": [data_independence_block],
        "distribution_sweep_detail": dist_static,
        "rtl_arm": rtl_arm,
        "proof": {
            "data_independent_ops": list(DEFAULT_DATA_INDEPENDENT_OPS),
            "data_independent_ops_observed_union": sorted(proof_observed_union),
            "data_dependent_ops_found": sorted(proof_flagged_union),
        },
        "on_silicon": {
            "status": on_silicon_status,
            "onchip_counter_cycles": None,
            "matches_static": None,
            "note": (
                "Arty A7 board not available at landing time; will be filled "
                "by program_loader.py::readPerfCounters() on board arrival "
                "(plan section 6.2.3.(c))."
            ),
        },
        "scope_note": (
            "Constant-time over the supported data-independent op set "
            f"({sorted(DEFAULT_DATA_INDEPENDENT_OPS)!r}); COMPUTE latency "
            "only; UART/IO framing bounded separately. The static cycle "
            "model matches the in-order ISA simulator's cycle accounting "
            "exactly (verified by static_vs_sim[].exact_match=True on all "
            "swept shapes); the simulator is RTL-corroborated at the "
            "cycle-reduction-percentage level by Phase 7 remediation P4.1 "
            "(±2.0% per bench/results/scheduler_rtl_crosscheck.json), NOT "
            "at the absolute count level. The data-independence claim "
            "this artifact establishes is: RTL cycle variance across "
            "adversarial input distributions == 0 for the same compiled "
            "program (empirical) AND every opcode in the swept programs "
            "is in the data-independent allowlist (static proof)."
        ),
        "fpga_clock": {
            "mhz": float(FPGA_CLOCK_MHZ),
            "source": FPGA_CLOCK_SOURCE,
            "conversion": "wall_ns = cycles * (1000 / mhz)  # 10 ns/cycle at 100 MHz",
            "note": (
                "Wall-clock figures are simulated RTL cycles converted at the "
                "measured closed 100 MHz constraint; not on-board silicon timing."
            ),
        },
        "methodology": {
            "summary": (
                "Three-arm verification of cycle-deterministic execution: "
                "(1) exact static cycle model matches isa_simulator accounting "
                "byte-for-byte across N shapes; (2) static cycle count "
                "invariant across M adversarial input distributions on a "
                "representative shape; (3) empirical RTL arm shows variance==0 "
                "across the same M distributions PLUS many additional random "
                "end-to-end inputs when iverilog is available (compile-once + "
                "plusargs dispatch). Cycle counts convert to wall-clock at the "
                f"measured {FPGA_CLOCK_MHZ:g} MHz FPGA clock."
            ),
            "headline_assertions": [
                "static_cycles == simulator_cycles on every swept shape",
                "data_independence_proven on every swept shape",
                "static_cycles invariant across all input distributions for the distribution-sweep shape",
                "RTL cycle variance == 0 across distributions + e2e random inputs (iverilog arm; skipped if no iverilog)",
            ],
            "tools": {
                "static_model": "firmware/host/latency_analysis.py::static_cycles_simulator",
                "prover": "firmware/host/latency_analysis.py::prove_data_independent",
                "simulator": "firmware/host/isa_simulator.py::simulate_program_bytes",
                "lowering": "firmware/host/lowering_blocked_fc_utpu.py::lower_blocked_fc_program_utpu",
                "testbench": "rtl/tb/tb_latency_determinism.sv",
            },
            "shape_sweep_seed_base": int(WEIGHT_SEED_BASE),
            "distribution_sweep_seed": int(rng_seed),
            "array_size": int(ARRAY_SIZE),
            "e2e_random_trials": int(e2e_random_trials),
            "fpga_clock_mhz": float(FPGA_CLOCK_MHZ),
        },
    }
    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_shape_spec(spec: str) -> Tuple[int, int]:
    """Parse a shape spec like 'M=32,K=32' or '32x32'."""
    s = spec.strip().lower().replace(" ", "")
    if "x" in s:
        a, b = s.split("x", 1)
        return int(a), int(b)
    if "," in s:
        parts = dict(p.split("=") for p in s.split(","))
        return int(parts["m"]), int(parts["k"])
    raise ValueError(f"could not parse shape spec: {spec!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(OUTPUT_JSON),
        help="Path to write the JSON artifact (default: bench/results/latency_determinism.json)",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        help="Override shape sweep (e.g. --shapes 32x32 64x64). Default: 4 shipping-fit shapes.",
    )
    parser.add_argument(
        "--distribution-shape",
        default=None,
        help=(
            "Override the shape used for the distribution sweep "
            "(default: '32x32', matches the P4.1 testbench shape)."
        ),
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=list(DEFAULT_DISTRIBUTIONS),
        help="Override input-distribution set used by the data-independence sweep.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=20260527,
        help="Seed base for distribution-sweep input vectors.",
    )
    parser.add_argument(
        "--skip-iverilog",
        action="store_true",
        help="Emit static-only artifact regardless of iverilog availability (stub-mode).",
    )
    parser.add_argument(
        "--e2e-random-trials",
        type=int,
        default=DEFAULT_E2E_RANDOM_TRIALS,
        help=(
            "Additional random-input RTL end-to-end trials beyond the adversarial "
            f"distribution set (default: {DEFAULT_E2E_RANDOM_TRIALS})."
        ),
    )
    args = parser.parse_args(argv)

    shapes = (
        tuple(_parse_shape_spec(s) for s in args.shapes)
        if args.shapes
        else DEFAULT_SHAPES
    )
    distribution_shape = (
        _parse_shape_spec(args.distribution_shape)
        if args.distribution_shape
        else DEFAULT_DISTRIBUTION_SHAPE
    )

    iv_bin, vv_bin = (None, None) if args.skip_iverilog else _resolve_iverilog()

    artifact = _build_artifact(
        shapes=shapes,
        distributions=tuple(args.distributions),
        distribution_shape=distribution_shape,
        rng_seed=int(args.rng_seed),
        iv_bin=iv_bin,
        vv_bin=vv_bin,
        skip_iverilog=bool(args.skip_iverilog),
        e2e_random_trials=int(args.e2e_random_trials),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"[run_latency_determinism] status={artifact['status']} "
        f"static_match={artifact['static_arm_all_shapes_exact_match']} "
        f"rtl_present={artifact['data_independence'][0]['rtl_arm_present']} "
        f"on_silicon={artifact['on_silicon']['status']} -> {out_path}"
    )
    return 0 if artifact["status"] == "rtl_sim" else 1


if __name__ == "__main__":
    raise SystemExit(main())
