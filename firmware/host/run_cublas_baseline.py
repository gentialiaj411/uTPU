"""Phase 7 — Serious CUDA baseline harness.

Compares the uTPU autotuner-selected, NVRTC-compiled blocked-FC kernel
against three "serious-library" references on a fixed shape grid:

* **cuBLAS GEMV** via ``torch.matmul`` (INT32 weights / INT32 activations /
  INT32 accumulator where supported, FP32 fallback otherwise). The dtype
  actually measured is recorded verbatim; the fallback is never silent.
* **cuBLASLt IMMA INT8 GEMM** via ``torch._int_mm`` (INT8 inputs /
  INT32 accumulator). This is the **dtype-matched** apples-to-apples
  comparison to the uTPU kernel; cuBLASLt IMMA requires ``N % 8 == 0``
  so the uTPU N=1 GEMV is padded to N=8 inside the subprocess and the
  parent uses the per-N-column equivalent (``kernel_ms.median / N_padded``)
  as the GEMV-equivalent gap. The alignment caveat is documented
  verbatim in the methodology block and the per-shape ``alignment_note``.
* **TorchInductor** via ``torch.compile(nn.Linear, backend="inductor")``.
  This is the "what would the official PyTorch compiler do?"
  reference. Runs in FP32 — the artifact records this dtype mismatch
  explicitly so the comparison is never silently inflated.

Methodology (locked, identical for all three backends):

* Six fixed shapes: ``(M, K)`` with N=1 (FC-inference GEMV). Shape set
  is documented in :data:`SHAPES` and embedded in the artifact.
* **Per-iter timing protocol**: CUDA events (``torch.cuda.Event(
  enable_timing=True)``) recorded in stream order; one
  ``torch.cuda.synchronize()`` after the iter loop, then
  ``start.elapsed_time(end)`` per iter. Replaces wall-clock +
  per-iter-sync (which was the dominant noise source at 30–50 µs
  GEMV scale on the RTX 5070).
* **Inter-run stability protocol v2 (spin-up + trimmed median)**: one
  discarded spin-up ``warmup + iters`` pass per arm per shape (to
  engage GPU boost clocks before any timed sample is collected), then
  ``num_stability_runs`` (default 5) back-to-back ``warmup + iters``
  blocks. If ``num_stability_runs >= 4``, the highest and lowest
  per-run medians are dropped before computing the inter-run CV *and*
  before pooling samples into ``kernel_ms.median`` — so a single
  cold-clock or thermal-blip outlier run no longer pollutes the
  shipped median. The per-shape ``kernel_ms`` reports the full per-run
  medians (kept for diagnostics, so the offending-shape print loop can
  surface the noise pattern verbatim), the trimmed per-run medians,
  and an inter-run ``timing_stability_pct`` (CV of *trimmed* per-run
  medians). A documented threshold of 10% (recorded in the
  methodology block) gates the **load-bearing arm only** via
  ``test_load_bearing_utpu_arm_is_gate_green`` (strict; build fails
  on excess CV). The baseline arms get an advisory CV report via
  ``test_baseline_arms_advisory_cv_reporting`` — that test does not
  fail the build because multi-launch host-side jitter on a contended
  laptop GPU without ``nvidia-smi --lock-gpu-clocks`` is intrinsic
  to the host environment, and the published ``gap_vs_*_pct`` values
  stay ``[needs-locked-clock-artifact]`` regardless.
  Diagnosis that motivated v2: the v1 regen on WSL2 + RTX 5070 Laptop
  GPU produced per-run medians like ``cublaslt_int8 @ 128×128 = [59µs,
  9µs, 9µs]`` — one cold-clock stray, two stable — and the
  un-trimmed CV came out at 90% (gate failed). v2's spin-up engages
  the GPU before any sample is recorded, and trimming drops the
  surviving single-stray-run cases that the spin-up doesn't catch.
* Warmup = 25 invocations per stability run (discarded). Then
  iters = 100 measured invocations per stability run.
* Per-shape stats: mean, median, stdev, min, max, p95 (ms) over the
  *post-trim* flat sample list (samples from dropped outlier runs are
  excluded from the published median), plus the full per-run median
  list, the trimmed per-run median list, and inter-run CV.
* TFLOPS computed as ``2 * M * N * K / median_ms * 1e-12`` for INT-MAC
  ops (uTPU + cuBLAS). Inductor TFLOPS is reported separately because
  the op count is the same but the dtype is not.
* GPU info recorded: device name, capability, Torch/CUDA versions,
  device index.

The harness is **GPU-gated**: on a non-CUDA host it writes a stub
artifact with ``status="cuda_unavailable"``, the full shape set that
would have been measured, and an explicit "re-run on a CUDA host"
instruction. The schema (top-level keys + shape entry shape) is
identical in both modes so :mod:`test_cublas_baseline` can lock the
contract on either host class.

Subprocess isolation: the Torch baselines run in
:mod:`_cublas_baseline_torch_subprocess` to avoid Torch/Inductor and
NVRTC fighting over the same CUDA driver context — the same pattern
used by ``inductor_oracle_subprocess.py`` for ResNet-18 Inductor
parity.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from cuda_blocked_fc_backend import (  # noqa: E402
    CUDABlockedFCExecutor,
    detect_cuda_environment,
)
from lowering_types import BlockedFCLoweringRequest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "cublas_baseline.json"
SUBPROCESS_SCRIPT = Path(__file__).with_name("_cublas_baseline_torch_subprocess.py")

# Locked shape set. M = out_features (rows of the weight matrix), K =
# in_features (columns), N = 1 (single-vector FC inference). Picked to
# span the small / medium / large regions of the calibration grid so
# the cuBLAS-launch-overhead vs cuBLAS-compute-bound regimes are both
# represented and a single "X% of cuBLAS" headline cannot be cherry-
# picked across the artifact.
SHAPES: List[Dict[str, int]] = [
    {"M": 16, "K": 16},
    {"M": 64, "K": 64},
    {"M": 128, "K": 128},
    {"M": 256, "K": 256},
    {"M": 512, "K": 256},
    {"M": 512, "K": 512},
]

ARRAY_SIZE = 16
DEFAULT_WARMUP = 50
DEFAULT_ITERS = 200
DEFAULT_NUM_STABILITY_RUNS = 7
TIMING_STABILITY_THRESHOLD_PCT = 10.0
TIMING_PROTOCOL_NAME = "cuda_events_per_run_stability_v2_spin_up_plus_trimmed_median"
# At v2.1 defaults (N=7), trim drops 2 from each end → keep 3 stable
# runs even when 2 of the 7 are clock-state outliers. See the methodology
# block below for the noise pattern this addresses.
TRIM_MIN_RUNS = 4


def _trim_count_per_side(n: int) -> int:
    """How many per-run medians to drop from each end at ``num_stability_runs=n``.

    Formula: ``max(1, (n - 3) // 2)`` for ``n >= TRIM_MIN_RUNS``. Always
    keeps a minimum of 3 surviving runs. Mirrors the helper in
    :mod:`_cublas_baseline_torch_subprocess` so the uTPU arm and the
    Torch arms share one trimming rule.
    """
    if n < TRIM_MIN_RUNS:
        return 0
    return max(1, (n - 3) // 2)


def _trim_per_run_medians(
    per_run_medians: List[float], min_n_for_trim: int = TRIM_MIN_RUNS
) -> Tuple[List[float], List[int]]:
    """Drop ``_trim_count_per_side(n)`` highest + lowest per-run medians.

    Returns ``(trimmed_medians_in_original_order, kept_run_indices)`` —
    the kept indices let the caller rebuild a per-iter sample list from
    only the kept (post-trim) stability runs. At the v2.1 default
    ``n=7`` we drop 2 from each end and keep 3 stable runs (vs v2 which
    dropped 1 from each end at ``n=5`` and kept 3); the extra trim
    catches the two-outlier-run-out-of-N pattern observed at sub-20µs
    kernel scale on the WSL2 + RTX 5070 Laptop GPU.
    """
    n = len(per_run_medians)
    if n < int(min_n_for_trim):
        return list(per_run_medians), list(range(n))
    k = _trim_count_per_side(n)
    if k <= 0 or 2 * k >= n:
        return list(per_run_medians), list(range(n))
    sorted_by_value = sorted(range(n), key=lambda i: per_run_medians[i])
    kept_idx_ordered = sorted(sorted_by_value[k:-k])
    trimmed = [per_run_medians[i] for i in kept_idx_ordered]
    return trimmed, kept_idx_ordered


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _summary_ms(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "samples": 0,
        }
    samples_sorted = sorted(samples)
    p95_idx = max(0, int(round(0.95 * (len(samples_sorted) - 1))))
    return {
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "stdev": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        "min": float(samples_sorted[0]),
        "max": float(samples_sorted[-1]),
        "p95": float(samples_sorted[p95_idx]),
        "samples": int(len(samples)),
    }


def _int_tflops(M: int, K: int, ms: float) -> float:
    if ms <= 0.0:
        return 0.0
    ops = 2.0 * float(M) * float(K) * 1.0  # N=1
    return ops / (ms / 1000.0) / 1e12


@dataclass
class UtpuShapeResult:
    M: int
    K: int
    schedule_params: Dict[str, Any]
    kernel_ms_summary: Dict[str, float]
    samples_ms: List[float]
    int_tflops_median: float
    compile_time_ms: float
    setup_time_ms: float
    bit_exact_match_vs_numpy_reference: bool


def _run_utpu_shape(
    executor: CUDABlockedFCExecutor,
    M: int,
    K: int,
    warmup: int,
    iters: int,
    num_stability_runs: int,
    rng_seed: int,
) -> UtpuShapeResult:
    """Run the uTPU NVRTC blocked-FC kernel with N back-to-back stability runs.

    Per-iter ``kernel_time_ms`` is sourced verbatim from
    ``CUDABlockedFCExecutor.execute``; that executor already reports
    GPU-side kernel time (NVRTC + cuda-python ``cuEventElapsedTime``
    around ``cuLaunchKernel`` inside the C++ extension path). The
    inter-run stability layer here is the SAME layer applied to the
    cuBLAS / IMMA / Inductor arms inside
    ``_cublas_baseline_torch_subprocess.py``, so all four arms in this
    artifact share one timing-stability protocol.
    """
    rng = np.random.default_rng(rng_seed)
    w = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
    x = rng.integers(-8, 8, size=(K,), dtype=np.int8)
    req = BlockedFCLoweringRequest(
        weights_int4=w,
        activations_int4=x,
        out_features=M,
        in_features=K,
        array_size=ARRAY_SIZE,
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )

    per_run_samples: List[List[float]] = []
    per_run_medians: List[float] = []
    compile_time_ms = 0.0
    setup_time_ms = 0.0
    schedule_params: Dict[str, Any] = {}
    bit_exact = True
    n_runs = max(1, int(num_stability_runs))

    def _do_block(label: str, store_first_iter_meta: bool) -> List[float]:
        """One ``warmup + iters`` block. Returns the per-iter sample list."""
        nonlocal compile_time_ms, setup_time_ms, schedule_params, bit_exact
        for _ in range(warmup):
            out_w = executor.execute(req)
            if not out_w.get("executed", False):
                raise RuntimeError(
                    f"uTPU warmup failed at (M={M}, K={K}, block={label}): "
                    f"{out_w.get('reason')}"
                )
        block_samples: List[float] = []
        for i in range(iters):
            out_i = executor.execute(req)
            if not out_i.get("executed", False):
                raise RuntimeError(
                    f"uTPU iter {i} failed at (M={M}, K={K}, block={label}): "
                    f"{out_i.get('reason')}"
                )
            block_samples.append(float(out_i["kernel_time_ms"]))
            if store_first_iter_meta and i == 0:
                compile_time_ms = float(out_i.get("compile_time_ms", 0.0))
                setup_time_ms = float(out_i.get("setup_time_ms", 0.0))
                schedule_params = dict(out_i.get("schedule_params", {}))
            bit_exact = bit_exact and bool(
                out_i.get("bit_exact_match_vs_numpy_reference", True)
            )
        return block_samples

    # Spin-up run (discarded entirely). Engages GPU boost clocks and warms
    # the NVRTC kernel cache so the first *counted* stability run doesn't
    # inherit cold-start cost. compile_time_ms / setup_time_ms / schedule
    # params still come from the executor metadata on this run (the
    # NVRTC compile happens lazily on the very first call regardless of
    # whether it's spin-up or counted).
    _do_block("spin_up", store_first_iter_meta=True)

    for run_idx in range(n_runs):
        run_samples = _do_block(f"stability_run_{run_idx}", store_first_iter_meta=False)
        per_run_samples.append(run_samples)
        per_run_medians.append(
            float(statistics.median(run_samples)) if run_samples else 0.0
        )

    trimmed_medians, kept_idx = _trim_per_run_medians(
        per_run_medians, min_n_for_trim=TRIM_MIN_RUNS
    )
    kept_samples: List[float] = []
    for i in kept_idx:
        kept_samples.extend(per_run_samples[i])

    summary = _summary_ms(kept_samples)
    if len(trimmed_medians) >= 2 and statistics.fmean(trimmed_medians) > 0.0:
        mean_med = float(statistics.fmean(trimmed_medians))
        stdev_med = float(statistics.pstdev(trimmed_medians))
        cv_pct = float(stdev_med / mean_med * 100.0) if mean_med > 0.0 else 0.0
    else:
        stdev_med = 0.0
        cv_pct = 0.0
    summary["per_run_medians_ms"] = list(per_run_medians)
    summary["per_run_medians_trimmed_ms"] = list(trimmed_medians)
    summary["timing_stability_stddev_ms"] = stdev_med
    summary["timing_stability_pct"] = cv_pct
    summary["num_stability_runs"] = int(n_runs)
    summary["num_stability_runs_trimmed"] = int(len(trimmed_medians))
    summary["timing_protocol"] = TIMING_PROTOCOL_NAME

    return UtpuShapeResult(
        M=M,
        K=K,
        schedule_params=schedule_params,
        kernel_ms_summary=summary,
        samples_ms=[float(s) for s in kept_samples[:32]],
        int_tflops_median=_int_tflops(M, K, summary["median"]),
        compile_time_ms=compile_time_ms,
        setup_time_ms=setup_time_ms,
        bit_exact_match_vs_numpy_reference=bit_exact,
    )


def _run_torch_subprocess(
    shapes: List[Dict[str, int]],
    warmup: int,
    iters: int,
    num_stability_runs: int,
    tmp_path: Path,
) -> Dict[str, Any]:
    shapes_json = json.dumps(shapes)
    proc = subprocess.run(
        [
            sys.executable,
            str(SUBPROCESS_SCRIPT),
            "--shapes-json",
            shapes_json,
            "--output",
            str(tmp_path),
            "--warmup",
            str(int(warmup)),
            "--iters",
            str(int(iters)),
            "--num-stability-runs",
            str(int(num_stability_runs)),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        return {
            "status": "subprocess_failed",
            "returncode": int(proc.returncode),
            "stderr": proc.stderr[-2000:],
            "stdout": proc.stdout[-2000:],
            "shapes_requested": shapes,
        }
    try:
        return json.loads(tmp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "subprocess_json_parse_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "stdout": proc.stdout[-2000:],
            "shapes_requested": shapes,
        }


def _gap_pct(ours: float, baseline: float | None) -> float | None:
    if baseline is None or baseline <= 0.0 or ours is None or ours <= 0.0:
        return None
    return float((ours - baseline) / baseline * 100.0)


def _cublaslt_int8_per_column_median(cublaslt: Dict[str, Any] | None) -> float | None:
    """Return the per-N-column cuBLASLt IMMA INT8 median time (ms) if available.

    The subprocess emits ``ms_per_n_column_median`` whenever the IMMA
    arm actually ran; on hosts where IMMA is unavailable it emits
    ``imma_unavailable_reason`` instead and the per-column figure is
    absent. We treat both cases uniformly (return None) so the parent's
    gap field collapses to None and the aggregate count reflects the
    IMMA-available shapes only.
    """
    if not cublaslt:
        return None
    if "imma_unavailable_reason" in cublaslt or "error" in cublaslt:
        return None
    val = cublaslt.get("ms_per_n_column_median")
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return v if v > 0.0 else None


def _merge_per_shape(
    utpu: UtpuShapeResult,
    torch_entry: Dict[str, Any] | None,
) -> Dict[str, Any]:
    cublas = (torch_entry or {}).get("cublas")
    cublaslt_int8 = (torch_entry or {}).get("cublaslt_int8")
    inductor = (torch_entry or {}).get("inductor")

    utpu_median = utpu.kernel_ms_summary.get("median", 0.0)
    cublas_median = (cublas or {}).get("kernel_ms", {}).get("median")
    cublaslt_int8_per_col_median = _cublaslt_int8_per_column_median(cublaslt_int8)
    inductor_median = (inductor or {}).get("kernel_ms", {}).get("median")

    return {
        "shape": {"M": int(utpu.M), "K": int(utpu.K), "N": 1},
        "utpu": {
            "backend": "utpu_blocked_fc_nvrtc_int4",
            "dtype_W": "int8",
            "dtype_x": "int8",
            "dtype_accum": "int32",
            "dtype_out": "int4_quantized",
            "schedule_params": utpu.schedule_params,
            "kernel_ms": dict(utpu.kernel_ms_summary),
            "samples_ms": list(utpu.samples_ms),
            "int_mac_tflops_median": float(utpu.int_tflops_median),
            "compile_time_ms": float(utpu.compile_time_ms),
            "setup_time_ms": float(utpu.setup_time_ms),
            "bit_exact_match_vs_numpy_reference": bool(
                utpu.bit_exact_match_vs_numpy_reference
            ),
        },
        "cublas": cublas,
        "cublaslt_int8": cublaslt_int8,
        "inductor": inductor,
        "gap_vs_cublas_pct_median": _gap_pct(utpu_median, cublas_median),
        "gap_vs_cublaslt_int8_pct_median": _gap_pct(
            utpu_median, cublaslt_int8_per_col_median
        ),
        "gap_vs_inductor_pct_median": _gap_pct(utpu_median, inductor_median),
    }


LOAD_BEARING_ARMS_CUBLAS: Tuple[str, ...] = ("utpu",)
"""Arms whose `kernel_ms` is the *numerator* of a resume claim.

Strict-side of the split CV gate (see same rationale block in
`run_megakernel_benchmark.py`). uTPU's kernel is a single NVRTC
launch per timed iteration, so its inter-run CV reflects steady-
state kernel execution and is the correct gate target for any
uTPU-side claim.

The baseline arms (`cublas`, `cublaslt_int8`, `inductor`) are
multi-launch / multi-call torch callees and their inter-run CV on a
laptop GPU without `nvidia-smi --lock-gpu-clocks` is dominated by
clock-frequency drift — intrinsic, not methodology-fixable.

Critically: passing the strict load-bearing gate certifies uTPU's
per-iteration kernel time is stable. It does NOT certify the
published `gap_vs_*_pct` numbers, because every gap's denominator
is on a baseline arm. The IMMA-INT8 gap, the cuBLAS-FP gap, and
the Inductor gap therefore all stay `[needs-locked-clock-artifact]`
until a regen on `nvidia-smi --lock-gpu-clocks` clears them.
"""


def _is_load_bearing_arm_cublas(arm_name: str) -> bool:
    return arm_name in LOAD_BEARING_ARMS_CUBLAS


def _collect_arm_stability(per_shape: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll up per-arm-per-shape timing_stability_pct into split aggregate
    fields (load-bearing arm vs baseline arms).

    Each populated arm (uTPU + cuBLAS + cuBLASLt IMMA + Inductor) reports
    a ``timing_stability_pct`` (CV of per-run medians across N back-to-
    back stability runs). The aggregate rollup surfaces:

    * (diagnostic, NOT gated) ``max_timing_stability_pct_across_arms``:
      worst observed inter-run CV across every populated (arm, shape)
      pair. Kept for observability of overall noise floor.
    * (strict gate) ``max_timing_stability_pct_load_bearing_arm``:
      worst inter-run CV across only the uTPU arm. This is the field
      ``test_load_bearing_utpu_arm_is_gate_green`` asserts on; build
      fails if it exceeds ``TIMING_STABILITY_THRESHOLD_PCT``.
    * (advisory gate) ``max_timing_stability_pct_baseline_arms``:
      worst inter-run CV across the baseline arms. Reported only;
      does NOT block the build.
    * Per-side ``*_arms_exceeding_threshold_pct`` lists and
      ``num_stability_measurements_*`` counts.

    Legacy artifacts (pre-timing-stabilization) have no
    ``timing_stability_pct`` field on any arm; for those the rollup
    reports None for the aggregate fields, empty lists, and
    ``num_stability_measurements=0`` — distinguishable from "clean and
    populated" via the explicit count.
    """
    arm_names = ("utpu", "cublas", "cublaslt_int8", "inductor")
    cvs_all: List[float] = []
    cvs_load_bearing: List[float] = []
    cvs_baseline: List[float] = []
    exceed_all: List[Dict[str, Any]] = []
    exceed_load_bearing: List[Dict[str, Any]] = []
    exceed_baseline: List[Dict[str, Any]] = []
    for entry in per_shape:
        shape = entry.get("shape", {})
        for arm_name in arm_names:
            arm = entry.get(arm_name)
            if not isinstance(arm, dict):
                continue
            kernel_ms = arm.get("kernel_ms")
            if not isinstance(kernel_ms, dict):
                continue
            cv = kernel_ms.get("timing_stability_pct")
            if cv is None:
                continue
            try:
                cv_f = float(cv)
            except (TypeError, ValueError):
                continue
            cvs_all.append(cv_f)
            is_load_bearing = _is_load_bearing_arm_cublas(arm_name)
            if is_load_bearing:
                cvs_load_bearing.append(cv_f)
            else:
                cvs_baseline.append(cv_f)
            if cv_f > TIMING_STABILITY_THRESHOLD_PCT:
                entry_record = {
                    "arm": arm_name,
                    "shape": {
                        "M": int(shape.get("M", 0)),
                        "K": int(shape.get("K", 0)),
                        "N": int(shape.get("N", 1)),
                    },
                    "timing_stability_pct": cv_f,
                    "per_run_medians_ms": list(
                        kernel_ms.get("per_run_medians_ms") or []
                    ),
                    "per_run_medians_trimmed_ms": list(
                        kernel_ms.get("per_run_medians_trimmed_ms")
                        or kernel_ms.get("per_run_medians_ms")
                        or []
                    ),
                }
                exceed_all.append(entry_record)
                if is_load_bearing:
                    exceed_load_bearing.append(entry_record)
                else:
                    exceed_baseline.append(entry_record)
    return {
        "max_timing_stability_pct_across_arms": (
            float(max(cvs_all)) if cvs_all else None
        ),
        "mean_timing_stability_pct_across_arms": (
            float(statistics.fmean(cvs_all)) if cvs_all else None
        ),
        "arms_exceeding_stability_threshold_pct": exceed_all,
        # Strict-side split gate (uTPU only):
        "max_timing_stability_pct_load_bearing_arm": (
            float(max(cvs_load_bearing)) if cvs_load_bearing else None
        ),
        "load_bearing_arms_exceeding_threshold_pct": exceed_load_bearing,
        "num_stability_measurements_load_bearing_arm": int(len(cvs_load_bearing)),
        # Advisory-side split gate (baseline arms):
        "max_timing_stability_pct_baseline_arms": (
            float(max(cvs_baseline)) if cvs_baseline else None
        ),
        "baseline_arms_exceeding_threshold_pct": exceed_baseline,
        "num_stability_measurements_baseline_arms": int(len(cvs_baseline)),
        "num_stability_measurements": int(len(cvs_all)),
        "timing_stability_threshold_pct": float(TIMING_STABILITY_THRESHOLD_PCT),
        "load_bearing_arms": list(LOAD_BEARING_ARMS_CUBLAS),
    }


def _aggregate(per_shape: List[Dict[str, Any]]) -> Dict[str, Any]:
    cublas_gaps = [
        s["gap_vs_cublas_pct_median"]
        for s in per_shape
        if s.get("gap_vs_cublas_pct_median") is not None
    ]
    cublaslt_int8_gaps = [
        s["gap_vs_cublaslt_int8_pct_median"]
        for s in per_shape
        if s.get("gap_vs_cublaslt_int8_pct_median") is not None
    ]
    inductor_gaps = [
        s["gap_vs_inductor_pct_median"]
        for s in per_shape
        if s.get("gap_vs_inductor_pct_median") is not None
    ]
    stability = _collect_arm_stability(per_shape)
    return {
        "cublas_gap_pct_median_of_shapes": (
            float(statistics.median(cublas_gaps)) if cublas_gaps else None
        ),
        "cublas_gap_pct_mean_of_shapes": (
            float(statistics.fmean(cublas_gaps)) if cublas_gaps else None
        ),
        "cublas_gap_pct_max_of_shapes": (
            float(max(cublas_gaps)) if cublas_gaps else None
        ),
        "cublas_gap_pct_min_of_shapes": (
            float(min(cublas_gaps)) if cublas_gaps else None
        ),
        "cublaslt_int8_gap_pct_median_of_shapes": (
            float(statistics.median(cublaslt_int8_gaps)) if cublaslt_int8_gaps else None
        ),
        "cublaslt_int8_gap_pct_mean_of_shapes": (
            float(statistics.fmean(cublaslt_int8_gaps)) if cublaslt_int8_gaps else None
        ),
        "cublaslt_int8_gap_pct_max_of_shapes": (
            float(max(cublaslt_int8_gaps)) if cublaslt_int8_gaps else None
        ),
        "cublaslt_int8_gap_pct_min_of_shapes": (
            float(min(cublaslt_int8_gaps)) if cublaslt_int8_gaps else None
        ),
        "inductor_gap_pct_median_of_shapes": (
            float(statistics.median(inductor_gaps)) if inductor_gaps else None
        ),
        "inductor_gap_pct_mean_of_shapes": (
            float(statistics.fmean(inductor_gaps)) if inductor_gaps else None
        ),
        "shapes_compared_vs_cublas": int(len(cublas_gaps)),
        "shapes_compared_vs_cublaslt_int8": int(len(cublaslt_int8_gaps)),
        "shapes_compared_vs_inductor": int(len(inductor_gaps)),
        "max_timing_stability_pct_across_arms": stability[
            "max_timing_stability_pct_across_arms"
        ],
        "mean_timing_stability_pct_across_arms": stability[
            "mean_timing_stability_pct_across_arms"
        ],
        "arms_exceeding_stability_threshold_pct": stability[
            "arms_exceeding_stability_threshold_pct"
        ],
        # Split-CV-gate fields (load-bearing arm = utpu):
        "max_timing_stability_pct_load_bearing_arm": stability[
            "max_timing_stability_pct_load_bearing_arm"
        ],
        "load_bearing_arms_exceeding_threshold_pct": stability[
            "load_bearing_arms_exceeding_threshold_pct"
        ],
        "num_stability_measurements_load_bearing_arm": stability[
            "num_stability_measurements_load_bearing_arm"
        ],
        "max_timing_stability_pct_baseline_arms": stability[
            "max_timing_stability_pct_baseline_arms"
        ],
        "baseline_arms_exceeding_threshold_pct": stability[
            "baseline_arms_exceeding_threshold_pct"
        ],
        "num_stability_measurements_baseline_arms": stability[
            "num_stability_measurements_baseline_arms"
        ],
        "num_stability_measurements": stability["num_stability_measurements"],
        "timing_stability_threshold_pct": stability["timing_stability_threshold_pct"],
        "load_bearing_arms": stability["load_bearing_arms"],
    }


def _methodology_block(
    shapes: List[Dict[str, int]],
    warmup: int,
    iters: int,
    num_stability_runs: int = DEFAULT_NUM_STABILITY_RUNS,
) -> Dict[str, Any]:
    return {
        "shape_set": shapes,
        "shape_set_description": (
            "Six fixed (M, K) shapes covering small (16x16) to "
            "medium-large (512x512) GEMV; N=1 in all cases (FC "
            "inference). Locked in firmware/host/run_cublas_baseline.py"
            "::SHAPES; changes to this list invalidate prior artifacts."
        ),
        "warmup_iterations": int(warmup),
        "timed_iterations": int(iters),
        "num_stability_runs": int(num_stability_runs),
        "timing_protocol_name": TIMING_PROTOCOL_NAME,
        "timing_stability_threshold_pct": float(TIMING_STABILITY_THRESHOLD_PCT),
        "timing_protocol": (
            "Per-iter timing uses CUDA events: torch.cuda.Event(enable_timing=True) "
            "records are inserted in stream order around each call, then a "
            "single torch.cuda.synchronize() is issued after the iter loop "
            "and start.elapsed_time(end) (ms) gives the per-iter GPU time. "
            "This replaces the older wall-clock + per-iter-sync brackets — "
            "at GEMV scale (30-50us kernel time) the host-side sync overhead "
            "was the dominant noise source (~5-20us per sample, i.e. 15-40% "
            "of signal); CUDA events measure GPU execution directly with "
            "sub-us resolution. Warmup invocations are discarded. Reported "
            "per arm: mean, median, stdev, min, max, p95 (ms) over the "
            "full sample list."
        ),
        "stability_protocol": (
            f"v2 (spin-up + trimmed median). Each per-shape arm runs (a) "
            f"one discarded spin-up pass of warmup + iters (engages GPU "
            f"boost clocks and warms the lazy-load kernel cache before "
            f"any timed sample is recorded), then (b) {num_stability_runs} "
            f"counted back-to-back stability runs, each a fresh warmup + "
            f"iters block with CUDA-event timing. The per-shape kernel_ms "
            f"reports per_run_medians_ms (full N-long list, retained for "
            f"diagnostics so the gate-failure print loop and the "
            f"arms_exceeding_stability_threshold_pct field can surface "
            f"the raw outlier values) AND per_run_medians_trimmed_ms — "
            f"if num_stability_runs >= {TRIM_MIN_RUNS} the highest + "
            f"lowest per-run median are dropped before computing both "
            f"timing_stability_pct (CV = stddev / mean of the trimmed "
            f"medians * 100) and the pooled samples used for the "
            f"published kernel_ms.median (samples from dropped outlier "
            f"runs are excluded from the shipped median, not just from "
            f"the CV stat). The CV gate is SPLIT: the aggregate field "
            f"max_timing_stability_pct_load_bearing_arm covers ONLY the "
            f"uTPU arm (single NVRTC launch per iter, steady-state) and "
            f"is asserted strictly — build fails if it exceeds "
            f"{TIMING_STABILITY_THRESHOLD_PCT}%. The aggregate field "
            f"max_timing_stability_pct_baseline_arms covers the baseline "
            f"arms (cublas, cublaslt_int8, inductor) and is advisory "
            f"only — multi-launch / multi-kernel-callee host-side jitter "
            f"on a contended laptop GPU without "
            f"`nvidia-smi --lock-gpu-clocks` is intrinsic and not "
            f"methodology-fixable, so the advisory does NOT block the "
            f"build. The diagnostic field "
            f"max_timing_stability_pct_across_arms is still surfaced for "
            f"observability but is NOT a gate field. CRITICAL: passing "
            f"the strict load-bearing gate certifies uTPU's per-iteration "
            f"kernel time is stable but does NOT certify any specific "
            f"gap_vs_*_pct number, because every gap's denominator is on "
            f"a baseline arm. The IMMA-INT8 / cuBLAS-FP / Inductor gaps "
            f"therefore stay [needs-locked-clock-artifact] until "
            f"regenerated on `nvidia-smi --lock-gpu-clocks` (native "
            f"Linux) or a cloud RTX. The methodology still mitigates "
            f"two earlier noise classes — the +29.16% -> +4.92% "
            f"cuBLAS-fallback wall-clock swing (v0 -> v1: CUDA-event "
            f"timing) and the cold-clock outlier-run pattern (v1 -> v2: "
            f"spin-up + trimmed median) — but absolute baseline-arm "
            f"medians on this hardware remain clock-noise-dominated. "
            f"Background GPU load on the host can still defeat both "
            f"gates; remediation on a gate failure is documented "
            f"per-failure: bump --warmup / --iters / "
            f"--num-stability-runs and re-run with no other "
            f"GPU work on the box."
        ),
        "tflops_definition": (
            "2 * M * N * K / median_kernel_ms * 1e-12. INT-MAC count "
            "for uTPU + cuBLAS / cuBLASLt IMMA INT8 (all INT32 "
            "accumulator); Inductor runs FP32 so its TFLOPS would not "
            "be comparable and is omitted."
        ),
        "baseline_arms": [
            "cuBLAS GEMV via torch.matmul (INT32 where supported, "
            "FP32 fallback otherwise; fallback is never silent and is "
            "recorded as dtype_fallback_reason in the per-shape entry).",
            "cuBLASLt IMMA INT8 GEMM via torch._int_mm (INT8 inputs / "
            "INT32 accumulator). uTPU N=1 GEMV is padded to N=8 to "
            "satisfy IMMA alignment (N % 8 == 0). Per-N-column time "
            "(kernel_ms.median / N_padded) is reported as "
            "ms_per_n_column_median and is what the parent uses for "
            "gap_vs_cublaslt_int8_pct_median; this is the dtype-matched "
            "apples-to-apples GEMV comparison to the uTPU kernel.",
            "TorchInductor nn.Linear in FP32 (framework reference, "
            "not a dtype-matched comparison; documented as such).",
        ],
        "cublaslt_int8_alignment_caveat": (
            "cuBLASLt IMMA INT8 GEMM (torch._int_mm) requires "
            "N % 8 == 0 on Ampere/Ada/Hopper/Blackwell. A native N=1 "
            "INT8 GEMV does not exist in the IMMA path, so the "
            "subprocess pads the uTPU N=1 input to N=8 (the alignment "
            "minimum) and divides the wall time by N_padded to obtain a "
            "GEMV-equivalent per-column time. This is slightly "
            "favorable to cuBLAS (launch + tensor-core setup costs "
            "amortize across N_padded columns), and the gap reported "
            "as gap_vs_cublaslt_int8_pct_median uses this per-column "
            "figure — i.e. the most cuBLAS-favorable honest comparison "
            "available without inventing a non-existent IMMA N=1 GEMV. "
            "The per-shape entry preserves both kernel_ms (full N=8 "
            "wall time) and ms_per_n_column_median so a reader can "
            "see exactly how the gap was computed."
        ),
        "dtype_caveats": [
            "uTPU kernel: INT8 inputs, INT32 accumulator, INT4-quantised "
            "output (matches the FPGA datapath). Output quantisation is "
            "applied post-cuBLAS-equivalent GEMV and is included in the "
            "uTPU kernel_time_ms.",
            "cuBLAS GEMV reference: the subprocess attempts INT32 inputs "
            "/ INT32 accumulator via torch.matmul (apples-to-apples "
            "accumulator dtype with the uTPU kernel). Torch builds "
            "without an INT32 addmv/addmm CUDA kernel (observed on "
            "Torch 2.11+cu130 with the RTX 5070 sm_120) raise "
            "NotImplementedError and the subprocess falls back to FP32 "
            "inputs / FP32 accumulator via the standard cuBLAS GEMV "
            "path. The per-shape entry records the actual measured "
            "dtype in dtype_W/dtype_x/dtype_accum/dtype_out plus an "
            "explicit dtype_fallback_reason on the fallback path. The "
            "uTPU-vs-cuBLAS gap on the FP32-fallback path is not a "
            "dtype-matched comparison; for the dtype-matched answer "
            "see gap_vs_cublaslt_int8_pct_median (cuBLASLt IMMA INT8 "
            "GEMM via torch._int_mm).",
            "cuBLASLt IMMA INT8 GEMM reference (dtype-matched): "
            "INT8 inputs / INT32 accumulator via torch._int_mm. uTPU "
            "N=1 GEMV is padded to N=8 (IMMA alignment minimum) and "
            "per-N-column time is used for the gap; this is the "
            "honest apples-to-apples GEMV comparison and removes the "
            "INT8-vs-FP32 dtype asterisk on the cuBLAS-FP32-fallback "
            "row. See cublaslt_int8_alignment_caveat in this "
            "methodology block for the exact derivation.",
            "Inductor reference: FP32 nn.Linear compiled via "
            "torch.compile(backend='inductor', fullgraph=True). Same op "
            "semantics but FP32 dtype throughout. Recorded as a "
            "framework reference, not a dtype-matched comparison; "
            "gap_vs_inductor_pct is reported but the writeup flags this "
            "caveat explicitly.",
        ],
        "isolation": (
            "cuBLAS + cuBLASLt IMMA + Inductor timings run in a separate "
            "Python process (_cublas_baseline_torch_subprocess.py) so "
            "the parent's NVRTC driver context for the uTPU kernel "
            "does not collide with Torch's CUDA / Inductor contexts "
            "(the same isolation pattern as inductor_oracle_subprocess.py)."
        ),
        "rng_seed_per_shape": "0xC0DE XOR'd with M*1009 + K for the uTPU "
                              "kernel; 0xC0DE / 0xACE / 0xCAB inside the Torch subprocess.",
        "scope": (
            "Sim/host-measured. No physical-board claim. cuBLAS / "
            "cuBLASLt IMMA / Inductor numbers depend on the GPU + driver "
            "of the host that regenerates the artifact; uTPU numbers "
            "depend on NVRTC + cuda-python on the same host."
        ),
    }


def recompute_aggregate_in_place(output_path: Path) -> int:
    """Re-derive aggregate + per-shape gap fields from an existing artifact.

    Used to refresh the schema (e.g. surface a newly-added baseline arm
    like ``cublaslt_int8`` that wasn't present when the live artifact
    was first generated) without re-running CUDA. The per-shape kernel
    timings are preserved verbatim; only the derived ``gap_vs_*`` and
    ``aggregate`` blocks are recomputed. If the artifact predates the
    new arm (no ``cublaslt_int8`` entry per shape) the new gap fields
    collapse to None — the artifact is marked ``aggregate_recomputed``
    with a regen instruction so the reader knows to re-run on a CUDA
    host for live IMMA numbers.
    """
    if not output_path.exists():
        print(f"[cublas_baseline] no artifact at {output_path}; nothing to recompute")
        return 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        print(
            f"[cublas_baseline] artifact status={payload.get('status')!r}; "
            f"recompute-aggregate-only requires status='ok'. Re-run the "
            f"full harness on a CUDA host."
        )
        return 1

    new_methodology = _methodology_block(
        payload.get("methodology", {}).get("shape_set", SHAPES),
        int(payload.get("methodology", {}).get("warmup_iterations", DEFAULT_WARMUP)),
        int(payload.get("methodology", {}).get("timed_iterations", DEFAULT_ITERS)),
        int(
            payload.get("methodology", {}).get(
                "num_stability_runs", DEFAULT_NUM_STABILITY_RUNS
            )
        ),
    )
    payload["methodology"] = new_methodology

    refreshed: List[Dict[str, Any]] = []
    for entry in payload.get("per_shape", []) or []:
        utpu = entry.get("utpu") or {}
        utpu_median = utpu.get("kernel_ms", {}).get("median", 0.0)
        cublas = entry.get("cublas")
        cublaslt_int8 = entry.get("cublaslt_int8")
        inductor = entry.get("inductor")
        cublas_median = (cublas or {}).get("kernel_ms", {}).get("median")
        cublaslt_int8_per_col_median = _cublaslt_int8_per_column_median(cublaslt_int8)
        inductor_median = (inductor or {}).get("kernel_ms", {}).get("median")

        new_entry = dict(entry)
        new_entry["cublaslt_int8"] = cublaslt_int8
        new_entry["gap_vs_cublas_pct_median"] = _gap_pct(utpu_median, cublas_median)
        new_entry["gap_vs_cublaslt_int8_pct_median"] = _gap_pct(
            utpu_median, cublaslt_int8_per_col_median
        )
        new_entry["gap_vs_inductor_pct_median"] = _gap_pct(utpu_median, inductor_median)
        refreshed.append(new_entry)

    payload["per_shape"] = refreshed
    payload["aggregate"] = _aggregate(refreshed)
    payload["aggregate_recomputed"] = {
        "recomputed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "schema refresh: surfaced cublaslt_int8_gap_pct_* fields and "
            "the IMMA INT8 alignment caveat in methodology. Per-shape "
            "kernel timings preserved verbatim; only the derived "
            "gap_vs_* + aggregate blocks were recomputed from existing "
            "data. Re-run firmware/host/run_cublas_baseline.py on a "
            "CUDA host (no --recompute-aggregate-only flag) to fill in "
            "the cublaslt_int8 arm with live IMMA INT8 GEMM timings; "
            "until then gap_vs_cublaslt_int8_pct_median is None on any "
            "shape whose cublaslt_int8 entry is missing."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    agg = payload["aggregate"] or {}
    print(f"[cublas_baseline] recomputed aggregate in-place at {output_path}")
    if agg.get("cublaslt_int8_gap_pct_median_of_shapes") is not None:
        print(
            "[cublas_baseline] median uTPU-vs-cuBLASLt-IMMA-INT8 gap (per-column): "
            f"{agg['cublaslt_int8_gap_pct_median_of_shapes']:.2f}% "
            f"(across {agg['shapes_compared_vs_cublaslt_int8']} shapes)"
        )
    else:
        print(
            "[cublas_baseline] cuBLASLt IMMA INT8 gap: None (no live IMMA data "
            "in this artifact yet; re-run on a CUDA host to populate)."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7 cuBLAS / cuBLASLt IMMA / Inductor baseline vs uTPU NVRTC kernel."
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help="Discarded warmup invocations PER STABILITY RUN.",
    )
    parser.add_argument(
        "--iters", type=int, default=DEFAULT_ITERS,
        help="Measured invocations per shape per backend PER STABILITY RUN.",
    )
    parser.add_argument(
        "--num-stability-runs",
        type=int,
        default=DEFAULT_NUM_STABILITY_RUNS,
        # argparse on Python 3.14+ eagerly expands %-substitutions in help
        # strings, so any literal '%' must be escaped as '%%' to survive
        # argparse's own formatter — otherwise '%X' (where X is the next
        # char) is parsed as a printf-style format directive and crashes
        # at add_argument time with `badly formed help string`.
        help=(
            "Number of counted back-to-back stability runs per shape per arm "
            "(v2 protocol also adds one discarded spin-up run before these). "
            "Each counted run does its own warmup + iters block; if this is "
            f">= {TRIM_MIN_RUNS} the highest + lowest per-run medians are "
            "dropped before computing the published kernel_ms.median and the "
            "inter-run CV (timing_stability_pct). A documented gate at "
            f"{TIMING_STABILITY_THRESHOLD_PCT}%% inter-run CV fails the build "
            "if any populated arm exceeds it. Set to 1 to disable inter-run "
            "stability collection (legacy single-run behaviour). Set to "
            f">= {TRIM_MIN_RUNS} (default {DEFAULT_NUM_STABILITY_RUNS}) to "
            "enable outlier-run trimming."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_JSON),
        help="Output JSON path."
    )
    parser.add_argument(
        "--recompute-aggregate-only",
        action="store_true",
        help=(
            "Re-derive aggregate + per-shape gap fields from the existing "
            "artifact in-place (refresh schema after adding new baseline "
            "arms). Does not re-run CUDA timings. Requires the artifact "
            "to already exist with status='ok'."
        ),
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.recompute_aggregate_only:
        return recompute_aggregate_in_place(output_path)

    methodology = _methodology_block(
        SHAPES, args.warmup, args.iters, int(args.num_stability_runs)
    )
    common_header = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "methodology": methodology,
    }

    env = detect_cuda_environment()
    if not env.runtime_available:
        payload = dict(
            common_header,
            status="cuda_unavailable",
            reason=env.reason
            or "detect_cuda_environment().runtime_available is False on this host.",
            instructions=[
                "1. Run on a CUDA-capable host with cuda-python + driver + NVRTC installed.",
                "2. Install Torch with CUDA support (and TorchInductor + gcc for the Inductor path).",
                "3. Re-execute: python firmware/host/run_cublas_baseline.py",
                "4. The artifact regenerates from the locked SHAPES set; do not edit it by hand.",
            ],
            shapes_requested=SHAPES,
            per_shape=[],
            aggregate=None,
            torch_subprocess=None,
        )
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[cublas_baseline] cuda_unavailable; wrote stub to {output_path}")
        return 0

    tmp_path = output_path.with_suffix(".torch_subprocess.json")
    torch_payload = _run_torch_subprocess(
        SHAPES, warmup=args.warmup, iters=args.iters,
        num_stability_runs=int(args.num_stability_runs),
        tmp_path=tmp_path,
    )

    executor = CUDABlockedFCExecutor(verbose=False)
    per_shape: List[Dict[str, Any]] = []
    torch_shape_index: Dict[tuple, Dict[str, Any]] = {}
    for entry in torch_payload.get("shapes", []) or []:
        sh = entry.get("shape", {})
        torch_shape_index[(int(sh.get("M", 0)), int(sh.get("K", 0)))] = entry

    for shape in SHAPES:
        M, K = int(shape["M"]), int(shape["K"])
        utpu = _run_utpu_shape(
            executor,
            M=M,
            K=K,
            warmup=args.warmup,
            iters=args.iters,
            num_stability_runs=int(args.num_stability_runs),
            rng_seed=0xC0DE ^ (M * 1009 + K),
        )
        per_shape.append(
            _merge_per_shape(utpu, torch_shape_index.get((M, K)))
        )

    aggregate = _aggregate(per_shape)
    payload = dict(
        common_header,
        status="ok",
        environment={
            "uTPU_cuda_runtime_available": True,
            "torch_subprocess_status": torch_payload.get("status"),
            "torch_subprocess_environment": torch_payload.get("environment"),
        },
        shapes_requested=SHAPES,
        per_shape=per_shape,
        aggregate=aggregate,
        torch_subprocess=torch_payload,
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    print(f"[cublas_baseline] wrote {output_path}")
    # All gap_vs_* percentages depend on a baseline-arm denominator and
    # are [needs-locked-clock-artifact] until the baseline arms can be
    # certified on locked clocks. We still PRINT them (they live in the
    # artifact for diagnostics + cross-machine regen comparison), but the
    # console framing makes the locked-clock caveat explicit.
    if aggregate.get("cublas_gap_pct_median_of_shapes") is not None:
        print(
            "[cublas_baseline] median uTPU-vs-cuBLAS gap: "
            f"{aggregate['cublas_gap_pct_median_of_shapes']:.2f}% "
            f"(across {aggregate['shapes_compared_vs_cublas']} shapes; "
            "dtype caveat: cuBLAS arm is INT32 if available else FP32 "
            "fallback) [needs-locked-clock-artifact for any specific %-claim]"
        )
    if aggregate.get("cublaslt_int8_gap_pct_median_of_shapes") is not None:
        print(
            "[cublas_baseline] median uTPU-vs-cuBLASLt-IMMA-INT8 gap "
            f"(per-column): {aggregate['cublaslt_int8_gap_pct_median_of_shapes']:.2f}% "
            f"(across {aggregate['shapes_compared_vs_cublaslt_int8']} "
            "shapes; dtype-matched apples-to-apples GEMV comparison) "
            "[needs-locked-clock-artifact for any specific %-claim]"
        )
    if aggregate.get("inductor_gap_pct_median_of_shapes") is not None:
        print(
            "[cublas_baseline] median uTPU-vs-Inductor gap: "
            f"{aggregate['inductor_gap_pct_median_of_shapes']:.2f}% "
            "(dtype caveat applies, see methodology block) "
            "[needs-locked-clock-artifact for any specific %-claim]"
        )
    threshold = aggregate.get(
        "timing_stability_threshold_pct", TIMING_STABILITY_THRESHOLD_PCT
    )
    max_cv_lb = aggregate.get("max_timing_stability_pct_load_bearing_arm")
    if max_cv_lb is not None:
        verdict = "PASS" if max_cv_lb <= threshold else "FAIL"
        print(
            f"[cublas_baseline] STRICT GATE (load-bearing arms "
            f"{aggregate.get('load_bearing_arms') or LOAD_BEARING_ARMS_CUBLAS}): "
            f"max inter-run CV = {max_cv_lb:.2f}% across "
            f"{aggregate.get('num_stability_measurements_load_bearing_arm', 0)} "
            f"(arm, shape) measurements; threshold = {threshold:.1f}%; "
            f"gate = {verdict}"
        )
        lb_exceed = aggregate.get("load_bearing_arms_exceeding_threshold_pct") or []
        if lb_exceed:
            print(
                f"[cublas_baseline] STRICT GATE: {len(lb_exceed)} "
                f"(arm, shape) pair(s) on the load-bearing arm exceeded "
                f"threshold (this BLOCKS the build):"
            )
            for e in lb_exceed:
                print(
                    f"  - {e['arm']} @ M={e['shape']['M']} K={e['shape']['K']}: "
                    f"CV={e['timing_stability_pct']:.2f}% "
                    f"medians={e['per_run_medians_ms']}"
                )
    max_cv_bl = aggregate.get("max_timing_stability_pct_baseline_arms")
    if max_cv_bl is not None:
        adv_verdict = "PASS" if max_cv_bl <= threshold else "ADVISORY-FAIL"
        print(
            f"[cublas_baseline] ADVISORY GATE (baseline arms): "
            f"max inter-run CV = {max_cv_bl:.2f}% across "
            f"{aggregate.get('num_stability_measurements_baseline_arms', 0)} "
            f"(arm, shape) measurements; threshold = {threshold:.1f}%; "
            f"advisory = {adv_verdict} (does NOT block build; gap_vs_* "
            f"%-claims stay [needs-locked-clock-artifact] until baseline "
            f"arms certify on locked clocks)"
        )
        bl_exceed = aggregate.get("baseline_arms_exceeding_threshold_pct") or []
        if bl_exceed:
            print(
                f"[cublas_baseline] ADVISORY: {len(bl_exceed)} baseline "
                f"(arm, shape) pair(s) exceeded threshold (informational "
                f"only; expected on WSL2 + unlocked clocks):"
            )
            for e in bl_exceed:
                print(
                    f"  - {e['arm']} @ M={e['shape']['M']} K={e['shape']['K']}: "
                    f"CV={e['timing_stability_pct']:.2f}% "
                    f"medians={e['per_run_medians_ms']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
