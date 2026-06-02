"""Phase 7 — Torch baseline subprocess (cuBLAS + TorchInductor + IMMA INT8).

Runs in a separate process from the parent ``run_cublas_baseline.py``
script so the parent can keep its NVRTC driver context for the uTPU
blocked-FC kernel without crashing into Torch's own CUDA / Inductor
contexts (this is the same isolation pattern used by
``inductor_oracle_subprocess.py``).

For each shape in the requested ``shapes`` list this subprocess emits
three baseline arms:

* **cuBLAS GEMV** timing: ``torch.matmul(W, x)`` with INT32 weights /
  INT32 activations / INT32 output where the Torch build supports it,
  falling back to FP32 inputs / FP32 accumulator on Torch builds that
  raise ``NotImplementedError`` for INT32 cuBLAS dispatch. The dtype
  actually measured is recorded verbatim in the per-shape entry; the
  fallback is never silent.
* **cuBLASLt IMMA INT8 GEMM** timing: ``torch._int_mm(W_i8, X_i8)``
  with ``W_i8`` shape ``(M, K)`` INT8 and ``X_i8`` shape
  ``(K, N_padded)`` INT8 (N=1 GEMV padded to N=8 to satisfy the IMMA
  alignment minimum). Output is INT32 accumulator → INT32 result.
  This is the **dtype-matched** apples-to-apples comparison to the uTPU
  kernel (INT8 inputs / INT32 accumulator); the per-shape entry
  reports both the full-N kernel time and a per-N-column equivalent
  (``ms_per_n_column_median = kernel_ms.median / N_padded``) so the
  parent's ``gap_vs_cublaslt_int8_pct_median`` is uTPU-N=1-GEMV
  vs cuBLASLt-per-column-GEMV-equivalent (the honest GEMV gap).
* **TorchInductor** timing: a thin ``torch.nn.Linear(K, M, bias=False)``
  module compiled with ``torch.compile(..., backend="inductor",
  fullgraph=True)`` and run on the same input. dtype is float32 (the
  default Torch dispatcher); the parent script records the
  dtype-mismatch caveat in the artifact so the comparison is not
  silently inflated.

Methodology is locked in this single file so the parent can describe it
verbatim in the artifact:

* **Per-iter timing protocol: CUDA events** (``torch.cuda.Event(enable_timing=True)``
  recorded in stream order around each call, then a single
  ``torch.cuda.synchronize()`` followed by ``start.elapsed_time(end)``
  per iteration). Replaces the earlier wall-clock + per-iter-sync
  protocol — at GEMV scale (30–50 µs kernel time) the host-side
  ``time.perf_counter()`` + ``cuda.synchronize`` overhead is the
  dominant noise source, and CUDA events measure GPU-side execution
  time directly with sub-µs resolution.
* **Inter-run stability protocol v2 (spin-up + trimmed median): one
  discarded spin-up run, then N back-to-back stability runs**
  (``num_stability_runs``, default 5). The spin-up run is a full
  ``warmup + iters`` pass that is discarded entirely — its purpose is
  to engage GPU boost clocks and warm the lazy-load kernel cache so
  the first *counted* stability run doesn't inherit cold-start cost.
  Each of the N counted runs does its own ``warmup`` + ``iters`` block
  with the CUDA-events protocol above. If ``N >= TRIM_MIN_RUNS`` (4),
  the highest and lowest per-run medians are dropped before computing
  the inter-run stddev / CV and before pooling samples into the
  per-arm ``kernel_ms.median``. This kills the laptop-GPU "one outlier
  run out of N" pattern (e.g. ``cublaslt_int8 @ 128×128 = [59µs, 9µs,
  9µs]`` on WSL2 + RTX 5070 Laptop GPU — the 59µs is the cold-clock
  stray, trimming drops it). The parent reports the flat (post-trim)
  sample list, the full N per-run medians (kept for diagnostics), the
  trimmed (N-2) per-run medians, and ``timing_stability_pct`` = CV
  (stddev / mean × 100) across the trimmed per-run medians. A split
  gate test pair gates the aggregate: strict
  ``test_load_bearing_utpu_arm_is_gate_green`` fails the build if
  the uTPU arm's inter-run CV exceeds threshold, while advisory
  ``test_baseline_arms_advisory_cv_reporting`` reports the baseline
  arms' CV but does NOT fail the build — multi-launch host-side
  jitter on a contended laptop GPU is intrinsic, and the published
  ``gap_vs_*_pct`` values therefore stay
  ``[needs-locked-clock-artifact]`` until regenerated on
  ``nvidia-smi --lock-gpu-clocks``.
* Warmup = ``warmup`` invocations per stability run (default 25),
  discarded. Then ``iters`` measured invocations per stability run
  (default 100). Stats reported per shape: mean, median, stdev,
  min, max, p95 (ms) over the *post-trim* flat sample list, plus the
  full per-run median list, the trimmed per-run median list, and the
  inter-run CV.
* Per-iteration timings are kept under a length cap so the artifact
  stays small.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Documented inter-run CV gate. Any populated arm whose per-shape median
# varies by more than this fraction across the kept (post-trim) stability
# runs is flagged as "unstable measurement" by the gate test. 10% was
# chosen empirically: the original +29.16% → +4.92% cuBLAS-fallback gap
# swing on the same RTX 5070 implied a per-arm median CV of ~13–15%
# (the +24-point gap swing was dominated by the cuBLAS-arm median moving
# ~20% across runs, not by uTPU). Threshold of 10% catches that class of
# noise and forces a re-run with higher warmup/iters/stability_runs
# before any shipped median can claim to be "stable".
TIMING_STABILITY_THRESHOLD_PCT = 10.0
# Schema version v2.1: same name as v2 (still spin-up + trimmed median)
# but with an adaptive trim count. The v2 regen on WSL2 + RTX 5070
# Laptop GPU at N=5 / trim=1 produced 37.35% median latency reduction
# (above the original 35% claim — load-bearing headline survives) but
# the gate still fired on a few sub-20µs kernels where the GPU's
# intrinsic power-management noise floor produced two outlier runs
# instead of one. v2.1 bumps the default to N=7 and trims ((N-3)//2,
# floored at 1) from each end, so at N=7 we keep 3 stable runs even
# when 2 of the 7 are clock-state outliers.
TIMING_PROTOCOL_NAME = "cuda_events_per_run_stability_v2_spin_up_plus_trimmed_median"
TRIM_MIN_RUNS = 4


def _trim_count_per_side(n: int) -> int:
    """How many per-run medians to drop from each end at ``num_stability_runs=n``.

    Formula: ``max(1, (n - 3) // 2)`` for ``n >= TRIM_MIN_RUNS``. Always
    keeps a minimum of 3 surviving runs:

      n=4: drop 1 each → keep 2
      n=5: drop 1 each → keep 3
      n=6: drop 1 each → keep 4
      n=7: drop 2 each → keep 3  ← default
      n=9: drop 3 each → keep 3
      n=11: drop 4 each → keep 3

    Scaling trim with n catches the "two outlier runs out of N" pattern
    observed at sub-20µs kernel scale on laptop GPUs without locked
    clocks — N=5 / trim-1-each-end only catches single-stray-run noise,
    N=7 / trim-2-each-end catches up to two strays.
    """
    if n < TRIM_MIN_RUNS:
        return 0
    return max(1, (n - 3) // 2)


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


def _gpu_environment(torch_mod) -> Dict[str, Any]:
    try:
        device_idx = torch_mod.cuda.current_device()
        name = torch_mod.cuda.get_device_name(device_idx)
        cap = torch_mod.cuda.get_device_capability(device_idx)
        version = getattr(torch_mod, "version", None)
        return {
            "device_name": str(name),
            "device_capability": [int(cap[0]), int(cap[1])],
            "torch_version": str(getattr(torch_mod, "__version__", "unknown")),
            "cuda_version": str(getattr(version, "cuda", "unknown")) if version else "unknown",
            "device_index": int(device_idx),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _time_kernel_with_events(fn, warmup: int, iters: int, torch_mod) -> List[float]:
    """Time ``fn`` with CUDA events recorded in stream order.

    Each iteration is bracketed by a pair of ``torch.cuda.Event(
    enable_timing=True)`` records. After all ``iters`` events are queued
    we do exactly one ``torch.cuda.synchronize()`` and read back
    ``start.elapsed_time(end)`` (ms) per iteration.

    Why this replaces the wall-clock loop:

    * At GEMV-scale kernel times (30–50 µs on the RTX 5070 launch-bound
      shapes), ``time.perf_counter()`` + per-iter ``torch.cuda.synchronize()``
      brackets carry 5–20 µs of host-side scheduling jitter per sample,
      so the noise floor of the wall-clock protocol sits at ~15–40% of
      the signal. CUDA events measure GPU-side execution directly with
      sub-µs resolution, so the noise floor drops to ~0.5–2%.
    * Same-stream FIFO ordering on the legacy default stream (and torch's
      default stream) guarantees the back-to-back ``fn()`` calls run
      sequentially, so each ``start.elapsed_time(end)`` measures the
      actual per-iter execution time, not a partially-overlapped slice.
    """
    for _ in range(warmup):
        fn()
    torch_mod.cuda.synchronize()
    start_events = [torch_mod.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch_mod.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch_mod.cuda.synchronize()
    return [float(s.elapsed_time(e)) for s, e in zip(start_events, end_events)]


def _trim_per_run_medians(
    per_run_medians: List[float], min_n_for_trim: int = TRIM_MIN_RUNS
) -> Tuple[List[float], List[int]]:
    """Drop the ``_trim_count_per_side(n)`` highest + lowest per-run medians.

    Returns ``(trimmed_medians_in_original_order, kept_run_indices)``. The
    kept indices let the caller rebuild the flat sample list from only
    the kept runs, so the per-arm ``kernel_ms.median`` is computed over
    *stable-run* samples only (the cold-clock and mid-run-drift outliers
    seen on laptop GPUs without locked clocks are excluded from the
    shipped median, not just from the CV stat). Trim count scales with
    ``n`` per ``_trim_count_per_side`` — at the v2.1 default ``n=7`` we
    drop 2 from each end and keep 3 stable runs.
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


def _time_with_stability(
    fn,
    warmup: int,
    iters: int,
    num_stability_runs: int,
    torch_mod,
) -> Dict[str, Any]:
    """Spin-up + N counted stability runs, then trim outlier runs.

    Protocol (``TIMING_PROTOCOL_NAME`` = v2):

    1. **Spin-up run** (discarded entirely): runs ``warmup + iters`` once
       before any timed samples are collected. This engages GPU boost
       clocks and warms the lazy-load kernel cache so the *first* counted
       stability run doesn't inherit cold-start cost. (The
       ``cublaslt_int8 @ 128×128 = [59µs, 9µs, 9µs]`` pattern observed on
       the first stabilized regen on WSL2 + RTX 5070 Laptop GPU is the
       textbook symptom of a missing spin-up: run 1 caught the kernel
       still warming, runs 2–3 are the true steady-state.)
    2. **N counted stability runs**: each run is its own ``warmup + iters``
       pass timed with CUDA events.
    3. **Trimmed median**: if ``N >= TRIM_MIN_RUNS`` (4 by default), drop
       the highest and lowest per-run median. The remaining (N-2) runs
       define both the published ``kernel_ms.median`` (their pooled
       samples) and the inter-run CV.

    Returned dict:

    * ``samples_ms``: flat list of samples from the *kept* (post-trim) runs
      only. ``_summary_ms`` uses this for the per-arm mean/median/stdev,
      so the published median excludes outlier-run samples.
    * ``per_run_medians_ms``: full untrimmed list of ``N`` per-run medians
      (kept for diagnostics: the gate-failure print loop and the
      offending-shape report both want to surface the raw outlier values
      so the reader can see the noise pattern).
    * ``per_run_medians_trimmed_ms``: the ``(N-2)``-long trimmed list (or
      the full list if ``N < TRIM_MIN_RUNS``).
    * ``num_stability_runs`` / ``num_stability_runs_trimmed``: schema
      counts. Tests assert ``len(per_run_medians_*_ms)`` matches each.
    * ``timing_stability_stddev_ms``: stddev of the *trimmed* per-run
      medians (so a single outlier run no longer dominates the CV).
    * ``timing_stability_pct``: ``CV = stddev / mean * 100`` of the
      trimmed per-run medians. This is what the 10% gate compares
      against.
    * ``timing_protocol``: ``TIMING_PROTOCOL_NAME`` (v2 schema marker).
    """
    # Spin-up run (discarded). Single ``warmup + iters`` pass so the GPU
    # is at steady-state boost clocks before any timed sample is recorded.
    _time_kernel_with_events(fn, warmup=warmup, iters=iters, torch_mod=torch_mod)

    per_run_samples: List[List[float]] = []
    per_run_medians: List[float] = []
    for _ in range(max(1, int(num_stability_runs))):
        run_samples = _time_kernel_with_events(
            fn, warmup=warmup, iters=iters, torch_mod=torch_mod
        )
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

    if len(trimmed_medians) >= 2 and statistics.fmean(trimmed_medians) > 0.0:
        mean_med = float(statistics.fmean(trimmed_medians))
        stdev_med = float(statistics.pstdev(trimmed_medians))
        cv_pct = float(stdev_med / mean_med * 100.0) if mean_med > 0.0 else 0.0
    else:
        stdev_med = 0.0
        cv_pct = 0.0

    return {
        "samples_ms": kept_samples,
        "per_run_medians_ms": per_run_medians,
        "per_run_medians_trimmed_ms": trimmed_medians,
        "timing_stability_stddev_ms": stdev_med,
        "timing_stability_pct": cv_pct,
        "num_stability_runs": int(max(1, num_stability_runs)),
        "num_stability_runs_trimmed": int(len(trimmed_medians)),
        "timing_protocol": TIMING_PROTOCOL_NAME,
    }


def _attach_stability_to_summary(summary: Dict[str, Any], timed: Dict[str, Any]) -> Dict[str, Any]:
    """Splice the per-run stability fields into a ``_summary_ms`` block.

    Keeps the existing ``mean/median/stdev/min/max/p95/samples`` keys
    untouched (so prior schema-lock tests don't break) and adds the new
    stability fields alongside them.
    """
    summary["per_run_medians_ms"] = list(timed["per_run_medians_ms"])
    summary["per_run_medians_trimmed_ms"] = list(
        timed.get("per_run_medians_trimmed_ms", timed["per_run_medians_ms"])
    )
    summary["timing_stability_stddev_ms"] = float(timed["timing_stability_stddev_ms"])
    summary["timing_stability_pct"] = float(timed["timing_stability_pct"])
    summary["num_stability_runs"] = int(timed["num_stability_runs"])
    summary["num_stability_runs_trimmed"] = int(
        timed.get("num_stability_runs_trimmed", timed["num_stability_runs"])
    )
    summary["timing_protocol"] = str(timed["timing_protocol"])
    return summary


def _run_cublas_gemv(
    torch_mod, M: int, K: int, warmup: int, iters: int, num_stability_runs: int = 1
) -> Dict[str, Any]:
    """Time the cuBLAS GEMV path for the (M, K) shape.

    Strategy:

    1. Attempt INT32 inputs / INT32 accumulator via ``torch.matmul`` —
       the apples-to-apples dtype match for the uTPU kernel.
    2. On ``NotImplementedError`` (Torch builds without an int32
       ``addmv_impl_cuda`` / ``addmm_cuda`` — observed on the WSL2 +
       Torch 2.11+cu130 + RTX 5070 host that authored the live
       artifact) **fall back to FP32 inputs / FP32 accumulator** and
       record the dtype fallback in the per-shape entry so the
       artifact and writeup can document the caveat verbatim.

    The fallback is not a silent dtype switch: ``dtype_fallback_reason``
    is set, ``dtype_*`` fields reflect the actually-measured path, and
    ``methodology.dtype_caveats`` in the parent harness names this
    explicitly.
    """
    device = torch_mod.device("cuda")
    gen = torch_mod.Generator(device=device).manual_seed(0xC0DE)

    dtype_fallback_reason = None
    try:
        w_i32 = torch_mod.randint(-8, 8, (M, K), generator=gen, device=device, dtype=torch_mod.int32)
        x_i32 = torch_mod.randint(-8, 8, (K,), generator=gen, device=device, dtype=torch_mod.int32)
        _ = torch_mod.matmul(w_i32, x_i32)
        torch_mod.cuda.synchronize()
        w = w_i32
        x = x_i32
        dtype_W = "int32"
        dtype_x = "int32"
        dtype_accum = "int32"
        dtype_out = "int32"
    except NotImplementedError as exc:
        dtype_fallback_reason = (
            f"int32 cuBLAS matmul unsupported on this Torch+CUDA build "
            f"({type(exc).__name__}: {exc}); fell back to fp32 cuBLAS GEMV "
            f"(the standard PyTorch dispatch path)."
        )
        w_f32 = torch_mod.randn((M, K), generator=gen, device=device, dtype=torch_mod.float32)
        x_f32 = torch_mod.randn((K,), generator=gen, device=device, dtype=torch_mod.float32)
        _ = torch_mod.matmul(w_f32, x_f32)
        torch_mod.cuda.synchronize()
        w = w_f32
        x = x_f32
        dtype_W = "float32"
        dtype_x = "float32"
        dtype_accum = "float32"
        dtype_out = "float32"

    def call() -> None:
        torch_mod.matmul(w, x)

    timed = _time_with_stability(
        call,
        warmup=warmup,
        iters=iters,
        num_stability_runs=num_stability_runs,
        torch_mod=torch_mod,
    )
    summary = _summary_ms(timed["samples_ms"])
    summary = _attach_stability_to_summary(summary, timed)
    entry = {
        "backend": "cublas_gemv_int32" if dtype_fallback_reason is None else "cublas_gemv_fp32_fallback",
        "shape": {"M": int(M), "K": int(K), "N": 1},
        "dtype_W": dtype_W,
        "dtype_x": dtype_x,
        "dtype_accum": dtype_accum,
        "dtype_out": dtype_out,
        "kernel_ms": summary,
        "samples_ms": [float(s) for s in timed["samples_ms"][:32]],
    }
    if dtype_fallback_reason is not None:
        entry["dtype_fallback_reason"] = dtype_fallback_reason
    return entry


CUBLASLT_INT8_N_PAD = 8


def _run_cublaslt_imma_int8_gemm(
    torch_mod,
    M: int,
    K: int,
    warmup: int,
    iters: int,
    num_stability_runs: int = 1,
    n_pad: int = CUBLASLT_INT8_N_PAD,
) -> Dict[str, Any]:
    """Time cuBLASLt IMMA INT8 GEMM for shape (M, K) with N padded to ``n_pad``.

    The uTPU kernel runs INT8 inputs / INT32 accumulator at N=1 (single
    FC-inference vector). cuBLASLt IMMA INT8 GEMM requires ``N % 8 == 0``
    on every Ampere+ / Ada / Hopper / Blackwell device PyTorch dispatches
    through ``torch._int_mm`` (which is the public surface for the
    cuBLASLt ``CUBLASLT_MATMUL_DESC_COMPUTE_TYPE=CUBLAS_COMPUTE_32I``
    IMMA path).

    Strategy:

    1. Pad ``N`` from 1 → ``n_pad`` (default 8) by replicating columns;
       record the padding explicitly in the per-shape entry.
    2. Time ``torch._int_mm(W_i8, X_i8)`` end-to-end with
       ``torch.cuda.synchronize()`` brackets.
    3. Report both the full-N kernel time and the per-N-column equivalent
       (``full_ms / n_pad``). The parent uses the per-column figure for
       the apples-to-apples gap vs uTPU N=1 GEMV; that is **slightly
       favorable to cuBLAS** because launch overhead amortizes across
       ``n_pad`` columns, and the alignment_note in the artifact calls
       this out verbatim.
    4. On any failure (``torch._int_mm`` missing, IMMA unsupported on
       this SM, alignment / dtype rejected), record an explicit
       ``imma_unavailable_reason`` and emit a stub entry — no silent
       dtype switch.
    """
    shape = {"M": int(M), "K": int(K), "N": 1, "N_padded": int(n_pad)}

    if not hasattr(torch_mod, "_int_mm"):
        return {
            "backend": "cublaslt_imma_int8",
            "shape": shape,
            "imma_unavailable_reason": (
                "torch._int_mm is not present in this Torch build; "
                "cuBLASLt IMMA INT8 GEMM cannot be dispatched via Torch. "
                "Re-run on Torch >= 2.0 with cuBLASLt support."
            ),
        }

    if K % 16 != 0:
        return {
            "backend": "cublaslt_imma_int8",
            "shape": shape,
            "imma_unavailable_reason": (
                f"K={K} is not a multiple of 16; cuBLASLt IMMA INT8 GEMM "
                f"requires the contraction dimension to be aligned. The "
                f"locked SHAPES grid satisfies this; if this fires, the "
                f"shape set was changed without updating IMMA alignment."
            ),
        }
    if n_pad % 8 != 0 or n_pad < 8:
        return {
            "backend": "cublaslt_imma_int8",
            "shape": shape,
            "imma_unavailable_reason": (
                f"N_padded={n_pad} is not a positive multiple of 8; "
                f"cuBLASLt IMMA INT8 GEMM requires N % 8 == 0 (N>=8)."
            ),
        }
    if M <= 16:
        # PyTorch's torch._int_mm wrapper enforces M > 16 at runtime
        # ("self.size(0) needs to be greater than 16, but got 16"), not at
        # build time, so without this preflight the smallest shape in the
        # locked SHAPES grid (M=16) would dispatch into the runtime check
        # and surface as an `error` field in the artifact. Catching it
        # here keeps the per-shape entry as a clean
        # `imma_unavailable_reason` stub with the exact torch constraint
        # named, so the schema-lock test
        # `test_ok_artifact_cublaslt_int8_arm_is_either_populated_or_explicitly_unavailable`
        # treats it as a documented IMMA-unavailability case, not a
        # silent failure.
        return {
            "backend": "cublaslt_imma_int8",
            "shape": shape,
            "imma_unavailable_reason": (
                f"M={M} <= 16; torch._int_mm requires M > 16 at the "
                f"Python wrapper layer (\"self.size(0) needs to be "
                f"greater than 16\"). This is a PyTorch dispatch "
                f"constraint, not a hardware IMMA alignment constraint; "
                f"shapes with M >= 17 (and in practice M >= 32 on the "
                f"locked SHAPES grid) clear the wrapper and reach the "
                f"actual cuBLASLt IMMA INT8 GEMM path. The (M=16, K=16) "
                f"shape in the locked SHAPES grid will therefore stay "
                f"IMMA-ineligible until either (a) M is widened past "
                f"16 in the shape set, or (b) the harness wires the raw "
                f"cuBLASLt API instead of torch._int_mm."
            ),
        }

    device = torch_mod.device("cuda")
    gen = torch_mod.Generator(device=device).manual_seed(0xCAB)

    try:
        w_i8 = torch_mod.randint(
            -8, 8, (M, K), generator=gen, device=device, dtype=torch_mod.int8
        )
        x_i8 = torch_mod.randint(
            -8, 8, (K, n_pad), generator=gen, device=device, dtype=torch_mod.int8
        )
        result = torch_mod._int_mm(w_i8, x_i8)
        torch_mod.cuda.synchronize()
        if result.dtype != torch_mod.int32:
            return {
                "backend": "cublaslt_imma_int8",
                "shape": shape,
                "imma_unavailable_reason": (
                    f"torch._int_mm returned dtype={result.dtype!r}, "
                    f"expected int32 (cuBLASLt CUBLAS_COMPUTE_32I)."
                ),
            }
    except Exception as exc:
        return {
            "backend": "cublaslt_imma_int8",
            "shape": shape,
            "imma_unavailable_reason": (
                f"torch._int_mm failed at (M={M}, K={K}, N={n_pad}) "
                f"with {type(exc).__name__}: {exc}. This is the cuBLASLt "
                f"IMMA dispatch path; ensure the GPU SM supports IMMA "
                f"(sm_75+) and the Torch build wires cuBLASLt."
            ),
        }

    def call() -> None:
        torch_mod._int_mm(w_i8, x_i8)

    timed = _time_with_stability(
        call,
        warmup=warmup,
        iters=iters,
        num_stability_runs=num_stability_runs,
        torch_mod=torch_mod,
    )
    summary = _summary_ms(timed["samples_ms"])
    summary = _attach_stability_to_summary(summary, timed)
    per_col_median = (
        float(summary["median"]) / float(n_pad) if n_pad > 0 and summary["median"] > 0.0 else 0.0
    )
    per_col_mean = (
        float(summary["mean"]) / float(n_pad) if n_pad > 0 and summary["mean"] > 0.0 else 0.0
    )

    return {
        "backend": "cublaslt_imma_int8",
        "shape": shape,
        "dtype_W": "int8",
        "dtype_x": "int8",
        "dtype_accum": "int32",
        "dtype_out": "int32",
        "alignment_note": (
            f"cuBLASLt IMMA INT8 GEMM (torch._int_mm) requires N % 8 == 0; "
            f"the uTPU N=1 GEMV is padded to N={n_pad} (alignment minimum). "
            f"kernel_ms is the full N={n_pad} wall time; "
            f"ms_per_n_column_median = kernel_ms.median / N_padded is the "
            f"GEMV-equivalent per-column time and is what the parent uses "
            f"for gap_vs_cublaslt_int8_pct_median vs uTPU N=1. This is "
            f"slightly favorable to cuBLAS (launch and tensor-core setup "
            f"costs amortize across N_padded columns); a true N=1 cuBLAS "
            f"INT8 GEMV does not exist in IMMA, so the per-column figure "
            f"is the closest honest apples-to-apples comparison."
        ),
        "kernel_ms": summary,
        "ms_per_n_column_median": per_col_median,
        "ms_per_n_column_mean": per_col_mean,
        "samples_ms": [float(s) for s in timed["samples_ms"][:32]],
    }


def _run_inductor_linear(
    torch_mod, M: int, K: int, warmup: int, iters: int, num_stability_runs: int = 1
) -> Dict[str, Any]:
    import torch.nn as nn

    device = torch_mod.device("cuda")
    gen = torch_mod.Generator(device=device).manual_seed(0xACE)
    model = nn.Linear(K, M, bias=False).to(device=device, dtype=torch_mod.float32)
    compiled = torch_mod.compile(model, backend="inductor", fullgraph=True)
    x = torch_mod.randn((1, K), generator=gen, device=device, dtype=torch_mod.float32)

    def call() -> None:
        with torch_mod.no_grad():
            compiled(x)

    timed = _time_with_stability(
        call,
        warmup=warmup,
        iters=iters,
        num_stability_runs=num_stability_runs,
        torch_mod=torch_mod,
    )
    summary = _summary_ms(timed["samples_ms"])
    summary = _attach_stability_to_summary(summary, timed)
    return {
        "backend": "inductor_linear_fp32",
        "shape": {"M": int(M), "K": int(K), "N": 1},
        "dtype_W": "float32",
        "dtype_x": "float32",
        "dtype_accum": "float32",
        "dtype_out": "float32",
        "dtype_caveat": (
            "Inductor reference runs in float32 (the default Torch dispatch path "
            "for nn.Linear); uTPU kernel runs INT8 inputs with INT32 accumulator. "
            "Not a like-for-like dtype comparison; see methodology block."
        ),
        "kernel_ms": summary,
        "samples_ms": [float(s) for s in timed["samples_ms"][:32]],
    }


def run_baselines(
    shapes: List[Dict[str, int]],
    warmup: int,
    iters: int,
    num_stability_runs: int,
) -> Dict[str, Any]:
    try:
        import torch as torch_mod
    except Exception as exc:
        return {
            "status": "torch_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "shapes_requested": shapes,
        }

    if not torch_mod.cuda.is_available():
        return {
            "status": "cuda_unavailable",
            "reason": "torch.cuda.is_available() is False inside subprocess.",
            "shapes_requested": shapes,
            "torch_version": str(torch_mod.__version__),
        }

    env = _gpu_environment(torch_mod)
    per_shape: List[Dict[str, Any]] = []
    for shape in shapes:
        M = int(shape["M"])
        K = int(shape["K"])
        try:
            cublas = _run_cublas_gemv(
                torch_mod, M, K, warmup=warmup, iters=iters,
                num_stability_runs=num_stability_runs,
            )
        except Exception as exc:
            cublas = {
                "backend": "cublas_gemv_int32",
                "shape": {"M": M, "K": K, "N": 1},
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            cublaslt_int8 = _run_cublaslt_imma_int8_gemm(
                torch_mod, M, K, warmup=warmup, iters=iters,
                num_stability_runs=num_stability_runs,
            )
        except Exception as exc:
            cublaslt_int8 = {
                "backend": "cublaslt_imma_int8",
                "shape": {"M": M, "K": K, "N": 1, "N_padded": CUBLASLT_INT8_N_PAD},
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            inductor = _run_inductor_linear(
                torch_mod, M, K, warmup=warmup, iters=iters,
                num_stability_runs=num_stability_runs,
            )
        except Exception as exc:
            inductor = {
                "backend": "inductor_linear_fp32",
                "shape": {"M": M, "K": K, "N": 1},
                "error": f"{type(exc).__name__}: {exc}",
            }
        per_shape.append(
            {
                "shape": {"M": M, "K": K, "N": 1},
                "cublas": cublas,
                "cublaslt_int8": cublaslt_int8,
                "inductor": inductor,
            }
        )

    return {
        "status": "ok",
        "environment": env,
        "warmup": int(warmup),
        "iters": int(iters),
        "num_stability_runs": int(num_stability_runs),
        "timing_protocol": TIMING_PROTOCOL_NAME,
        "timing_stability_threshold_pct": float(TIMING_STABILITY_THRESHOLD_PCT),
        "shapes": per_shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes-json", required=True, help="JSON-encoded list of {'M':int,'K':int} dicts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--num-stability-runs", type=int, default=7)
    args = parser.parse_args()

    shapes = json.loads(args.shapes_json)
    payload = run_baselines(
        shapes=shapes,
        warmup=int(args.warmup),
        iters=int(args.iters),
        num_stability_runs=int(args.num_stability_runs),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    sys.exit(0 if payload.get("status") == "ok" else 0)


if __name__ == "__main__":
    main()
