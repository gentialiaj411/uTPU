"""GPU subprocess for Phase 7 remediation P2.2 selection A/B benchmark.

For each shape (M, K) and each schedule (cost-model A / oracle B), times
the NVRTC blocked-FC kernel with the EXPLICIT `schedule_params` dict
passed by the parent harness. This is the same execution path
`CompiledMLPRuntime._execute_compiled_resident` takes via
`CUDABlockedFCExecutor.execute` once
`schedule_source="cost_model"` is plumbed (see
`firmware/host/compiled_runtime.py::_schedule_params_for_op`).

Methodology (locked, mirrors `_cublas_baseline_torch_subprocess.py`):
- warmup=10 + iters=50 (parent-controlled, defaults below)
- `torch.cuda.synchronize` brackets around each timed call
- median-of-N + mean/stdev/min/max/p95
- deterministic per-shape RNG seed so weight/activation distributions
  are reproducible across re-runs
- bit-exactness check between A's output and a NumPy oracle (both A and
  B must match within `atol=0` because INT4 weights + INT32 accum)

Emits a single JSON line to stdout: `{"status": "ok", "results": [...]}`
or `{"status": "cuda_unavailable", "reason": "..."}` on failure to
import or initialise CUDA.

The parent (`run_selection_ab.py`) consumes the JSON, computes realized
regret per shape, and stitches with the predicted regret from
`bench/results/cost_model_selection.json`.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Any, Dict, List


def _import_cuda_executor():
    """Import the NVRTC executor + force a one-time CUDA context init.

    Returns the executor instance or raises with a string reason."""
    from cuda_blocked_fc_backend import (
        CUDABlockedFCExecutor,
        detect_cuda_environment,
    )

    env = detect_cuda_environment()
    if not env.runtime_available:
        raise RuntimeError(f"cuda_unavailable: {env.reason}")
    return CUDABlockedFCExecutor(verbose=False)


def _make_request(
    out_features: int,
    in_features: int,
    array_size: int,
    seed: int,
):
    import numpy as np
    from lowering_types import BlockedFCLoweringRequest

    rng = np.random.default_rng(seed)
    weights = rng.integers(-8, 8, size=(out_features, in_features), dtype=np.int8)
    activations = rng.integers(-8, 8, size=(in_features,), dtype=np.int8)
    return BlockedFCLoweringRequest(
        weights_int4=weights,
        activations_int4=activations,
        out_features=int(out_features),
        in_features=int(in_features),
        array_size=int(array_size),
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )


def _numpy_oracle_int4(req) -> "Any":
    import numpy as np

    accum = req.weights_int4.astype(np.int32) @ req.activations_int4.astype(np.int32).reshape(-1)
    quantised = np.clip(accum, -8, 7).astype(np.int8)
    return quantised


def _summary(ms_samples: List[float]) -> Dict[str, float]:
    return {
        "mean": float(statistics.fmean(ms_samples)),
        "median": float(statistics.median(ms_samples)),
        "stdev": float(statistics.pstdev(ms_samples)) if len(ms_samples) > 1 else 0.0,
        "min": float(min(ms_samples)),
        "max": float(max(ms_samples)),
        "p95": float(sorted(ms_samples)[max(0, int(round(0.95 * len(ms_samples))) - 1)]),
        "samples": int(len(ms_samples)),
    }


def _time_interleaved_ab(
    executor,
    out_features: int,
    in_features: int,
    array_size: int,
    schedule_a: Dict[str, int],
    schedule_b: Dict[str, int],
    seed: int,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    """Time two schedules for the same (shape, weights, activations) by
    **interleaving** them per iteration: warmup(A) -> warmup(B) ->
    [A, B] x iters. Interleaving keeps GPU clocking / thermal state
    symmetric across the two arms, which sub-millisecond GEMV
    measurements are very sensitive to. The earlier "all A then all
    B" pattern produced realized regrets dominated by inter-arm GPU
    clock state drift; interleaving cuts that out without changing
    the methodology beyond what the existing dtype caveats already
    cover.
    """
    import torch

    req = _make_request(out_features, in_features, array_size, seed)
    oracle = _numpy_oracle_int4(req)

    for _ in range(warmup):
        _ = executor.execute(req, schedule_params=schedule_a)
        torch.cuda.synchronize()
        _ = executor.execute(req, schedule_params=schedule_b)
        torch.cuda.synchronize()

    a_samples: List[float] = []
    b_samples: List[float] = []
    last_a_output = None
    last_b_output = None
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a_result = executor.execute(req, schedule_params=schedule_a)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        b_result = executor.execute(req, schedule_params=schedule_b)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        a_samples.append((t1 - t0) * 1000.0)
        b_samples.append((t2 - t1) * 1000.0)
        last_a_output = a_result.get("output_unpadded")
        last_b_output = b_result.get("output_unpadded")

    a_bit_exact = bool(
        last_a_output is not None
        and (oracle == last_a_output[: oracle.shape[0]]).all()
    )
    b_bit_exact = bool(
        last_b_output is not None
        and (oracle == last_b_output[: oracle.shape[0]]).all()
    )

    return {
        "a": {
            "schedule": dict(schedule_a),
            "kernel_ms": _summary(a_samples),
            "bit_exact_vs_numpy_oracle": a_bit_exact,
        },
        "b": {
            "schedule": dict(schedule_b),
            "kernel_ms": _summary(b_samples),
            "bit_exact_vs_numpy_oracle": b_bit_exact,
        },
    }


def _run(plan: Dict[str, Any]) -> Dict[str, Any]:
    try:
        executor = _import_cuda_executor()
    except Exception as exc:
        return {"status": "cuda_unavailable", "reason": str(exc)}

    warmup = int(plan.get("warmup", 10))
    iters = int(plan.get("iters", 50))
    array_size = int(plan.get("array_size", 16))

    results: List[Dict[str, Any]] = []
    for entry in plan["shapes"]:
        out_features = int(entry["out_features"])
        in_features = int(entry["in_features"])
        seed = int(entry.get("seed", 0xABCDEF))

        cost_model_schedule = dict(entry["cost_model_schedule"])
        oracle_schedule = dict(entry["oracle_schedule"])

        timed = _time_interleaved_ab(
            executor, out_features, in_features, array_size,
            cost_model_schedule, oracle_schedule, seed, warmup, iters,
        )
        a = timed["a"]
        b = timed["b"]

        cost_med = a["kernel_ms"]["median"]
        oracle_med = b["kernel_ms"]["median"]
        if oracle_med > 0.0:
            realized_regret_pct = (cost_med - oracle_med) / oracle_med * 100.0
        else:
            realized_regret_pct = None

        results.append({
            "shape": {
                "out_features": out_features,
                "in_features": in_features,
                "array_size": array_size,
            },
            "seed": seed,
            "cost_model_run": a,
            "oracle_run": b,
            "realized_regret_pct": realized_regret_pct,
            "schedules_identical": cost_model_schedule == oracle_schedule,
        })

    return {
        "status": "ok",
        "warmup": warmup,
        "iters": iters,
        "array_size": array_size,
        "results": results,
    }


def main():
    plan_text = sys.stdin.read()
    plan = json.loads(plan_text)
    out = _run(plan)
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
