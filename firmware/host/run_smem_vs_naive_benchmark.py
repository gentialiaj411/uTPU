"""Benchmark blocked_fc naive vs shared-memory input staging on the same shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HOST_DIR = Path(__file__).resolve().parent
REPO_ROOT = HOST_DIR.parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "bench" / "results" / "smem_vs_naive.json"

if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from cuda_blocked_fc_backend import CUDABlockedFCExecutor, detect_cuda_environment  # noqa: E402
from lowering_types import BlockedFCLoweringRequest  # noqa: E402

# Shapes from cost-model / autotuner calibration grid (small enough for SMEM).
DEFAULT_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]


def _make_request(out_features: int, in_features: int, seed: int) -> BlockedFCLoweringRequest:
    rng = np.random.default_rng(seed)
    w = rng.integers(-8, 8, size=(out_features, in_features), dtype=np.int8)
    x = rng.integers(-8, 8, size=(in_features,), dtype=np.int8)
    return BlockedFCLoweringRequest(
        weights_int4=w,
        activations_int4=x,
        out_features=out_features,
        in_features=in_features,
        array_size=16,
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )


def _time_kernel(
    executor: CUDABlockedFCExecutor,
    request: BlockedFCLoweringRequest,
    use_smem: bool,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    params = {"threads_per_block": 128, "unroll_factor": 1, "use_smem": use_smem}
    for _ in range(warmup):
        warm = executor.execute(request, schedule_params=params)
        if not warm.get("executed"):
            return {"executed": False, "reason": warm.get("reason")}
    samples: List[float] = []
    last: Dict[str, Any] = {}
    for _ in range(iters):
        result = executor.execute(request, schedule_params=params)
        if not result.get("executed"):
            return {"executed": False, "reason": result.get("reason")}
        samples.append(float(result["kernel_time_ms"]))
        last = result
    return {
        "executed": True,
        "kernel_median_ms": float(statistics.median(samples)),
        "kernel_mean_ms": float(statistics.fmean(samples)),
        "max_abs_diff_vs_numpy_reference": int(last.get("max_abs_diff_vs_numpy_reference", -1)),
        "kernel_name": last.get("kernel_name"),
        "smem_fallback": bool(last.get("smem_fallback", False)),
        "shared_mem_bytes": int(last.get("shared_mem_bytes", 0)),
    }


def run_benchmark(
    shapes: List[tuple[int, int]],
    warmup: int = 5,
    iters: int = 30,
    seed: int = 0,
) -> Dict[str, Any]:
    env = detect_cuda_environment()
    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warmup": int(warmup),
        "iters": int(iters),
        "shapes": [],
    }
    if not env.runtime_available:
        payload["status"] = "cuda_unavailable"
        payload["reason"] = env.reason
        return payload

    executor = CUDABlockedFCExecutor(verbose=False)
    payload["status"] = "ok"
    for idx, (m, k) in enumerate(shapes):
        req = _make_request(m, k, seed=int(seed) + idx)
        naive = _time_kernel(executor, req, use_smem=False, warmup=warmup, iters=iters)
        smem = _time_kernel(executor, req, use_smem=True, warmup=warmup, iters=iters)
        row: Dict[str, Any] = {
            "out_features": int(m),
            "in_features": int(k),
            "naive": naive,
            "smem": smem,
        }
        if naive.get("executed") and smem.get("executed"):
            n_med = float(naive["kernel_median_ms"])
            s_med = float(smem["kernel_median_ms"])
            row["speedup_vs_naive"] = (n_med / s_med) if s_med > 0 else None
        payload["shapes"].append(row)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = run_benchmark(DEFAULT_SHAPES, warmup=args.warmup, iters=args.iters, seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[run_smem_vs_naive_benchmark] wrote {out_path} status={payload.get('status')}")


if __name__ == "__main__":
    main()
