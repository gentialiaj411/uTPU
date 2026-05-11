import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from lowering_types import BlockedFCLoweringRequest
from cuda_blocked_fc_backend import CUDABlockedFCExecutor, detect_cuda_environment


def _int8_matmul_tops(m: int, n: int, k: int, ms: float) -> float:
    if ms <= 0:
        return 0.0
    ops = 2.0 * m * n * k
    return ops / (ms / 1000.0) / 1e12


def _run_executor_once(exec_engine: CUDABlockedFCExecutor, req: BlockedFCLoweringRequest) -> Dict[str, Any]:
    out = exec_engine.execute(req)
    if not out.get("executed", False):
        raise RuntimeError(out.get("reason", "CUDA execution failed"))
    return out


def _maybe_cublas_baseline(m: int, n: int, k: int, iters: int) -> Dict[str, Any]:
    """
    Optional baseline via CuPy/cuBLAS if cupy is installed.
    Returns unavailable status otherwise.
    """
    try:
        import cupy as cp
    except Exception as e:
        return {
            "available": False,
            "reason": f"cupy unavailable: {e}",
        }

    try:
        a = cp.random.randint(-8, 8, size=(m, k), dtype=cp.int8)
        b = cp.random.randint(-8, 8, size=(k, n), dtype=cp.int8)
        # int32 accumulation path
        # Use matmul with promoted int32 inputs to guarantee accumulator type.
        a32 = a.astype(cp.int32)
        b32 = b.astype(cp.int32)
        cp.matmul(a32, b32)
        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            cp.matmul(a32, b32)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) * 1000.0 / iters
        return {
            "available": True,
            "avg_ms": avg_ms,
            "tops": _int8_matmul_tops(m, n, k, avg_ms),
            "note": "CuPy matmul int32 path (cuBLAS-backed).",
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"cupy present but cuBLAS path unavailable: {e}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CUDA blocked-FC kernel.")
    parser.add_argument("--m", type=int, default=10, help="Output features (rows).")
    parser.add_argument("--k", type=int, default=9, help="Input features.")
    parser.add_argument("--array-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-json", type=str, default="build/reports/cuda_blocked_fc_benchmark.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    env = detect_cuda_environment()
    if not env.runtime_available:
        out = {
            "status": "blocked",
            "reason": env.reason,
            "measurement_steps": [
                "Ensure NVIDIA driver is installed and GPU is visible via nvidia-smi.",
                "Ensure CUDA NVRTC DLL directory is on PATH.",
                "Install optional CuPy for cuBLAS baseline.",
            ],
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 0

    m = int(args.m)
    k = int(args.k)
    n = 1
    rng = np.random.default_rng(42)
    w = rng.integers(-8, 8, size=(m, k), dtype=np.int8)
    x = rng.integers(-8, 8, size=(k,), dtype=np.int8)
    req = BlockedFCLoweringRequest(
        weights_int4=w,
        activations_int4=x,
        out_features=m,
        in_features=k,
        array_size=int(args.array_size),
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )

    engine = CUDABlockedFCExecutor(verbose=args.verbose)
    for _ in range(args.warmup):
        _run_executor_once(engine, req)

    kernel_ms = []
    h2d_ms = []
    d2h_ms = []
    e2e_ms = []
    for _ in range(args.iters):
        r = _run_executor_once(engine, req)
        kernel_ms.append(float(r["kernel_time_ms"]))
        h2d_ms.append(float(r["h2d_time_ms"]))
        d2h_ms.append(float(r["d2h_time_ms"]))
        e2e_ms.append(float(r["end_to_end_time_ms"]))

    kernel_avg = float(np.mean(kernel_ms))
    h2d_avg = float(np.mean(h2d_ms))
    d2h_avg = float(np.mean(d2h_ms))
    e2e_avg = float(np.mean(e2e_ms))
    transfer_avg = h2d_avg + d2h_avg
    transfer_pct = float((transfer_avg / e2e_avg) * 100.0) if e2e_avg > 0 else 0.0

    cublas = _maybe_cublas_baseline(m, n, k, args.iters)
    cublas_pct = None
    if cublas.get("available", False):
        cublas_ms = float(cublas["avg_ms"])
        cublas_pct = float((cublas_ms / kernel_avg) * 100.0) if kernel_avg > 0 else None

    result = {
        "status": "ok",
        "shape": {"m": m, "n": n, "k": k},
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "timing_ms": {
            "kernel_avg": kernel_avg,
            "h2d_avg": h2d_avg,
            "d2h_avg": d2h_avg,
            "transfer_avg": transfer_avg,
            "end_to_end_avg": e2e_avg,
        },
        "transfer_overhead_pct_of_e2e": transfer_pct,
        "kernel_tops": _int8_matmul_tops(m, n, k, kernel_avg),
        "cublas_baseline": cublas,
        "kernel_speed_vs_cublas_pct": cublas_pct,
        "notes": [
            "kernel_speed_vs_cublas_pct is computed only when cuBLAS baseline is available.",
            "Current backend computes one output vector (n=1) for FC inference shape.",
        ],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
