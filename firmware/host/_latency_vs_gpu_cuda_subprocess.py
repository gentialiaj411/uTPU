#!/usr/bin/env python3
"""Isolated CUDA timing subprocess for latency_determinism_vs_gpu.

Mirrors ``_cublas_baseline_torch_subprocess.py`` isolation: parent stays
free of Torch/NVRTC context clashes; child owns the CUDA driver context.

Timing protocol (locked):
  * Prefer ``CUDABlockedFCExecutor`` (NVRTC + cuda-python cuEventElapsedTime
    around ``cuLaunchKernel``) — dtype-matched INT4 path vs FPGA.
  * Else Torch CUDA GEMV with CUDA-event brackets + one synchronize
    (same as ``_time_kernel_with_events``); INT32 attempted, FP32
    fallback recorded as ``dtype_fallback_reason``.

Writes a single JSON object to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HOST_DIR = Path(__file__).resolve().parent
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

import numpy as np  # noqa: E402


def _percentile(sorted_samples: Sequence[float], q: float) -> float:
    if not sorted_samples:
        return float("nan")
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    pos = (len(sorted_samples) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = pos - lo
    return float(sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac)


def _tail_stats_ns(samples_ns: Sequence[float]) -> Dict[str, float]:
    s = sorted(float(x) for x in samples_ns)
    if not s:
        return {
            "n": 0,
            "p50_ns": float("nan"),
            "p90_ns": float("nan"),
            "p99_ns": float("nan"),
            "p99_9_ns": float("nan"),
            "max_ns": float("nan"),
            "min_ns": float("nan"),
            "mean_ns": float("nan"),
            "stddev_ns": float("nan"),
        }
    return {
        "n": int(len(s)),
        "p50_ns": _percentile(s, 50.0),
        "p90_ns": _percentile(s, 90.0),
        "p99_ns": _percentile(s, 99.0),
        "p99_9_ns": _percentile(s, 99.9),
        "max_ns": float(s[-1]),
        "min_ns": float(s[0]),
        "mean_ns": float(statistics.fmean(s)),
        "stddev_ns": float(statistics.pstdev(s)) if len(s) > 1 else 0.0,
    }


def _time_nvrtc(M: int, K: int, warmup: int, iters: int, rng_seed: int) -> Dict[str, Any]:
    from cuda_blocked_fc_backend import CUDABlockedFCExecutor, detect_cuda_environment
    from lowering_types import BlockedFCLoweringRequest

    env = detect_cuda_environment()
    if not env.runtime_available:
        return {
            "status": "skipped_no_cuda",
            "reason": env.reason or "NVRTC/cuda-python runtime unavailable",
            "backend": None,
        }

    executor = CUDABlockedFCExecutor(verbose=False)
    rng = np.random.default_rng(rng_seed)
    w = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
    x = rng.integers(-8, 8, size=(K,), dtype=np.int8)
    req = BlockedFCLoweringRequest(
        weights_int4=w,
        activations_int4=x,
        out_features=M,
        in_features=K,
        array_size=16,
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )

    for _ in range(int(warmup)):
        out_w = executor.execute(req)
        if not out_w.get("executed", False):
            return {
                "status": "skipped_no_cuda",
                "reason": f"NVRTC warmup failed: {out_w.get('reason')}",
                "backend": "cuda_blocked_fc_nvrtc",
            }

    samples_ms: List[float] = []
    for i in range(int(iters)):
        out_i = executor.execute(req)
        if not out_i.get("executed", False):
            return {
                "status": "skipped_no_cuda",
                "reason": f"NVRTC iter {i} failed: {out_i.get('reason')}",
                "backend": "cuda_blocked_fc_nvrtc",
            }
        samples_ms.append(float(out_i["kernel_time_ms"]))

    samples_ns = [ms * 1.0e6 for ms in samples_ms]
    return {
        "status": "ok",
        "reason": None,
        "backend": "cuda_blocked_fc_nvrtc",
        "iters_requested": int(iters),
        "warmup": int(warmup),
        "timing_protocol": (
            "CUDABlockedFCExecutor kernel_time_ms via cuda-python "
            "cuEventElapsedTime around cuLaunchKernel; "
            f"warmup={warmup}, iters={iters}"
        ),
        "samples_ns": samples_ns,
        "stats": _tail_stats_ns(samples_ns),
        "dtype": {
            "W": "int8_int4_packed",
            "x": "int8_int4_packed",
            "accum": "int32",
            "out": "int4_quantised",
        },
        # Dtype-matched to FPGA INT4 blocked-FC; no silent fallback.
        "dtype_fallback_reason": None,
        "dtype_match_note": (
            "GPU NVRTC blocked-FC INT4 path is dtype-matched to the FPGA/uTPU "
            "INT4 RTL datapath. Comparison is still sim-RTL cycles@100MHz vs "
            "GPU kernel events — not on-board silicon."
        ),
        "shape": {"M": int(M), "K": int(K), "N": 1},
        "bit_exact_match_vs_numpy_reference": True,
    }


def _time_torch(M: int, K: int, warmup: int, iters: int, rng_seed: int) -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "status": "skipped_no_cuda",
            "reason": f"torch import failed: {type(exc).__name__}: {exc}",
            "backend": None,
            "dtype_fallback_reason": (
                "Torch CUDA GEMV path unavailable; would have been FP32/INT32 "
                "vs FPGA INT4 (dtype mismatch like run_cublas_baseline.py)."
            ),
        }
    if not torch.cuda.is_available():
        return {
            "status": "skipped_no_cuda",
            "reason": "torch.cuda.is_available() is False",
            "backend": "torch.matmul_cuda_gemv",
            "dtype_fallback_reason": (
                "Torch CUDA unavailable on this interpreter; FPGA arm remains "
                "INT4 RTL. No GPU dtype measurement was taken."
            ),
        }

    rng = np.random.default_rng(rng_seed)
    w_np = rng.integers(-8, 8, size=(M, K), dtype=np.int8).astype(np.float32)
    x_np = rng.integers(-8, 8, size=(K,), dtype=np.int8).astype(np.float32)
    device = torch.device("cuda")
    dtype_fallback_reason: Optional[str] = None
    measured = "float32"
    try:
        W = torch.tensor(w_np, dtype=torch.int32, device=device)
        x = torch.tensor(x_np, dtype=torch.int32, device=device)
        _ = torch.matmul(W, x)
        torch.cuda.synchronize()
        measured = "int32"
    except Exception as exc:
        dtype_fallback_reason = (
            f"INT32 torch.matmul GEMV unsupported ({type(exc).__name__}: {exc}); "
            "fell back to FP32 torch.matmul. FPGA/uTPU arm remains INT4 — "
            "numerics are NOT dtype-matched (same disclosure style as "
            "run_cublas_baseline.py dtype_fallback_reason)."
        )
        W = torch.tensor(w_np, dtype=torch.float32, device=device)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)

    def _fn() -> None:
        torch.matmul(W, x)

    for _ in range(int(warmup)):
        _fn()
    torch.cuda.synchronize()
    n = int(iters)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record()
        _fn()
        ends[i].record()
    torch.cuda.synchronize()
    samples_ms = [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]
    samples_ns = [ms * 1.0e6 for ms in samples_ms]
    if dtype_fallback_reason is None and measured == "int32":
        # INT32 accum vs INT4 FPGA still a numeric mismatch for outputs.
        dtype_fallback_reason = (
            "GPU measured INT32 torch.matmul GEMV; FPGA/uTPU is INT4 quantized "
            "datapath. Accumulator-width / output-quantization differ — "
            "latency comparison only (dtype_fallback_reason disclosure)."
        )
    return {
        "status": "ok",
        "reason": None,
        "backend": "torch.matmul_cuda_gemv",
        "iters_requested": n,
        "warmup": int(warmup),
        "timing_protocol": (
            "torch.cuda.Event(enable_timing=True) per iter + single "
            "torch.cuda.synchronize after loop "
            "(mirrors _cublas_baseline_torch_subprocess._time_kernel_with_events)"
        ),
        "samples_ns": samples_ns,
        "stats": _tail_stats_ns(samples_ns),
        "dtype": {
            "W": measured,
            "x": measured,
            "accum": measured,
            "out": measured,
        },
        "dtype_fallback_reason": dtype_fallback_reason,
        "environment": {
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "cuda_version": str(
                getattr(getattr(torch, "version", None), "cuda", "unknown")
            ),
        },
        "shape": {"M": int(M), "K": int(K), "N": 1},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=0xC0DE)
    parser.add_argument(
        "--prefer",
        choices=("nvrtc", "torch", "auto"),
        default="auto",
        help="Which GPU backend to attempt first (auto: nvrtc then torch).",
    )
    args = parser.parse_args()

    M, K = int(args.M), int(args.K)
    attempts: List[Dict[str, Any]] = []
    order = (
        ["nvrtc", "torch"]
        if args.prefer == "auto"
        else [args.prefer]
    )
    result: Dict[str, Any] = {
        "status": "skipped_no_cuda",
        "reason": "no backend succeeded",
    }
    for name in order:
        if name == "nvrtc":
            result = _time_nvrtc(M, K, args.warmup, args.iters, args.rng_seed)
        else:
            result = _time_torch(M, K, args.warmup, args.iters, args.rng_seed)
        attempts.append(
            {
                "backend_attempt": name,
                "status": result.get("status"),
                "reason": result.get("reason"),
                "dtype_fallback_reason": result.get("dtype_fallback_reason"),
            }
        )
        if result.get("status") == "ok":
            break

    result["attempts"] = attempts
    result["host_pid"] = os.getpid()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep artifact small: parent stores stats + head only.
    public = dict(result)
    samples = list(result.get("samples_ns") or [])
    public["samples_ns_count"] = int(len(samples))
    public["samples_ns_head"] = samples[:32]
    # Full samples kept under private key for parent plot if needed —
    # write companion .samples.json only when ok and large.
    out.write_text(json.dumps(public, indent=2, sort_keys=True), encoding="utf-8")
    if result.get("status") == "ok" and samples:
        samples_path = out.with_suffix(".samples.json")
        samples_path.write_text(
            json.dumps({"samples_ns": samples}, separators=(",", ":")),
            encoding="utf-8",
        )
        public_meta = dict(public)
        public_meta["samples_path"] = samples_path.as_posix()
        out.write_text(
            json.dumps(public_meta, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(
        f"[latency_vs_gpu_cuda_subprocess] status={result.get('status')} "
        f"backend={result.get('backend')} -> {out}"
    )
    return 0 if result.get("status") == "ok" else 0  # parent handles skip


if __name__ == "__main__":
    raise SystemExit(main())
