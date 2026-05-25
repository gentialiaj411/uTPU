"""Phase 7 — Torch baseline subprocess (cuBLAS + TorchInductor).

Runs in a separate process from the parent ``run_cublas_baseline.py``
script so the parent can keep its NVRTC driver context for the uTPU
blocked-FC kernel without crashing into Torch's own CUDA / Inductor
contexts (this is the same isolation pattern used by
``inductor_oracle_subprocess.py``).

For each shape in the requested ``shapes`` list this subprocess emits:

* **cuBLAS GEMV** timing: ``torch.matmul(W, x)`` with INT32 weights /
  INT32 activations / INT32 output, dispatched through cuBLAS by Torch.
  This is the apples-to-apples comparison for our uTPU kernel
  (INT8 inputs / INT32 accumulation / INT4 quantised output): same
  accumulator dtype, same op semantics for GEMV. INT4 quantisation
  happens off-cuBLAS so it isn't counted in the baseline kernel time;
  the parent script documents this caveat in the artifact.
* **TorchInductor** timing: a thin ``torch.nn.Linear(K, M, bias=False)``
  module compiled with ``torch.compile(..., backend="inductor",
  fullgraph=True)`` and run on the same input. dtype is float32 (the
  default Torch dispatcher); the parent script records the
  dtype-mismatch caveat in the artifact so the comparison is not
  silently inflated.

Methodology is locked in this single file so the parent can describe it
verbatim in the artifact:

* Warmup ``warmup`` invocations (default 10), discarded.
* Then ``iters`` measured invocations (default 50), each bracketed by
  ``torch.cuda.synchronize()`` so we only count kernel wall time.
* Stats reported per shape: mean, median, stdev, min, max, p95 (ms).
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
from typing import Any, Dict, List


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


def _time_kernel(fn, warmup: int, iters: int, sync) -> List[float]:
    for _ in range(warmup):
        fn()
    sync()
    samples: List[float] = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return samples


def _run_cublas_gemv(torch_mod, M: int, K: int, warmup: int, iters: int) -> Dict[str, Any]:
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

    samples = _time_kernel(
        call,
        warmup=warmup,
        iters=iters,
        sync=lambda: torch_mod.cuda.synchronize(),
    )
    entry = {
        "backend": "cublas_gemv_int32" if dtype_fallback_reason is None else "cublas_gemv_fp32_fallback",
        "shape": {"M": int(M), "K": int(K), "N": 1},
        "dtype_W": dtype_W,
        "dtype_x": dtype_x,
        "dtype_accum": dtype_accum,
        "dtype_out": dtype_out,
        "kernel_ms": _summary_ms(samples),
        "samples_ms": [float(s) for s in samples[:32]],
    }
    if dtype_fallback_reason is not None:
        entry["dtype_fallback_reason"] = dtype_fallback_reason
    return entry


def _run_inductor_linear(torch_mod, M: int, K: int, warmup: int, iters: int) -> Dict[str, Any]:
    import torch.nn as nn

    device = torch_mod.device("cuda")
    gen = torch_mod.Generator(device=device).manual_seed(0xACE)
    model = nn.Linear(K, M, bias=False).to(device=device, dtype=torch_mod.float32)
    compiled = torch_mod.compile(model, backend="inductor", fullgraph=True)
    x = torch_mod.randn((1, K), generator=gen, device=device, dtype=torch_mod.float32)

    def call() -> None:
        with torch_mod.no_grad():
            compiled(x)

    samples = _time_kernel(
        call,
        warmup=warmup,
        iters=iters,
        sync=lambda: torch_mod.cuda.synchronize(),
    )
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
        "kernel_ms": _summary_ms(samples),
        "samples_ms": [float(s) for s in samples[:32]],
    }


def run_baselines(shapes: List[Dict[str, int]], warmup: int, iters: int) -> Dict[str, Any]:
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
            cublas = _run_cublas_gemv(torch_mod, M, K, warmup=warmup, iters=iters)
        except Exception as exc:
            cublas = {
                "backend": "cublas_gemv_int32",
                "shape": {"M": M, "K": K, "N": 1},
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            inductor = _run_inductor_linear(torch_mod, M, K, warmup=warmup, iters=iters)
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
                "inductor": inductor,
            }
        )

    return {
        "status": "ok",
        "environment": env,
        "warmup": int(warmup),
        "iters": int(iters),
        "shapes": per_shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes-json", required=True, help="JSON-encoded list of {'M':int,'K':int} dicts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    shapes = json.loads(args.shapes_json)
    payload = run_baselines(
        shapes=shapes,
        warmup=int(args.warmup),
        iters=int(args.iters),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    sys.exit(0 if payload.get("status") == "ok" else 0)


if __name__ == "__main__":
    main()
