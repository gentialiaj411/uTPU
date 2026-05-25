"""Phase 7 remediation P1.2 — CUDA fusion benchmark subprocess.

Times the three fusion-payoff workloads on a real CUDA device in two
modes:

* **eager**: PyTorch eager-mode execution (the "unfused" reference;
  each op dispatches as a separate kernel launch + intermediate tensor).
* **inductor**: ``torch.compile(model, backend="inductor",
  fullgraph=True)`` — the "fused" reference, because TorchInductor is
  the production compiler that actually fuses the same producer-consumer
  pairs (Linear+ReLU, Scale+Softmax, Conv+BN) into single GPU kernels.

This is the honest CUDA analog of the existing NumPy-reference fusion
benchmark in :mod:`run_fusion_benchmark`. It measures *what fusion buys
on a real GPU when a serious framework fuses it*, not what our uTPU
ISA path buys (which has no separate "unfused" CUDA backend to time
against).

The subprocess pattern matches ``_cublas_baseline_torch_subprocess.py``
and ``inductor_oracle_subprocess.py`` so a parent process holding an
NVRTC driver context will not collide with Torch + Inductor's CUDA
contexts.

Methodology (locked in this file so the parent can describe it
verbatim):

* Same three workloads, same input shapes, same dtype (FP32 — matches
  Torch defaults; the dtype caveat vs the NumPy section is explicit).
* Warmup ``warmup`` invocations (default 5; Inductor needs at least 1
  warmup to compile + cache the kernel). Then ``iters`` measured
  invocations (default 30), each bracketed by
  ``torch.cuda.synchronize()`` so we only count kernel + dispatch wall
  time.
* Per-workload stats: mean, median, stdev, min, max, p95 (ms).
* Correctness: max abs error between eager and inductor outputs. Both
  paths are FP32, so we expect ~1e-5/1e-4 noise from fusion-reordered
  arithmetic; the tolerance is recorded in the artifact.

Failure modes are not silent: if a workload fails to compile under
Inductor, the entry records ``compile_error`` and zero samples; the
parent's correctness gate is then explicitly "skipped" for that
workload.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


WORKLOAD_SPECS = {
    "linear_relu_mlp_3x256": {
        "rule_name": "linear_relu_fusion",
        "input_shape": [64, 256],
        "description": "3-layer MLP (256x256) with ReLU between layers; activation fusion target",
    },
    "scale_softmax_attention_8x128x128": {
        "rule_name": "scale_softmax_fusion",
        "input_shape": [8, 128, 128],
        "description": "Pre-softmax scale + softmax over (8,128,128); attention-style fusion target",
    },
    "conv_bn_resnet_block_1x16x32x32": {
        "rule_name": "conv_bn_fusion",
        "input_shape": [1, 16, 32, 32],
        "description": "Conv2d(16->32,k=3,p=1) + BatchNorm2d, ResNet-style; weight-fold fusion target",
    },
}

DEFAULT_TOLERANCE_ABS = 1e-3
DEFAULT_TOLERANCE_REL = 1e-3


def _summary_ms(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "samples": 0}
    sorted_s = sorted(samples)
    p95 = max(0, int(round(0.95 * (len(sorted_s) - 1))))
    return {
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "stdev": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        "min": float(sorted_s[0]),
        "max": float(sorted_s[-1]),
        "p95": float(sorted_s[p95]),
        "samples": int(len(samples)),
    }


def _build_modules(torch_mod, workload: str):
    import torch.nn as nn

    if workload == "linear_relu_mlp_3x256":
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256, 256)
                self.fc2 = nn.Linear(256, 256)
                self.fc3 = nn.Linear(256, 256)

            def forward(self, x):
                h = self.fc1(x)
                h = torch_mod.relu(h)
                h = self.fc2(h)
                h = torch_mod.relu(h)
                h = self.fc3(h)
                h = torch_mod.relu(h)
                return h

        return MLP()

    if workload == "scale_softmax_attention_8x128x128":
        class ScaleSoftmax(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 0.125

            def forward(self, x):
                return torch_mod.softmax(x * self.scale, dim=-1)

        return ScaleSoftmax()

    if workload == "conv_bn_resnet_block_1x16x32x32":
        class ConvBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(32)

            def forward(self, x):
                return self.bn(self.conv(x))

        m = ConvBN()
        m.eval()
        return m

    raise ValueError(f"unknown workload: {workload}")


def _time_one(torch_mod, fn, warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn()
    torch_mod.cuda.synchronize()
    samples: List[float] = []
    for _ in range(iters):
        torch_mod.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch_mod.cuda.synchronize()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return samples


def _max_abs_rel(torch_mod, a, b) -> Dict[str, float]:
    a_f = a.detach().float()
    b_f = b.detach().float()
    abs_err = (a_f - b_f).abs()
    denom = a_f.abs().clamp(min=1e-12)
    rel = abs_err / denom
    return {
        "max_abs_error": float(abs_err.max().item()) if abs_err.numel() else 0.0,
        "max_rel_error": float(rel.max().item()) if rel.numel() else 0.0,
        "tolerance_abs": DEFAULT_TOLERANCE_ABS,
        "tolerance_rel": DEFAULT_TOLERANCE_REL,
    }


def _run_workload(torch_mod, name: str, seed: int, warmup: int, iters: int) -> Dict[str, Any]:
    spec = WORKLOAD_SPECS[name]
    device = torch_mod.device("cuda")

    torch_mod.manual_seed(int(seed))
    eager_mod = _build_modules(torch_mod, name).to(device=device, dtype=torch_mod.float32).eval()
    torch_mod.manual_seed(int(seed))
    compiled_mod = _build_modules(torch_mod, name).to(device=device, dtype=torch_mod.float32).eval()
    compiled_mod.load_state_dict(eager_mod.state_dict())

    inductor_fn = torch_mod.compile(compiled_mod, backend="inductor", fullgraph=True)

    gen = torch_mod.Generator(device=device).manual_seed(int(seed))
    x = torch_mod.randn(spec["input_shape"], generator=gen, device=device, dtype=torch_mod.float32)

    with torch_mod.no_grad():
        eager_out = eager_mod(x)

    inductor_out = None
    compile_error = None
    try:
        with torch_mod.no_grad():
            inductor_out = inductor_fn(x)
        torch_mod.cuda.synchronize()
    except Exception as exc:
        compile_error = f"{type(exc).__name__}: {exc}"

    correctness: Dict[str, Any]
    if inductor_out is not None:
        correctness = _max_abs_rel(torch_mod, eager_out, inductor_out)
        correctness["within_tolerance"] = bool(
            correctness["max_abs_error"] <= DEFAULT_TOLERANCE_ABS
            or correctness["max_rel_error"] <= DEFAULT_TOLERANCE_REL
        )
    else:
        correctness = {
            "max_abs_error": float("nan"),
            "max_rel_error": float("nan"),
            "tolerance_abs": DEFAULT_TOLERANCE_ABS,
            "tolerance_rel": DEFAULT_TOLERANCE_REL,
            "within_tolerance": False,
            "skipped_reason": compile_error,
        }

    eager_call = (lambda: eager_mod(x)) if eager_mod is not None else None
    inductor_call = (lambda: inductor_fn(x)) if compile_error is None else None

    eager_samples: List[float] = []
    inductor_samples: List[float] = []
    with torch_mod.no_grad():
        if eager_call is not None:
            eager_samples = _time_one(torch_mod, eager_call, warmup=warmup, iters=iters)
        if inductor_call is not None:
            inductor_samples = _time_one(torch_mod, inductor_call, warmup=warmup, iters=iters)

    eager_stats = _summary_ms(eager_samples)
    inductor_stats = _summary_ms(inductor_samples)
    median_eager = eager_stats.get("median", 0.0)
    median_inductor = inductor_stats.get("median", 0.0)
    throughput_delta_pct = (
        ((median_eager - median_inductor) / median_eager) * 100.0
        if median_eager > 0.0 and median_inductor > 0.0
        else 0.0
    )
    speedup = (median_eager / median_inductor) if median_inductor > 0.0 else float("nan")

    return {
        "workload": name,
        "description": spec["description"],
        "rule_name": spec["rule_name"],
        "input_shape": spec["input_shape"],
        "input_dtype": "float32",
        "seed": int(seed),
        "eager_kernel_ms": eager_stats,
        "inductor_kernel_ms": inductor_stats,
        "throughput_delta_pct": float(throughput_delta_pct),
        "speedup": float(speedup) if speedup == speedup else None,
        "correctness": correctness,
        "compile_error": compile_error,
    }


def run_cuda_fusion(seeds_per_workload: Dict[str, int], warmup: int, iters: int) -> Dict[str, Any]:
    try:
        import torch as torch_mod
    except Exception as exc:
        return {
            "status": "torch_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    if not torch_mod.cuda.is_available():
        return {
            "status": "cuda_unavailable",
            "reason": "torch.cuda.is_available() is False inside subprocess.",
            "torch_version": str(torch_mod.__version__),
        }

    device_idx = torch_mod.cuda.current_device()
    env = {
        "device_name": str(torch_mod.cuda.get_device_name(device_idx)),
        "device_capability": list(torch_mod.cuda.get_device_capability(device_idx)),
        "torch_version": str(torch_mod.__version__),
        "cuda_version": str(getattr(torch_mod.version, "cuda", "unknown")),
        "device_index": int(device_idx),
    }

    workload_results: List[Dict[str, Any]] = []
    for name in WORKLOAD_SPECS:
        seed = int(seeds_per_workload.get(name, 0xBEEF))
        try:
            entry = _run_workload(torch_mod, name=name, seed=seed, warmup=warmup, iters=iters)
        except Exception as exc:
            entry = {
                "workload": name,
                "description": WORKLOAD_SPECS[name]["description"],
                "rule_name": WORKLOAD_SPECS[name]["rule_name"],
                "input_shape": WORKLOAD_SPECS[name]["input_shape"],
                "input_dtype": "float32",
                "seed": int(seed),
                "error": f"{type(exc).__name__}: {exc}",
            }
        workload_results.append(entry)

    return {
        "status": "ok",
        "environment": env,
        "warmup": int(warmup),
        "iters": int(iters),
        "tolerance_abs": DEFAULT_TOLERANCE_ABS,
        "tolerance_rel": DEFAULT_TOLERANCE_REL,
        "workloads": workload_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-json", required=True, help="JSON dict mapping workload name -> seed int")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    seeds = json.loads(args.seeds_json)
    payload = run_cuda_fusion(
        seeds_per_workload=seeds,
        warmup=int(args.warmup),
        iters=int(args.iters),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
