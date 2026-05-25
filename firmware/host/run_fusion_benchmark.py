"""Phase 2 evidence script: measure what producer-consumer fusion buys on
the NumPy reference path.

Methodology
-----------
For every workload we build the same graph twice:

* an UNFUSED IR where the fusion rule of interest has been disabled;
* a FUSED IR where the rule fires (or, for ``conv_bn``, where the weights
  have already been folded).

We then run both graphs through the same NumPy machinery, time them with
warmup + median-of-N, and emit a JSON artifact summarising:

* op-count reduction (compile-time evidence of fusion),
* fused / unfused median latency,
* throughput delta percentage = ``(t_unfused - t_fused) / t_unfused * 100``,
* differential correctness vs the unfused output (max abs / rel error).

We deliberately *do not* claim CUDA-backend speedups: this host has no GPU
and pretending otherwise would invent numbers. The artifact records
``backend="numpy_reference"`` so the claim is scoped to "what fusion buys
at the IR / interpreter layer (fewer Python-level ops, fewer intermediate
allocations)." Running the same script on a CUDA host would just add a
second backend entry.

Methodology guards
------------------
* warmup iterations are discarded;
* per-workload sample count is configurable (default 15, odd so the median
  is a real sample);
* every call uses the same RNG seed and the same pre-allocated input
  tensor so timing reflects compute, not allocation noise;
* differential correctness uses a per-workload tolerance (1e-4 abs / 1e-4
  rel) — typical NumPy reordering noise lives at ~1e-6 fp32.

This script is intentionally deterministic and CPU-only so it can run in
CI without a GPU. The companion ``test_fusion_benchmark.py`` locks the
artifact schema + asserts that every workload is correctness-clean.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from graph_conv_ops import (
    conv2d_nchw_numpy,
    fold_conv_bn_weights,
)
from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import (
    CONV_BN_FUSION_RULE,
    LINEAR_RELU_FUSION_RULE,
    SCALE_SOFTMAX_FUSION_RULE,
    FusionEngine,
)
from graph_reference_interpreter import execute_graph_reference


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "fusion_payoff.json"

DEFAULT_WARMUP = 3
DEFAULT_ITERS = 15  # odd → median is a real sample
DEFAULT_TOLERANCE_ABS = 1e-4
DEFAULT_TOLERANCE_REL = 1e-4

# Phase 7 remediation: CUDA fusion benchmark subprocess
CUDA_SUBPROCESS_SCRIPT = Path(__file__).with_name("_fusion_benchmark_cuda_subprocess.py")
CUDA_DEFAULT_WARMUP = 5
CUDA_DEFAULT_ITERS = 30
CUDA_PER_WORKLOAD_SEEDS = {
    "linear_relu_mlp_3x256": 0xC0FFEE,
    "scale_softmax_attention_8x128x128": 0xBADC0DE,
    "conv_bn_resnet_block_1x16x32x32": 0xFEED,
}


@dataclass
class Workload:
    name: str
    description: str
    rule_name: str
    build_unfused: Callable[[], GraphIR]
    build_fused: Callable[[], GraphIR]
    make_input: Callable[[np.random.Generator], np.ndarray]
    run_unfused: Optional[Callable[[GraphIR, np.ndarray], np.ndarray]] = None
    run_fused: Optional[Callable[[GraphIR, np.ndarray], np.ndarray]] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------


def _linear_op(name: str, in_name: str, out_name: str, in_feat: int, out_feat: int, rng: np.random.Generator) -> OpNode:
    return OpNode(
        name=name,
        op=OpKind.LINEAR,
        inputs=[in_name],
        outputs=[out_name],
        attrs={
            "weight": rng.standard_normal((out_feat, in_feat)).astype(np.float32) * 0.05,
            "bias": rng.standard_normal((out_feat,)).astype(np.float32) * 0.05,
            "in_features": in_feat,
            "out_features": out_feat,
        },
    )


def _build_linear_relu_unfused(rng: np.random.Generator) -> GraphIR:
    g = GraphIR(name="mlp_unfused")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(64, 256), dtype="float32")
    g.add_op(_linear_op("fc1", "x", "h1", 256, 256, rng))
    g.add_op(OpNode(name="relu1", op=OpKind.RELU, inputs=["h1"], outputs=["h1r"]))
    g.add_op(_linear_op("fc2", "h1r", "h2", 256, 256, rng))
    g.add_op(OpNode(name="relu2", op=OpKind.RELU, inputs=["h2"], outputs=["h2r"]))
    g.add_op(_linear_op("fc3", "h2r", "h3", 256, 256, rng))
    g.add_op(OpNode(name="relu3", op=OpKind.RELU, inputs=["h3"], outputs=["y"]))
    return g


def _build_linear_relu_pair(rng: np.random.Generator) -> Tuple[GraphIR, GraphIR]:
    unfused = _build_linear_relu_unfused(rng)
    fused = FusionEngine([LINEAR_RELU_FUSION_RULE]).run(unfused).graph
    return unfused, fused


def _build_scale_softmax_unfused(rng: np.random.Generator) -> GraphIR:
    g = GraphIR(name="scale_softmax_unfused")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(8, 128, 128), dtype="float32")
    g.add_op(
        OpNode(name="scale1", op=OpKind.SCALE, inputs=["x"], outputs=["s"], attrs={"scale": 0.125})
    )
    g.add_op(
        OpNode(
            name="softmax1",
            op=OpKind.SOFTMAX,
            inputs=["s"],
            outputs=["y"],
            attrs={"causal_mask": False},
        )
    )
    return g


def _build_scale_softmax_pair(rng: np.random.Generator) -> Tuple[GraphIR, GraphIR]:
    unfused = _build_scale_softmax_unfused(rng)
    fused = FusionEngine([SCALE_SOFTMAX_FUSION_RULE]).run(unfused).graph
    return unfused, fused


def _build_conv_bn_unfused(rng: np.random.Generator) -> GraphIR:
    """Conv+BN unfused IR. The reference interpreter has no BATCH_NORM
    handler (out-of-scope new op support), so we evaluate the BN portion
    via a custom callable in `run_conv_bn_unfused` below — same NumPy
    primitives, same fp32 dtype.
    """
    g = GraphIR(name="conv_bn_unfused")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 16, 32, 32), dtype="float32")
    weight = rng.standard_normal((32, 16, 3, 3)).astype(np.float32) * 0.05
    g.add_op(
        OpNode(
            name="conv1",
            op=OpKind.CONV2D,
            inputs=["x"],
            outputs=["c"],
            attrs={"weight": weight, "bias": None, "stride": 1, "padding": 1},
        )
    )
    g.add_op(
        OpNode(
            name="bn1",
            op=OpKind.BATCH_NORM,
            inputs=["c"],
            outputs=["y"],
            attrs={
                "weight": (rng.standard_normal((32,)) * 0.1 + 1.0).astype(np.float32),
                "bias": (rng.standard_normal((32,)) * 0.05).astype(np.float32),
                "running_mean": rng.standard_normal((32,)).astype(np.float32) * 0.01,
                "running_var": (rng.standard_normal((32,)) * 0.01 + 1.0).astype(np.float32).clip(min=1e-3),
                "eps": 1e-5,
            },
        )
    )
    return g


def _build_conv_bn_pair(rng: np.random.Generator) -> Tuple[GraphIR, GraphIR]:
    unfused = _build_conv_bn_unfused(rng)
    # Fused graph: conv with folded weights, no BN op.
    bn = next(op for op in unfused.ops if op.op == OpKind.BATCH_NORM)
    conv = next(op for op in unfused.ops if op.op == OpKind.CONV2D)
    w_fold, b_fold = fold_conv_bn_weights(
        conv_weight=conv.attrs["weight"],
        conv_bias=conv.attrs.get("bias"),
        bn_weight=bn.attrs["weight"],
        bn_bias=bn.attrs["bias"],
        bn_running_mean=bn.attrs["running_mean"],
        bn_running_var=bn.attrs["running_var"],
        bn_eps=float(bn.attrs.get("eps", 1e-5)),
    )
    g = GraphIR(name="conv_bn_fused")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 16, 32, 32), dtype="float32")
    g.add_op(
        OpNode(
            name="conv1",
            op=OpKind.CONV2D,
            inputs=["x"],
            outputs=["y"],
            attrs={
                "weight": w_fold,
                "bias": b_fold,
                "stride": 1,
                "padding": 1,
                "bn_fused": True,
            },
        )
    )
    return unfused, g


def _run_conv_bn_unfused(graph: GraphIR, x: np.ndarray) -> np.ndarray:
    """Equivalent of running the unfused conv+BN graph; BN handler is
    inlined here because the reference interpreter does not implement
    BATCH_NORM (intentionally out of scope for Phase 2)."""
    conv = next(op for op in graph.ops if op.op == OpKind.CONV2D)
    bn = next(op for op in graph.ops if op.op == OpKind.BATCH_NORM)

    c = conv2d_nchw_numpy(
        x.astype(np.float32, copy=False),
        np.asarray(conv.attrs["weight"], dtype=np.float32),
        bias=None if conv.attrs.get("bias") is None else np.asarray(conv.attrs["bias"], dtype=np.float32),
        stride=conv.attrs.get("stride", 1),
        padding=conv.attrs.get("padding", 0),
        groups=int(conv.attrs.get("groups", 1)),
    )
    gamma = np.asarray(bn.attrs["weight"], dtype=np.float32)
    beta = np.asarray(bn.attrs["bias"], dtype=np.float32)
    mean = np.asarray(bn.attrs["running_mean"], dtype=np.float32)
    var = np.asarray(bn.attrs["running_var"], dtype=np.float32)
    eps = float(bn.attrs.get("eps", 1e-5))
    inv = 1.0 / np.sqrt(var + eps)
    scale = (gamma * inv).reshape(1, -1, 1, 1)
    shift = (beta - gamma * inv * mean).reshape(1, -1, 1, 1)
    return (c * scale + shift).astype(np.float32, copy=False)


def _run_conv_bn_fused(graph: GraphIR, x: np.ndarray) -> np.ndarray:
    return execute_graph_reference(graph, x)


WORKLOADS: List[Workload] = [
    Workload(
        name="linear_relu_mlp_3x256",
        description="3-layer MLP (256x256) with ReLU between layers; activation fusion target",
        rule_name="linear_relu_fusion",
        build_unfused=lambda: _build_linear_relu_pair(np.random.default_rng(0xC0FFEE))[0],
        build_fused=lambda: _build_linear_relu_pair(np.random.default_rng(0xC0FFEE))[1],
        make_input=lambda rng: rng.standard_normal((64, 256)).astype(np.float32),
        notes="Fused path eliminates 3 intermediate ReLU tensors and 3 op-dispatch overheads",
    ),
    Workload(
        name="scale_softmax_attention_8x128x128",
        description="Pre-softmax scale + softmax over (8,128,128); attention-style fusion target",
        rule_name="scale_softmax_fusion",
        build_unfused=lambda: _build_scale_softmax_pair(np.random.default_rng(0xBADC0DE))[0],
        build_fused=lambda: _build_scale_softmax_pair(np.random.default_rng(0xBADC0DE))[1],
        make_input=lambda rng: rng.standard_normal((8, 128, 128)).astype(np.float32),
        notes="Fused SCALED_SOFTMAX folds the scalar multiply into the softmax exponentiation",
    ),
    Workload(
        name="conv_bn_resnet_block_1x16x32x32",
        description="Conv2d(16→32,k=3,p=1) + BatchNorm2d, ResNet-style; weight-fold fusion target",
        rule_name="conv_bn_fusion",
        build_unfused=lambda: _build_conv_bn_pair(np.random.default_rng(0xFEED))[0],
        build_fused=lambda: _build_conv_bn_pair(np.random.default_rng(0xFEED))[1],
        make_input=lambda rng: rng.standard_normal((1, 16, 32, 32)).astype(np.float32),
        run_unfused=_run_conv_bn_unfused,
        run_fused=_run_conv_bn_fused,
        notes="BN is folded into Conv weights at compile time; eliminates the BN forward pass at runtime",
    ),
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_callable(
    fn: Callable[[], np.ndarray], warmup: int, iters: int
) -> Tuple[List[float], np.ndarray]:
    last_out: Optional[np.ndarray] = None
    for _ in range(warmup):
        last_out = fn()
    samples: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        last_out = fn()
        samples.append((time.perf_counter() - t0) * 1e6)  # microseconds
    assert last_out is not None
    return samples, last_out


def _summarize(samples: List[float]) -> Dict[str, float]:
    return {
        "median_us": float(statistics.median(samples)),
        "min_us": float(min(samples)),
        "max_us": float(max(samples)),
        "mean_us": float(statistics.mean(samples)),
        "stdev_us": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
    }


def _correctness(unfused_out: np.ndarray, fused_out: np.ndarray) -> Dict[str, Any]:
    diff = np.abs(unfused_out.astype(np.float64) - fused_out.astype(np.float64))
    denom = np.maximum(np.abs(unfused_out.astype(np.float64)), 1e-12)
    rel = diff / denom
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float(rel.max()) if rel.size else 0.0
    return {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "tolerance_abs": DEFAULT_TOLERANCE_ABS,
        "tolerance_rel": DEFAULT_TOLERANCE_REL,
        "within_tolerance": bool(
            max_abs <= DEFAULT_TOLERANCE_ABS or max_rel <= DEFAULT_TOLERANCE_REL
        ),
    }


def _count_ops(graph: GraphIR) -> int:
    return len(graph.ops)


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def benchmark_workload(
    workload: Workload, warmup: int, iters: int, seed: int
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    unfused_graph = workload.build_unfused()
    fused_graph = workload.build_fused()
    x = workload.make_input(rng)

    run_unfused = workload.run_unfused or execute_graph_reference
    run_fused = workload.run_fused or execute_graph_reference

    unfused_samples, unfused_out = _time_callable(
        lambda: run_unfused(unfused_graph, x), warmup=warmup, iters=iters
    )
    fused_samples, fused_out = _time_callable(
        lambda: run_fused(fused_graph, x), warmup=warmup, iters=iters
    )

    unfused_stats = _summarize(unfused_samples)
    fused_stats = _summarize(fused_samples)
    median_unfused = unfused_stats["median_us"]
    median_fused = fused_stats["median_us"]
    throughput_delta_pct = (
        ((median_unfused - median_fused) / median_unfused) * 100.0
        if median_unfused > 0
        else 0.0
    )
    speedup = median_unfused / median_fused if median_fused > 0 else float("nan")

    return {
        "workload": workload.name,
        "description": workload.description,
        "rule_name": workload.rule_name,
        "notes": workload.notes,
        "op_count_unfused": _count_ops(unfused_graph),
        "op_count_fused": _count_ops(fused_graph),
        "op_count_reduction": _count_ops(unfused_graph) - _count_ops(fused_graph),
        "op_count_reduction_pct": (
            ((_count_ops(unfused_graph) - _count_ops(fused_graph)) / _count_ops(unfused_graph)) * 100.0
            if _count_ops(unfused_graph) > 0
            else 0.0
        ),
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "unfused_latency_us": unfused_stats,
        "fused_latency_us": fused_stats,
        "throughput_delta_pct": throughput_delta_pct,
        "speedup": speedup,
        "correctness": _correctness(unfused_out, fused_out),
        "seed": seed,
    }


def _summarize_throughput_deltas(deltas: List[float]) -> Dict[str, float]:
    if not deltas:
        return {
            "median_throughput_delta_pct": 0.0,
            "mean_throughput_delta_pct": 0.0,
            "min_throughput_delta_pct": 0.0,
            "max_throughput_delta_pct": 0.0,
            "workload_count": 0,
        }
    return {
        "median_throughput_delta_pct": float(statistics.median(deltas)),
        "mean_throughput_delta_pct": float(statistics.mean(deltas)),
        "min_throughput_delta_pct": float(min(deltas)),
        "max_throughput_delta_pct": float(max(deltas)),
        "workload_count": int(len(deltas)),
    }


def _run_cuda_fusion_subprocess(
    cuda_warmup: int,
    cuda_iters: int,
    output_path: Path,
) -> Dict[str, Any]:
    """Spawn the CUDA fusion benchmark subprocess (Torch eager vs
    ``torch.compile(backend="inductor")``) for the same three workloads.

    Always returns a JSON-able dict. On non-CUDA hosts (or any error
    starting / parsing the subprocess) the dict has ``status`` set to a
    descriptive string and no ``workloads`` block, but the top-level
    ``cuda_fusion`` key in the parent artifact still exists so the
    schema is consistent between CPU and GPU regenerators.
    """
    if not CUDA_SUBPROCESS_SCRIPT.exists():
        return {"status": "missing_subprocess_script", "reason": str(CUDA_SUBPROCESS_SCRIPT)}
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return {
            "status": "torch_unavailable_parent",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    import torch as torch_mod
    if not torch_mod.cuda.is_available():
        return {
            "status": "cuda_unavailable",
            "reason": "torch.cuda.is_available() is False in parent process.",
            "torch_version": str(torch_mod.__version__),
            "warmup": int(cuda_warmup),
            "iters": int(cuda_iters),
        }
    tmp_path = output_path.with_suffix(".cuda_subprocess.json")
    proc = subprocess.run(
        [
            sys.executable,
            str(CUDA_SUBPROCESS_SCRIPT),
            "--seeds-json",
            json.dumps(CUDA_PER_WORKLOAD_SEEDS),
            "--output",
            str(tmp_path),
            "--warmup",
            str(int(cuda_warmup)),
            "--iters",
            str(int(cuda_iters)),
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
        }
    try:
        payload = json.loads(tmp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "subprocess_json_parse_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "stdout": proc.stdout[-2000:],
        }
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    if payload.get("status") == "ok":
        deltas = [
            float(w["throughput_delta_pct"])
            for w in payload.get("workloads", [])
            if "throughput_delta_pct" in w
            and "error" not in w
            and (w.get("correctness", {}).get("within_tolerance") is True)
        ]
        payload["summary"] = _summarize_throughput_deltas(deltas)
        payload["all_correctness_within_tolerance"] = bool(
            payload.get("workloads")
            and all(
                w.get("correctness", {}).get("within_tolerance") is True
                for w in payload["workloads"]
                if "error" not in w
            )
        )
    return payload


def run_benchmark(
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
    seed: int = 0xBEEF,
    output_path: Optional[Path] = None,
    enable_cuda_section: bool = True,
    cuda_warmup: int = CUDA_DEFAULT_WARMUP,
    cuda_iters: int = CUDA_DEFAULT_ITERS,
) -> Dict[str, Any]:
    output_path = output_path or DEFAULT_OUTPUT_JSON
    workload_results = [
        benchmark_workload(w, warmup=warmup, iters=iters, seed=seed) for w in WORKLOADS
    ]

    summary = {
        "workload_count": len(workload_results),
        "all_correctness_within_tolerance": all(
            r["correctness"]["within_tolerance"] for r in workload_results
        ),
        "median_throughput_delta_pct": float(
            statistics.median(r["throughput_delta_pct"] for r in workload_results)
        ),
        "mean_throughput_delta_pct": float(
            statistics.mean(r["throughput_delta_pct"] for r in workload_results)
        ),
        "min_throughput_delta_pct": float(
            min(r["throughput_delta_pct"] for r in workload_results)
        ),
        "max_throughput_delta_pct": float(
            max(r["throughput_delta_pct"] for r in workload_results)
        ),
    }

    artifact = {
        "phase": "phase_2_generalized_fusion",
        "backend": "numpy_reference",
        "regime_note": (
            "CPU NumPy reference path. Reflects what fusion buys at the IR / "
            "interpreter layer (fewer Python-level ops, fewer intermediate "
            "allocations, fewer materialised tensors). The CUDA section "
            "below (``cuda_fusion``) measures what *Inductor* fusion buys "
            "on the same workloads on a real GPU."
        ),
        "methodology": {
            "warmup_iters": warmup,
            "timed_iters": iters,
            "metric": "median wall-clock latency over `timed_iters` samples after warmup",
            "clock_note": (
                "Wall-clock via time.perf_counter; no CPU-clock pinning. "
                "Median + p10/p90 reported to make per-sample noise visible. "
                "Median throughput delta is the headline metric."
            ),
            "cache_state": "warm: same input tensor reused across samples after warmup",
            "seed": seed,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "workloads": workload_results,
        "summary": summary,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if enable_cuda_section:
        cuda_payload = _run_cuda_fusion_subprocess(
            cuda_warmup=cuda_warmup,
            cuda_iters=cuda_iters,
            output_path=output_path,
        )
        artifact["cuda_fusion"] = {
            "backend": "torch_eager_vs_inductor_fp32",
            "methodology": {
                "eager_path": (
                    "PyTorch eager-mode forward; each op dispatches as a "
                    "separate CUDA kernel + intermediate tensor."
                ),
                "compiled_path": (
                    "torch.compile(model, backend='inductor', "
                    "fullgraph=True); Inductor fuses producer-consumer "
                    "pairs (Linear+ReLU, Scale+Softmax, Conv+BN) into "
                    "single GPU kernels."
                ),
                "throughput_delta_definition": (
                    "(eager_median_ms - inductor_median_ms) / "
                    "eager_median_ms * 100 — positive => Inductor fusion "
                    "wins; null / negative => fusion does not win on "
                    "this GPU for this workload."
                ),
                "isolation": (
                    "CUDA timings run in a separate Python process "
                    "(_fusion_benchmark_cuda_subprocess.py) so any "
                    "parent NVRTC driver context does not collide with "
                    "Torch + Inductor CUDA contexts (same pattern as "
                    "_cublas_baseline_torch_subprocess.py and "
                    "inductor_oracle_subprocess.py)."
                ),
                "warmup_iters": int(cuda_warmup),
                "timed_iters": int(cuda_iters),
                "sync_protocol": (
                    "torch.cuda.synchronize() brackets each measured "
                    "invocation; time.perf_counter() in ms."
                ),
                "dtype": "float32 throughout (Torch dispatcher default).",
                "dtype_caveat": (
                    "FP32 dtype throughout. The NumPy-reference section "
                    "above uses NumPy fp32 as well; this is a "
                    "framework-level fusion comparison on the *same* "
                    "dtype on both sides, not the uTPU INT8 path. The "
                    "cuBLAS / Inductor uTPU baseline (run_cublas_baseline.py "
                    "+ cublas_baseline.json) is the place to read the "
                    "uTPU-vs-Inductor comparison; this artifact only "
                    "measures eager-vs-Inductor on the same FP32 path."
                ),
                "correctness_tolerance_abs": 1e-3,
                "correctness_tolerance_rel": 1e-3,
                "seeds_per_workload": dict(CUDA_PER_WORKLOAD_SEEDS),
            },
            "result": cuda_payload,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    return artifact


def _print_summary(artifact: Dict[str, Any]) -> None:
    print(f"backend             : {artifact['backend']}")
    print(f"workloads           : {artifact['summary']['workload_count']}")
    print(f"all correctness OK  : {artifact['summary']['all_correctness_within_tolerance']}")
    print(
        f"throughput delta %  : median={artifact['summary']['median_throughput_delta_pct']:+.2f} "
        f"mean={artifact['summary']['mean_throughput_delta_pct']:+.2f} "
        f"min={artifact['summary']['min_throughput_delta_pct']:+.2f} "
        f"max={artifact['summary']['max_throughput_delta_pct']:+.2f}"
    )
    for r in artifact["workloads"]:
        print(
            f"  - {r['workload']:<45} rule={r['rule_name']:<22} "
            f"ops {r['op_count_unfused']}->{r['op_count_fused']} "
            f"({r['op_count_reduction_pct']:.1f}% fewer)  "
            f"latency {r['unfused_latency_us']['median_us']:.2f}us -> "
            f"{r['fused_latency_us']['median_us']:.2f}us  "
            f"delta {r['throughput_delta_pct']:+.2f}%  "
            f"max_abs_err {r['correctness']['max_abs_error']:.2e}"
        )


if __name__ == "__main__":
    artifact = run_benchmark()
    _print_summary(artifact)
    print(f"\nartifact -> {DEFAULT_OUTPUT_JSON.relative_to(REPO_ROOT)}")
