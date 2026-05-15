import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOST_DIR = os.path.join(REPO_ROOT, "firmware", "host")
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from cuda_autotuner import DEFAULT_CACHE_PATH, lookup_best_schedule, tune_many_shapes
from pytorch_compiler import compile_mlp_model


REPORT_PATH = os.path.join(REPO_ROOT, "build", "reports", "tuned_mlp_benchmark.json")


@dataclass(frozen=True)
class ShapeConfig:
    name: str
    in_features: int
    hidden_features: int
    out_features: int


class IntegerMLP(nn.Module):
    def __init__(self, cfg: ShapeConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.in_features, cfg.hidden_features, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg.hidden_features, cfg.out_features, bias=False)
        g = torch.Generator(device="cpu")
        g.manual_seed(2000 + cfg.in_features + cfg.hidden_features + cfg.out_features)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randint(-2, 3, self.fc1.weight.shape, generator=g, dtype=torch.float32))
            self.fc2.weight.copy_(torch.randint(-2, 3, self.fc2.weight.shape, generator=g, dtype=torch.float32))

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _ensure_cache_for_shapes(shapes: List[ShapeConfig]) -> None:
    all_linear_shapes = []
    missing = []
    for cfg in shapes:
        for name, out_features, in_features in (
            (f"{cfg.name}_fc1", cfg.hidden_features, cfg.in_features),
            (f"{cfg.name}_fc2", cfg.out_features, cfg.hidden_features),
        ):
            all_linear_shapes.append((name, out_features, in_features))
            if lookup_best_schedule(out_features, in_features, path=DEFAULT_CACHE_PATH) is None:
                missing.append((name, out_features, in_features))
    if missing:
        tune_many_shapes(all_linear_shapes, warmup=1, iters=3, cache_path=DEFAULT_CACHE_PATH)


def benchmark_shape(cfg: ShapeConfig) -> Dict[str, Any]:
    model = IntegerMLP(cfg).eval()
    x = torch.randint(-2, 3, (1, cfg.in_features), dtype=torch.float32)

    fixed = compile_mlp_model(model, x, target="cuda", use_tuned_schedule=False)
    tuned = compile_mlp_model(model, x, target="cuda", use_tuned_schedule=True, autotune_cache_path=DEFAULT_CACHE_PATH)

    fixed_bench = fixed.benchmark(x, warmup=3, iters=20)
    tuned_bench = tuned.benchmark(x, warmup=3, iters=20)

    with torch.no_grad():
        tuned_out = tuned(x)
        fixed_out = fixed(x)
        quant_ref = torch.as_tensor(tuned.runtime.quantized_reference(x), dtype=torch.float32)
        float_ref = model(x)
    tuned_vs_quantized = float(torch.max(torch.abs(tuned_out.detach().cpu() - quant_ref)).item())
    fixed_vs_quantized = float(torch.max(torch.abs(fixed_out.detach().cpu() - quant_ref)).item())
    quantized_vs_float = float(torch.max(torch.abs(quant_ref - float_ref.detach().cpu())).item())

    fixed_ms = float(fixed_bench["kernel_time_ms"])
    tuned_ms = float(tuned_bench["kernel_time_ms"])
    kernel_improvement = ((fixed_ms - tuned_ms) / fixed_ms) * 100.0 if fixed_ms > 0.0 else None
    fixed_wall = float(fixed_bench["steady_state_wall_ms"])
    tuned_wall = float(tuned_bench["steady_state_wall_ms"])
    wall_improvement = ((fixed_wall - tuned_wall) / fixed_wall) * 100.0 if fixed_wall > 0.0 else None
    return {
        "shape_name": cfg.name,
        "input_shape": [1, cfg.in_features],
        "layer_shapes": [
            {"op": "linear", "in_features": cfg.in_features, "out_features": cfg.hidden_features},
            {"op": "relu"},
            {"op": "linear", "in_features": cfg.hidden_features, "out_features": cfg.out_features},
        ],
        "fixed_steady_state_ms": fixed_wall,
        "tuned_steady_state_ms": tuned_wall,
        "fixed_kernel_ms": fixed_ms,
        "tuned_kernel_ms": tuned_ms,
        "kernel_improvement_pct": kernel_improvement,
        "steady_state_improvement_pct": wall_improvement,
        "fixed_h2d_ms": float(fixed_bench["h2d_time_ms"]),
        "fixed_d2h_ms": float(fixed_bench["d2h_time_ms"]),
        "tuned_h2d_ms": float(tuned_bench["h2d_time_ms"]),
        "tuned_d2h_ms": float(tuned_bench["d2h_time_ms"]),
        "fixed_h2d_count": int(fixed_bench["h2d_count"]),
        "fixed_d2h_count": int(fixed_bench["d2h_count"]),
        "tuned_h2d_count": int(tuned_bench["h2d_count"]),
        "tuned_d2h_count": int(tuned_bench["d2h_count"]),
        "fixed_fallback_ops": list(fixed_bench["fallback_ops"]),
        "tuned_fallback_ops": list(tuned_bench["fallback_ops"]),
        "fixed_vs_quantized_reference_max_error": fixed_vs_quantized,
        "tuned_vs_quantized_reference_max_error": tuned_vs_quantized,
        "compiled_vs_quantized_reference_max_error": tuned_vs_quantized,
        "quantized_reference_vs_float_pytorch_max_error": quantized_vs_float,
        "max_abs_error": tuned_vs_quantized,
        "tuned_op_traces": tuned_bench["last_op_traces"],
    }


def main():
    shapes = [
        ShapeConfig("tiny_mlp", 4, 3, 2),
        ShapeConfig("fc1_like_small", 64, 128, 64),
        ShapeConfig("fc2_like_small", 128, 64, 16),
    ]
    _ensure_cache_for_shapes(shapes)
    rows = [benchmark_shape(cfg) for cfg in shapes]

    report = {
        "autotune_cache_path": DEFAULT_CACHE_PATH,
        "results": rows,
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("Tuned MLP benchmark")
    print("===================")
    print("shape,fixed_kernel_ms,tuned_kernel_ms,kernel_improvement_pct,fixed_steady_state_ms,tuned_steady_state_ms,steady_state_improvement_pct,tuned_h2d_count,tuned_d2h_count,tuned_vs_quantized_reference_max_error,quantized_reference_vs_float_pytorch_max_error,tuned_fallback_ops")
    for r in rows:
        print(
            f"{r['shape_name']},{r['fixed_kernel_ms']:.4f},{r['tuned_kernel_ms']:.4f},"
            f"{r['kernel_improvement_pct']:.4f},{r['fixed_steady_state_ms']:.4f},"
            f"{r['tuned_steady_state_ms']:.4f},{r['steady_state_improvement_pct']:.4f},"
            f"{r['tuned_h2d_count']},{r['tuned_d2h_count']},"
            f"{r['tuned_vs_quantized_reference_max_error']:.8f},"
            f"{r['quantized_reference_vs_float_pytorch_max_error']:.8f},{r['tuned_fallback_ops']}"
        )
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
