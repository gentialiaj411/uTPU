import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOST_DIR = os.path.join(REPO_ROOT, "firmware", "host")
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from pytorch_compiler import compile_mlp_model


REPORT_PATH = os.path.join(REPO_ROOT, "build", "reports", "mlp_baseline_comparison.json")


@dataclass(frozen=True)
class ShapeConfig:
    name: str
    in_features: int
    hidden_features: int
    out_features: int

    @property
    def input_shape(self):
        return [1, self.in_features]

    @property
    def layer_shapes(self):
        return [
            {"op": "linear", "in_features": self.in_features, "out_features": self.hidden_features},
            {"op": "relu"},
            {"op": "linear", "in_features": self.hidden_features, "out_features": self.out_features},
        ]


class IntegerMLP(nn.Module):
    def __init__(self, cfg: ShapeConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.in_features, cfg.hidden_features, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg.hidden_features, cfg.out_features, bias=False)
        g = torch.Generator(device="cpu")
        g.manual_seed(1000 + cfg.in_features + cfg.hidden_features + cfg.out_features)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randint(-2, 3, self.fc1.weight.shape, generator=g, dtype=torch.float32))
            self.fc2.weight.copy_(torch.randint(-2, 3, self.fc2.weight.shape, generator=g, dtype=torch.float32))

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median_wall_ms(fn, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        _sync()
        samples = []
        for _ in range(iters):
            _sync()
            t0 = time.perf_counter()
            fn()
            _sync()
            samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


def _try_torch_compile(model, x, warmup: int, iters: int) -> Optional[float]:
    if not hasattr(torch, "compile"):
        return None
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        compiled(x)
        return _median_wall_ms(lambda: compiled(x), warmup=warmup, iters=iters)
    except Exception as e:
        print(f"torch.compile skipped: {e}")
        return None


def _matmul_baseline_ms(model: IntegerMLP, x, warmup: int, iters: int) -> float:
    w1 = model.fc1.weight.detach()
    w2 = model.fc2.weight.detach()
    return _median_wall_ms(lambda: torch.relu(x.matmul(w1.t())).matmul(w2.t()), warmup=warmup, iters=iters)


def _win_loss(compiled_ms: Optional[float], baseline_ms: Optional[float]) -> Optional[str]:
    if compiled_ms is None or baseline_ms is None:
        return None
    if compiled_ms < baseline_ms:
        return "win"
    if compiled_ms > baseline_ms:
        return "lose"
    return "tie"


def benchmark_shape(cfg: ShapeConfig, warmup: int = 10, iters: int = 50) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntegerMLP(cfg).eval().to(device)
    x = torch.randint(-2, 3, tuple(cfg.input_shape), dtype=torch.float32, device=device)

    eager_ms = _median_wall_ms(lambda: model(x), warmup=warmup, iters=iters)
    compile_ms = _try_torch_compile(model, x, warmup=warmup, iters=iters)
    matmul_ms = _matmul_baseline_ms(model, x, warmup=warmup, iters=iters)

    compiled = compile_mlp_model(model.cpu(), x.detach().cpu(), target="cuda")
    bench = compiled.benchmark(x.detach().cpu(), warmup=max(2, warmup // 2), iters=iters)

    with torch.no_grad():
        compiled_out = compiled(x.detach().cpu())
        quant_ref = torch.as_tensor(compiled.runtime.quantized_reference(x.detach().cpu()), dtype=torch.float32)
        float_ref = model.cpu()(x.detach().cpu())
    compiled_vs_quantized = float(torch.max(torch.abs(compiled_out.detach().cpu() - quant_ref)).item())
    quantized_vs_float = float(torch.max(torch.abs(quant_ref - float_ref.detach().cpu())).item())

    steady = float(bench["steady_state_wall_ms"])
    row = {
        "shape_name": cfg.name,
        "input_shape": cfg.input_shape,
        "layer_shapes": cfg.layer_shapes,
        "pytorch_eager_ms": eager_ms,
        "torch_compile_ms": compile_ms,
        "torch_matmul_or_cublas_ms": matmul_ms,
        "compiled_first_call_ms": float(bench["first_call_wall_ms"]),
        "compiled_steady_state_ms": steady,
        "compiled_kernel_ms": float(bench["kernel_time_ms"]),
        "compiled_h2d_ms": float(bench["h2d_time_ms"]),
        "compiled_d2h_ms": float(bench["d2h_time_ms"]),
        "compiled_h2d_count": int(bench["h2d_count"]),
        "compiled_d2h_count": int(bench["d2h_count"]),
        "compiled_setup_ms": float(bench["setup_time_ms"]),
        "compiled_compile_ms": float(bench["compile_time_ms"]),
        "compiled_end_to_end_ms": float(bench["h2d_time_ms"] + bench["kernel_time_ms"] + bench["d2h_time_ms"]),
        "compiled_vs_quantized_reference_max_error": compiled_vs_quantized,
        "quantized_reference_vs_float_pytorch_max_error": quantized_vs_float,
        "max_abs_error": compiled_vs_quantized,
        "fallback_ops": list(bench["fallback_ops"]),
        "backend_linear_ops_executed": int(bench["backend_linear_ops_executed"]),
        "backend_elementwise_ops_executed": int(bench["backend_elementwise_ops_executed"]),
        "wins_vs": {
            "pytorch_eager": _win_loss(steady, eager_ms),
            "torch_compile": _win_loss(steady, compile_ms),
            "torch_matmul_or_cublas": _win_loss(steady, matmul_ms),
        },
    }
    return row


def main():
    shapes = [
        ShapeConfig("tiny_mlp", 4, 3, 2),
        ShapeConfig("fc1_like_small", 64, 128, 64),
        ShapeConfig("fc2_like_small", 128, 64, 16),
        ShapeConfig("stress_256_256_128", 256, 256, 128),
    ]
    rows: List[Dict[str, Any]] = []
    for cfg in shapes:
        print(f"benchmarking {cfg.name} input={cfg.input_shape} layers={cfg.layer_shapes}")
        rows.append(benchmark_shape(cfg))

    report = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "warmup_policy": "PyTorch baselines use 10 warmups; compiled runtime uses first-call plus warmed steady-state.",
        "results": rows,
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("\nMLP baseline comparison")
    print("=======================")
    print("shape,pytorch_eager_ms,torch_compile_ms,torch_matmul_or_cublas_ms,compiled_first_call_ms,compiled_steady_state_ms,compiled_kernel_ms,h2d_count,d2h_count,compiled_vs_quantized_reference_max_error,quantized_reference_vs_float_pytorch_max_error,fallback_ops,wins_vs")
    for r in rows:
        torch_compile_text = "null" if r["torch_compile_ms"] is None else f"{r['torch_compile_ms']:.4f}"
        print(
            f"{r['shape_name']},{r['pytorch_eager_ms']:.4f},"
            f"{torch_compile_text},"
            f"{r['torch_matmul_or_cublas_ms']:.4f},"
            f"{r['compiled_first_call_ms']:.4f},{r['compiled_steady_state_ms']:.4f},"
            f"{r['compiled_kernel_ms']:.4f},{r['compiled_h2d_count']},{r['compiled_d2h_count']},"
            f"{r['compiled_vs_quantized_reference_max_error']:.8f},"
            f"{r['quantized_reference_vs_float_pytorch_max_error']:.8f},"
            f"{r['fallback_ops']},{r['wins_vs']}"
        )
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
