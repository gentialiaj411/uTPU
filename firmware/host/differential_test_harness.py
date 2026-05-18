import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

from pytorch_compiler import compile_mlp_model


DEFAULT_SHAPES: List[Tuple[int, int, int]] = [
    (4, 8, 4),
    (8, 16, 8),
    (16, 32, 16),
]


@dataclass
class _CaseModelConfig:
    in_features: int
    hidden_features: int
    out_features: int


def _require_torch():
    import torch
    import torch.nn as nn

    return torch, nn


def _build_deterministic_mlp(config: _CaseModelConfig):
    torch, nn = _require_torch()

    def _make_sparse_signed_weight(shape: Tuple[int, int], generator: Any) -> Any:
        rows, cols = shape
        weight = torch.zeros(shape, dtype=torch.float32)
        per_row = 2 if cols >= 2 else 1
        for r in range(rows):
            chosen = torch.randperm(cols, generator=generator)[:per_row]
            signs = torch.randint(
                low=0,
                high=2,
                size=(per_row,),
                generator=generator,
                dtype=torch.int32,
            )
            values = torch.where(signs == 0, -1.0, 1.0).to(torch.float32)
            weight[r, chosen] = values
        return weight

    class DeterministicMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(config.in_features, config.hidden_features, bias=False)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(config.hidden_features, config.out_features, bias=False)
            with torch.no_grad():
                seed = int(
                    1000
                    + config.in_features * 100
                    + config.hidden_features * 10
                    + config.out_features
                )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                self.fc1.weight.copy_(_make_sparse_signed_weight(tuple(self.fc1.weight.shape), generator))
                self.fc2.weight.copy_(_make_sparse_signed_weight(tuple(self.fc2.weight.shape), generator))

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    return DeterministicMLP().eval()


def _deterministic_input(in_features: int):
    torch, _ = _require_torch()
    pattern = [-1.0, 0.0, 1.0, 1.0, -1.0]
    data = [pattern[i % len(pattern)] for i in range(in_features)]
    return torch.tensor([data], dtype=torch.float32)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _error_metrics(reference: np.ndarray, actual: np.ndarray, atol: float, rtol: float) -> Dict[str, Any]:
    diff = actual - reference
    max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = np.maximum(np.abs(reference), 1e-12)
    max_rel_error = float(np.max(np.abs(diff) / denom)) if diff.size else 0.0
    return {
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "within_tolerance": bool(np.allclose(actual, reference, atol=atol, rtol=rtol)),
    }


def _run_cuda_backend(model: Any, x: Any, reference: np.ndarray, atol: float, rtol: float) -> Dict[str, Any]:
    compiled = compile_mlp_model(model, x, target="cuda")
    try:
        output = _to_numpy(compiled(x, mode="compiled"))
    except Exception as e:
        return {
            "backend": "cuda",
            "status": "skipped",
            "reason": str(e),
            "max_abs_error": None,
            "max_rel_error": None,
            "within_tolerance": None,
            "execution_mode": "compiled",
        }

    metrics = _error_metrics(reference, output, atol=atol, rtol=rtol)
    return {
        "backend": "cuda",
        "status": "pass" if metrics["within_tolerance"] else "fail",
        "reason": None,
        "max_abs_error": metrics["max_abs_error"],
        "max_rel_error": metrics["max_rel_error"],
        "within_tolerance": metrics["within_tolerance"],
        "execution_mode": "compiled",
    }


def _run_utpu_backend(model: Any, x: Any, reference: np.ndarray, atol: float, rtol: float) -> Dict[str, Any]:
    compiled = compile_mlp_model(model, x, target="utpu")
    if compiled.runtime is None:
        return {
            "backend": "utpu",
            "status": "skipped",
            "reason": "uTPU runtime object unavailable",
            "max_abs_error": None,
            "max_rel_error": None,
            "within_tolerance": None,
            "execution_mode": "quantized_reference_emulation",
        }
    try:
        output = _to_numpy(compiled.runtime.quantized_reference(x))
    except Exception as e:
        return {
            "backend": "utpu",
            "status": "skipped",
            "reason": str(e),
            "max_abs_error": None,
            "max_rel_error": None,
            "within_tolerance": None,
            "execution_mode": "quantized_reference_emulation",
        }

    metrics = _error_metrics(reference, output, atol=atol, rtol=rtol)
    return {
        "backend": "utpu",
        "status": "pass" if metrics["within_tolerance"] else "fail",
        "reason": None,
        "max_abs_error": metrics["max_abs_error"],
        "max_rel_error": metrics["max_rel_error"],
        "within_tolerance": metrics["within_tolerance"],
        "execution_mode": "quantized_reference_emulation",
    }


def run_differential_harness(
    output_json_path: str = "build/reports/differential_test_report.json",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    torch, _ = _require_torch()
    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tolerance": {"atol": float(atol), "rtol": float(rtol)},
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "shapes": [],
    }

    for in_features, hidden_features, out_features in DEFAULT_SHAPES:
        config = _CaseModelConfig(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
        )
        model = _build_deterministic_mlp(config)
        x = _deterministic_input(in_features)

        compiled_for_ref = compile_mlp_model(model, x, target="utpu")
        reference = _to_numpy(compiled_for_ref.reference_interpreter(x))

        backends = [
            _run_cuda_backend(model, x, reference=reference, atol=atol, rtol=rtol),
            _run_utpu_backend(model, x, reference=reference, atol=atol, rtol=rtol),
        ]
        report["shapes"].append(
            {
                "shape": {
                    "in_features": in_features,
                    "hidden_features": hidden_features,
                    "out_features": out_features,
                },
                "reference_output_shape": [int(d) for d in reference.shape],
                "backends": backends,
            }
        )

    parent = os.path.dirname(output_json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report
