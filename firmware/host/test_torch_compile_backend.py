import json
import os
from pathlib import Path

import pytest

from cuda_blocked_fc_backend import detect_cuda_environment
from graph_passes import supported_ops_for_backend
from torch_compile_backend import (
    dump_stats_artifact,
    get_stats,
    register_backend,
    reset_stats,
)


def _require_torch():
    torch = pytest.importorskip("torch")
    return torch


def _require_dynamo():
    torch = _require_torch()
    if not hasattr(torch, "_dynamo") or not hasattr(torch._dynamo, "register_backend"):
        pytest.skip("torch._dynamo backend registration is unavailable on this platform/build")
    try:
        m = torch.nn.Linear(4, 4).eval()
        x = torch.randn(1, 4)
        compiled = torch.compile(m, backend="eager")
        _ = compiled(x)
    except Exception as e:
        pytest.skip(f"torch.compile/torch._dynamo unavailable or broken on this platform: {e}")
    return torch


def _require_dynamo_and_cuda():
    torch = _require_dynamo()
    env = detect_cuda_environment()
    if not env.runtime_available:
        pytest.skip(f"CUDA runtime unavailable for compiled CUDA execution: {env.reason}")
    return torch


def test_register_backend_idempotent():
    _require_dynamo()
    assert register_backend() is True
    assert register_backend() is True


def test_supported_mlp_compiles_and_runs_with_zero_fallback():
    torch = _require_dynamo_and_cuda()
    reset_stats()
    register_backend()

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 3)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(3, 2)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = TinyMLP().eval()
    x = torch.randn(1, 4)
    eager = model(x)
    compiled = torch.compile(model, backend="utpu", fullgraph=True)
    out = compiled(x)
    assert torch.allclose(out, eager, atol=1e-4, rtol=1e-4)
    stats = get_stats()
    assert stats["subgraphs_compiled"] >= 1
    assert stats["subgraphs_fallback"] == 0


def test_supported_plus_unsupported_mixed_falls_back_cleanly():
    torch = _require_dynamo()
    reset_stats()
    register_backend()

    class MixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

        def forward(self, x):
            y = self.fc(x)
            return torch.sin(y)

    model = MixedModel().eval()
    x = torch.randn(1, 4)
    eager = model(x)
    compiled = torch.compile(model, backend="utpu", fullgraph=True)
    out = compiled(x)
    assert torch.allclose(out, eager, atol=1e-4, rtol=1e-4)
    stats = get_stats()
    assert stats["subgraphs_fallback"] >= 1


def test_pure_unsupported_falls_back_cleanly():
    torch = _require_dynamo()
    reset_stats()
    register_backend()

    class UnsupportedModel(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x) + torch.cos(x)

    model = UnsupportedModel().eval()
    x = torch.randn(1, 4)
    eager = model(x)
    compiled = torch.compile(model, backend="utpu", fullgraph=True)
    out = compiled(x)
    assert torch.allclose(out, eager, atol=1e-4, rtol=1e-4)
    stats = get_stats()
    assert stats["subgraphs_fallback"] >= 1
    assert stats["subgraphs_compiled"] >= 0


def test_stats_artifact_schema_shape():
    payload = dump_stats_artifact("build/reports/torch_compile_backend_report.json")
    path = Path("build/reports/torch_compile_backend_report.json")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    for key in [
        "backend_name",
        "subgraphs_seen",
        "subgraphs_compiled",
        "subgraphs_fallback",
        "unsupported_op_counts",
        "supported_op_set",
        "models_exercised",
        "generated_at_utc",
        "torch_version",
        "dynamo_available",
    ]:
        assert key in loaded
    assert loaded["backend_name"] == "utpu"
    assert isinstance(loaded["unsupported_op_counts"], dict)
    if loaded["dynamo_available"]:
        # CPU-runnable fallback tests should contribute non-zero dispatch evidence.
        assert loaded["subgraphs_seen"] >= 1
        assert loaded["subgraphs_fallback"] >= 1
    assert loaded == payload


def test_supported_op_set_source_of_truth_matches_backend_legality():
    target = (os.environ.get("UTPU_TORCH_COMPILE_TARGET", "cuda") or "cuda").strip().lower()
    stats = get_stats()
    assert set(stats["supported_op_set"]) == set(supported_ops_for_backend(target))
