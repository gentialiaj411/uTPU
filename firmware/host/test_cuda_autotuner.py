import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from cuda_autotuner import (
    CUDATuningSearchSpace,
    load_schedule_cache,
    make_cache_key,
    save_schedule_cache,
    tune_blocked_fc_shape,
)
from cuda_blocked_fc_backend import DEFAULT_CUDA_SCHEDULE_PARAMS, detect_cuda_environment
from pytorch_compiler import compile_mlp_model


class TinyIntegerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
            self.fc2.weight.copy_(torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _temp_cache_path():
    return os.path.join(tempfile.mkdtemp(prefix="utpu_autotune_"), "cache.json")


def _write_tiny_cache(path):
    cache = {"version": 1, "backend": "cuda_blocked_fc", "results": {}}
    for out_features, in_features in ((3, 4), (2, 3)):
        key = make_cache_key(out_features, in_features, 16, "int4_i32", "cuda")
        cache["results"][key] = {
            "best_schedule": dict(DEFAULT_CUDA_SCHEDULE_PARAMS),
            "shape": {"M": out_features, "N": 1, "K": in_features, "array_size": 16},
        }
    save_schedule_cache(cache, path)


def test_autotuner_search_space_nonempty():
    space = CUDATuningSearchSpace(threads_per_block=(64, 128), unroll_factor=(1, 2))
    candidates = space.candidates()
    assert candidates
    assert space.schema()["threads_per_block"]["values"] == [64, 128]


def test_autotuner_runs_one_shape():
    env = detect_cuda_environment()
    path = _temp_cache_path()
    result = tune_blocked_fc_shape(
        out_features=3,
        in_features=4,
        search_space=CUDATuningSearchSpace(threads_per_block=(64,), unroll_factor=(1,)),
        warmup=1,
        iters=2,
        cache_path=path,
    )
    if env.runtime_available:
        assert result.executed is True
        assert result.best_schedule["threads_per_block"] in (64, 128)
        assert result.best_latency_ms is not None
    else:
        assert result.executed is False


def test_autotuner_preserves_correctness():
    env = detect_cuda_environment()
    result = tune_blocked_fc_shape(
        out_features=5,
        in_features=7,
        search_space=CUDATuningSearchSpace(threads_per_block=(32,), unroll_factor=(1, 2)),
        warmup=1,
        iters=2,
        cache_path=_temp_cache_path(),
    )
    if env.runtime_available:
        assert result.max_abs_error == 0
        assert result.best_latency_ms is not None
    else:
        assert result.executed is False


def test_best_schedule_cache_roundtrip():
    path = _temp_cache_path()
    key = make_cache_key(5, 7, 16, "int4_i32", "cuda")
    cache = {"version": 1, "backend": "cuda_blocked_fc", "results": {}}
    cache["results"][key] = {"best_schedule": {"threads_per_block": 64, "unroll_factor": 2}}
    save_schedule_cache(cache, path)
    loaded = load_schedule_cache(path)
    assert loaded["results"][key]["best_schedule"]["threads_per_block"] == 64


def test_compiled_runtime_can_use_tuned_schedule():
    path = _temp_cache_path()
    _write_tiny_cache(path)
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda", use_tuned_schedule=True, autotune_cache_path=path)

    with torch.no_grad():
        y_compiled = compiled(x)
        y_ref = model(x)
    report = compiled.execution_report()
    max_abs = float(torch.max(torch.abs(y_compiled.detach().cpu() - y_ref.detach().cpu())).item())

    assert max_abs == 0.0
    assert report["fallback_ops"] == []
    assert any(trace["engine"] == "nvrtc_cuda_blocked_fc_tuned" for trace in report["op_traces"])


def test_fixed_schedule_still_available():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda", use_tuned_schedule=False)

    y = compiled(x)
    report = compiled.execution_report()

    assert tuple(y.shape) == (1, 2)
    assert report["fallback_ops"] == []
    assert any(trace["engine"] == "nvrtc_cuda_blocked_fc" for trace in report["op_traces"])


def run_all():
    test_autotuner_search_space_nonempty()
    test_autotuner_runs_one_shape()
    test_autotuner_preserves_correctness()
    test_best_schedule_cache_roundtrip()
    test_compiled_runtime_can_use_tuned_schedule()
    test_fixed_schedule_still_available()
    print("test_cuda_autotuner: PASS")


if __name__ == "__main__":
    run_all()
