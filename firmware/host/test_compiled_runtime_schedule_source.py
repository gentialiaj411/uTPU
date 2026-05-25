"""Phase 7 remediation P2.1 contract: the runtime backend must actually
**consume** `RuntimeOpPlan.cuda_schedule` when
`schedule_source="cost_model"`, not silently re-search.

The cost-model choice is recorded on the runtime plan by
`build_graph_runtime_plan`. Prior to this remediation the CUDA backend
ignored that choice and re-derived the schedule from the autotuner
cache, so the Phase 1 claim "cost model drives schedule selection" was
aspirational. These tests lock the wiring so the cost-model-selected
schedule is the one passed to the CUDA backend.

Tests run host-only: the CUDA backend's NVRTC bindings are loaded
lazily on first execute, so we can instantiate `CompiledMLPRuntime`
with `target="cuda"` on a CPU-only host and exercise
`_schedule_params_for_op` directly.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from compiled_runtime import CompiledMLPRuntime, CompiledRuntimeError
from graph_runtime_plan import RuntimeOpPlan, build_graph_runtime_plan
from pytorch_compiler import compile_model


class _TinyMLP(nn.Module):
    def __init__(self, in_features: int = 64, hidden: int = 32, out: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, out, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _compile_mlp_cost_model(schedule_source: str | None = "cost_model"):
    torch.manual_seed(7)
    model = _TinyMLP().eval()
    x = torch.randn(1, 64)
    return compile_model(
        model, x, target="cuda",
        schedule_source=schedule_source,
    )


def test_runtime_plan_records_cost_model_schedule_for_every_linear_op():
    """`build_graph_runtime_plan` must populate `cuda_schedule` on every
    LINEAR / LINEAR_RELU op when target='cuda' (precondition for the
    backend to actually consume it).
    """
    compiled = _compile_mlp_cost_model()
    linear_ops = [op for op in compiled.runtime_plan.ops if op.op == "linear"]
    assert len(linear_ops) >= 2, "expected >= 2 linear ops in TinyMLP"
    for op in linear_ops:
        assert op.cuda_schedule is not None, (
            f"runtime plan recorded no cuda_schedule for {op.graph_op}"
        )
        assert "threads_per_block" in op.cuda_schedule
        assert "unroll_factor" in op.cuda_schedule
        provenance = op.cuda_schedule_provenance
        assert provenance is not None
        assert provenance.get("selector") == "cost_model.select"


def test_schedule_source_cost_model_returns_planned_schedule_for_every_op():
    """With `schedule_source='cost_model'`, the resolver must return the
    EXACT schedule dict recorded on `RuntimeOpPlan.cuda_schedule` for
    every op, tagged with source='cost_model'. No autotuner-cache
    lookup may shadow the planned choice.
    """
    compiled = _compile_mlp_cost_model("cost_model")
    runtime = compiled.runtime
    assert runtime.schedule_source == "cost_model"

    for op in runtime.runtime_plan.ops:
        if op.op != "linear":
            continue
        w_np = runtime.params[op.weight_buffer]
        params, source_tag = runtime._schedule_params_for_op(op, w_np)
        assert source_tag == "cost_model"
        assert params is not None
        for key in ("threads_per_block", "unroll_factor"):
            assert int(params[key]) == int(op.cuda_schedule[key]), (
                f"executed schedule for {op.graph_op} ({params}) does not "
                f"match the planned cost-model choice ({op.cuda_schedule}) "
                f"-- backend is silently re-searching"
            )


def test_schedule_source_cost_model_missing_returns_missing_tag():
    """If the runtime plan ops have `cuda_schedule=None` (e.g. selector
    was disabled), the resolver must NOT silently fall back to the
    autotuner cache when `schedule_source='cost_model'`; it must
    return source_tag='cost_model:missing' instead, so the A/B harness
    can detect coverage gaps.
    """
    compiled = compile_model(
        _TinyMLP().eval(), torch.randn(1, 64),
        target="cuda", schedule_source="cost_model",
    )
    runtime = compiled.runtime
    op_no_schedule = RuntimeOpPlan(
        graph_op="fake_linear",
        op="linear",
        inputs=["x"],
        output="y",
        weight_buffer="fake.weight",
        cuda_schedule=None,
    )
    params, source_tag = runtime._schedule_params_for_op(
        op_no_schedule, np.zeros((16, 16), dtype=np.float32),
    )
    assert params is None
    assert source_tag == "cost_model:missing"


def test_schedule_source_cost_model_then_autotuner_falls_back_only_when_planned_is_missing():
    """`cost_model_then_autotuner` must prefer the planned schedule when
    present (no silent re-search), and only consult the autotuner cache
    when the plan recorded `cuda_schedule=None`.
    """
    compiled = compile_model(
        _TinyMLP().eval(), torch.randn(1, 64),
        target="cuda", schedule_source="cost_model_then_autotuner",
    )
    runtime = compiled.runtime
    assert runtime.schedule_source == "cost_model_then_autotuner"

    op_with_plan = compiled.runtime_plan.ops[0]
    assert op_with_plan.cuda_schedule is not None
    params, source_tag = runtime._schedule_params_for_op(
        op_with_plan, np.zeros((16, 16), dtype=np.float32),
    )
    assert source_tag == "cost_model_then_autotuner.cost_model"
    assert int(params["threads_per_block"]) == int(op_with_plan.cuda_schedule["threads_per_block"])
    assert int(params["unroll_factor"]) == int(op_with_plan.cuda_schedule["unroll_factor"])


def test_schedule_source_none_disables_all_schedule_hints():
    """Default `schedule_source='none'` must return (None, 'none') for
    every op so the backend stays on its baseline kernel.
    """
    compiled = compile_model(
        _TinyMLP().eval(), torch.randn(1, 64),
        target="cuda", schedule_source="none",
    )
    runtime = compiled.runtime
    assert runtime.schedule_source == "none"
    for op in runtime.runtime_plan.ops:
        if op.op != "linear":
            continue
        params, source_tag = runtime._schedule_params_for_op(
            op, np.zeros((16, 16), dtype=np.float32),
        )
        assert params is None
        assert source_tag == "none"


def test_legacy_use_tuned_schedule_true_maps_to_autotuner_cache():
    """Backward-compat: `use_tuned_schedule=True` without explicit
    `schedule_source` must continue to mean "autotuner_cache", so
    existing call sites and CI artifacts do not break.
    """
    compiled = compile_model(
        _TinyMLP().eval(), torch.randn(1, 64),
        target="cuda", use_tuned_schedule=True,
    )
    assert compiled.runtime.schedule_source == "autotuner_cache"
    assert compiled.runtime.use_tuned_schedule is True


def test_explicit_schedule_source_overrides_legacy_flag():
    compiled = compile_model(
        _TinyMLP().eval(), torch.randn(1, 64),
        target="cuda", use_tuned_schedule=True, schedule_source="cost_model",
    )
    assert compiled.runtime.schedule_source == "cost_model"


def test_unknown_schedule_source_raises():
    model = _TinyMLP().eval()
    x = torch.randn(1, 64)
    with pytest.raises(CompiledRuntimeError, match="Unknown schedule_source"):
        compile_model(model, x, target="cuda", schedule_source="random_search")
