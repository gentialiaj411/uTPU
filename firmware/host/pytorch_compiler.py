from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend_lowering import create_backend_lowerer
from compiled_runtime import CompiledMLPRuntime
from fx_importer import FXImportError, import_fx_graph_module
from graph_ir import GraphIR
from graph_lowering import GraphCompilePlan, PlannedOp, plan_blocked_fc_graph
from graph_runtime_plan import GraphRuntimePlan, build_graph_runtime_plan


class PyTorchCompileError(ValueError):
    pass


@dataclass
class BackendLoweredOp:
    graph_op: str
    op: str
    target: str
    lowering: Dict[str, Any]
    fused_activation: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class PyTorchCompileResult:
    model_name: str
    target: str
    fx_graph: Optional[Any] = None
    graph_ir: Optional[GraphIR] = None
    plan: Optional[GraphCompilePlan] = None
    runtime_plan: Optional[GraphRuntimePlan] = None
    backend_ops: List[BackendLoweredOp] = field(default_factory=list)
    runtime: Optional[CompiledMLPRuntime] = None
    import_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        runtime_ok = self.runtime_plan is None or self.runtime_plan.executable
        return (
            self.import_error is None
            and self.plan is not None
            and not self.plan.unsupported_ops
            and runtime_ok
        )

    @property
    def fully_lowered_to_backend(self) -> bool:
        return self.ok and self.plan is not None and self.plan.is_fully_supported

    @property
    def callable(self) -> bool:
        return self.runtime is not None and self.target == "cuda" and self.ok

    def __call__(self, *args, mode: str = "compiled"):
        if self.runtime is None:
            raise PyTorchCompileError(
                f"Compiled model for target '{self.target}' does not have an executable runtime"
            )
        return self.runtime(*args, mode=mode)

    def execution_report(self) -> Dict[str, Any]:
        if self.runtime is None:
            return {
                "target": self.target,
                "mode": "not_available",
                "backend_linear_ops_executed": 0,
                "pytorch_fallback_ops": 0,
                "fallback_ops": [],
                "import_error": self.import_error,
            }
        return self.runtime.execution_report()

    def benchmark(self, *args, warmup: int = 5, iters: int = 50) -> Dict[str, Any]:
        if self.runtime is None:
            raise PyTorchCompileError(
                f"Compiled model for target '{self.target}' does not have an executable runtime"
            )
        return self.runtime.benchmark(*args, warmup=warmup, iters=iters)

    def summary(self) -> Dict[str, Any]:
        fallback_ops = self.plan.fallback_ops if self.plan is not None else []
        unsupported_ops = self.plan.unsupported_ops if self.plan is not None else []
        graph_ops = self.graph_ir.ops if self.graph_ir is not None else []
        runtime_unsupported = self.runtime_plan.unsupported_ops if self.runtime_plan is not None else []
        return {
            "model_name": self.model_name,
            "target": self.target,
            "ok": self.ok,
            "callable": self.callable,
            "fully_lowered_to_backend": self.fully_lowered_to_backend,
            "graph_op_count": len(graph_ops),
            "backend_lowered_op_count": len(self.backend_ops),
            "runtime_op_count": len(self.runtime_plan.ops) if self.runtime_plan is not None else 0,
            "fallback_ops": [op.graph_op for op in fallback_ops],
            "unsupported_ops": [op.graph_op for op in unsupported_ops],
            "runtime_unsupported": runtime_unsupported,
            "import_error": self.import_error,
        }


def _require_torch():
    try:
        import torch
        from torch.fx.passes.shape_prop import ShapeProp
    except Exception as e:
        raise PyTorchCompileError(
            "PyTorch is required for compile_mlp_model, but it is not available"
        ) from e
    return torch, ShapeProp


def _as_input_tuple(example_inputs: Any) -> Tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def _activation_bindings(graph: GraphIR, example_inputs: Tuple[Any, ...]) -> Dict[str, Any]:
    bindings: Dict[str, Any] = {}
    for name, value in zip(graph.inputs, example_inputs):
        if hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        bindings[name] = arr.reshape(-1)
    return bindings


def _lower_backend_ops(
    planned_ops: List[PlannedOp],
    target: str,
) -> List[BackendLoweredOp]:
    lowerer = create_backend_lowerer(target)
    lowered = []
    for op in planned_ops:
        if op.request is None:
            continue
        lowered.append(
            BackendLoweredOp(
                graph_op=op.graph_op,
                op=op.op,
                target=target,
                lowering=lowerer.lower_blocked_fc(op.request),
                fused_activation=op.fused_activation,
                notes=list(op.notes),
            )
        )
    return lowered


def compile_mlp_model(
    model: Any,
    example_inputs: Any,
    target: str = "cuda",
    array_size: int = 16,
    apply_quant: bool = True,
    strict: bool = False,
    use_tuned_schedule: bool = False,
    autotune_cache_path: Optional[str] = None,
) -> PyTorchCompileResult:
    """
    Compile a small PyTorch MLP-style model into Graph IR plus blocked-FC backend lowering plans.

    This is a frontend/compiler entrypoint, not a full torch.compile backend. It traces with
    torch.fx, imports supported nodes into Graph IR, plans Linear/ReLU patterns, and lowers
    supported Linear ops through the existing uTPU/CUDA blocked-FC backend infrastructure.
    """
    torch, ShapeProp = _require_torch()
    inputs = _as_input_tuple(example_inputs)
    model_name = model.__class__.__name__
    target_name = (target or "cuda").strip().lower()

    try:
        fx_graph = torch.fx.symbolic_trace(model)
        ShapeProp(fx_graph).propagate(*inputs)
        graph = import_fx_graph_module(fx_graph, name=model_name)
    except FXImportError as e:
        if strict:
            raise PyTorchCompileError(str(e)) from e
        return PyTorchCompileResult(
            model_name=model_name,
            target=target_name,
            import_error=str(e),
        )

    plan = plan_blocked_fc_graph(
        graph,
        array_size=array_size,
        apply_quant=apply_quant,
        activation_values=_activation_bindings(graph, inputs),
    )
    runtime_plan = build_graph_runtime_plan(graph, target_name)
    backend_ops = _lower_backend_ops(plan.lowered_ops, target_name)
    result = PyTorchCompileResult(
        model_name=model_name,
        target=target_name,
        fx_graph=fx_graph,
        graph_ir=graph,
        plan=plan,
        runtime_plan=runtime_plan,
        backend_ops=backend_ops,
    )
    result.runtime = CompiledMLPRuntime(
        graph=graph,
        runtime_plan=runtime_plan,
        target=target_name,
        reference_model=model,
        array_size=array_size,
        use_tuned_schedule=use_tuned_schedule,
        autotune_cache_path=autotune_cache_path if autotune_cache_path is not None else None,
    )

    if strict and not result.fully_lowered_to_backend:
        fallback = [op.graph_op for op in plan.fallback_ops]
        unsupported = [op.graph_op for op in plan.unsupported_ops]
        runtime_unsupported = runtime_plan.unsupported_ops
        raise PyTorchCompileError(
            f"Model contains ops that were not lowered to backend. "
            f"fallback_ops={fallback}, unsupported_ops={unsupported}, "
            f"runtime_unsupported={runtime_unsupported}"
        )

    return result
