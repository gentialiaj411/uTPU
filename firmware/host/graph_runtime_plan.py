from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from graph_ir import GraphIR, OpKind, OpNode


@dataclass(frozen=True)
class BufferPlan:
    name: str
    kind: str
    shape: Optional[Tuple[int, ...]]
    dtype: Optional[str]
    source: Optional[str] = None


@dataclass(frozen=True)
class RuntimeOpPlan:
    graph_op: str
    op: str
    inputs: List[str]
    output: str
    weight_buffer: Optional[str] = None
    bias_buffer: Optional[str] = None
    apply_relu: bool = False
    # Phase 1: cost-model-selected CUDA blocked-FC schedule for this op
    # (LINEAR / LINEAR_RELU only, target == "cuda", known shape). `None`
    # means the cost model did not commit a choice (unknown shape, not a
    # blocked-FC op, or non-CUDA target). The backend is free to ignore
    # this hint; it is recorded for provenance and downstream selection
    # consumers.
    cuda_schedule: Optional[Dict[str, int]] = None
    cuda_schedule_provenance: Optional[Dict[str, Any]] = None


@dataclass
class GraphRuntimePlan:
    graph_name: str
    target: str
    input_buffers: List[BufferPlan] = field(default_factory=list)
    weight_buffers: List[BufferPlan] = field(default_factory=list)
    intermediate_buffers: List[BufferPlan] = field(default_factory=list)
    output_buffers: List[BufferPlan] = field(default_factory=list)
    ops: List[RuntimeOpPlan] = field(default_factory=list)
    unsupported_ops: List[str] = field(default_factory=list)
    memory_plan: Dict[str, Any] = field(default_factory=dict)

    @property
    def executable(self) -> bool:
        return not self.unsupported_ops


def _buffer_for_value(graph: GraphIR, name: str, kind: str, source: Optional[str] = None) -> BufferPlan:
    value = graph.values.get(name)
    return BufferPlan(
        name=name,
        kind=kind,
        shape=value.shape if value is not None else None,
        dtype=value.dtype if value is not None else None,
        source=source,
    )


_DEFAULT_CUDA_SCHEDULE_GRID: Tuple[Dict[str, int], ...] = tuple(
    {"threads_per_block": t, "unroll_factor": u}
    for t in (32, 64, 128, 256)
    for u in (1, 2, 4, 8)
)


def _default_cuda_schedule_selector(
    out_features: int,
    in_features: int,
    array_size: int,
) -> Optional[Dict[str, Any]]:
    """Default cost-model selection: pick a CUDA blocked-FC schedule for the op.

    Imported lazily to avoid a circular dependency between graph_runtime_plan
    and the calibrated cost-model target loader.
    """
    from cost_model import select as cost_model_select  # local to avoid cycles
    try:
        from cuda_autotuner import load_cost_model_target
        target = load_cost_model_target()
    except Exception:
        target = "cuda"

    shape = {
        "out_features": int(out_features),
        "in_features": int(in_features),
        "batch": 1,
        "array_size": int(array_size),
        "apply_quant": True,
    }
    choice = cost_model_select(shape, _DEFAULT_CUDA_SCHEDULE_GRID, target=target)
    return {
        "schedule": dict(choice.schedule),
        "provenance": {
            "selector": "cost_model.select",
            "candidates_considered": int(choice.candidates_considered),
            "predicted_latency_us": float(choice.predicted_latency_us),
            "runner_up_schedule": dict(choice.runner_up_schedule) if choice.runner_up_schedule else None,
            "runner_up_predicted_latency_us": (
                float(choice.runner_up_predicted_latency_us)
                if choice.runner_up_predicted_latency_us is not None
                else None
            ),
            "margin_pct": float(choice.margin_pct),
            "confidence": float(choice.confidence),
            "target_name": choice.target_name,
            "rank": int(choice.rank),
        },
    }


def build_graph_runtime_plan(
    graph: GraphIR,
    target: str,
    cuda_schedule_selector: Optional[Callable[[int, int, int], Optional[Dict[str, Any]]]] = None,
) -> GraphRuntimePlan:
    target_name = (target or "cuda").strip().lower()
    plan = GraphRuntimePlan(graph_name=graph.name, target=target_name)
    plan.memory_plan = dict(graph.metadata.get("memory_plan", {}))
    op_by_name: Dict[str, OpNode] = {op.name: op for op in graph.ops}
    consumed_relu = set()

    # Default cost-model selector is wired only on CUDA. Caller may pass an
    # explicit selector (e.g. for tests, or to point at a different cost
    # model). Passing `lambda *_: None` disables the hint entirely.
    if cuda_schedule_selector is None and target_name == "cuda":
        cuda_schedule_selector = _default_cuda_schedule_selector

    for input_name in graph.inputs:
        plan.input_buffers.append(_buffer_for_value(graph, input_name, kind="input"))

    produced_outputs = set(graph.outputs)
    for op in graph.ops:
        if op.name in consumed_relu:
            continue

        if op.op not in {
            OpKind.LINEAR,
            OpKind.LINEAR_RELU,
            OpKind.RELU,
            OpKind.ADD,
            OpKind.VIEW,
            OpKind.PERMUTE,
            OpKind.SOFTMAX,
            OpKind.LAYER_NORM,
            OpKind.SCALED_DOT_PRODUCT_ATTENTION,
            OpKind.BATCHED_MATMUL,
            OpKind.SCALE,
            OpKind.SCALED_SOFTMAX,
            OpKind.CONV2D,
            OpKind.MAX_POOL2D,
            OpKind.ADAPTIVE_AVG_POOL2D,
        }:
            plan.unsupported_ops.append(f"Runtime does not support op '{op.name}' kind '{op.op}'")
            continue

        if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
            output_name = op.outputs[0]
            apply_relu = op.op == OpKind.LINEAR_RELU
            if not apply_relu:
                output_value = graph.values.get(output_name)
                if output_value is not None and len(output_value.consumers) == 1:
                    consumer = op_by_name.get(output_value.consumers[0])
                    if consumer is not None and consumer.op == OpKind.RELU:
                        apply_relu = True
                        consumed_relu.add(consumer.name)
                        output_name = consumer.outputs[0]

            weight_name = f"{op.name}.weight"
            bias_name = f"{op.name}.bias"
            plan.weight_buffers.append(
                BufferPlan(
                    name=weight_name,
                    kind="weight",
                    shape=tuple(op.attrs["weight"].shape),
                    dtype=str(op.attrs["weight"].dtype),
                    source=op.name,
                )
            )
            if op.attrs.get("bias") is not None:
                plan.weight_buffers.append(
                    BufferPlan(
                        name=bias_name,
                        kind="bias",
                        shape=tuple(op.attrs["bias"].shape),
                        dtype=str(op.attrs["bias"].dtype),
                        source=op.name,
                    )
                )

            cuda_schedule: Optional[Dict[str, int]] = None
            cuda_schedule_provenance: Optional[Dict[str, Any]] = None
            if cuda_schedule_selector is not None:
                weight_shape = tuple(op.attrs["weight"].shape)
                if len(weight_shape) == 2:
                    out_features = int(weight_shape[0])
                    in_features = int(weight_shape[1])
                    array_size = int(op.attrs.get("array_size", 16))
                    try:
                        choice_dict = cuda_schedule_selector(out_features, in_features, array_size)
                    except Exception as exc:
                        choice_dict = None
                        cuda_schedule_provenance = {
                            "selector_error": f"{type(exc).__name__}: {exc}",
                        }
                    if choice_dict is not None:
                        cuda_schedule = dict(choice_dict.get("schedule", {})) or None
                        provenance = choice_dict.get("provenance")
                        if provenance is not None:
                            cuda_schedule_provenance = dict(provenance)

            plan.ops.append(
                RuntimeOpPlan(
                    graph_op=op.name,
                    op=OpKind.LINEAR,
                    inputs=list(op.inputs),
                    output=output_name,
                    weight_buffer=weight_name,
                    bias_buffer=bias_name if op.attrs.get("bias") is not None else None,
                    apply_relu=apply_relu,
                    cuda_schedule=cuda_schedule,
                    cuda_schedule_provenance=cuda_schedule_provenance,
                )
            )
        else:
            plan.ops.append(
                RuntimeOpPlan(
                    graph_op=op.name,
                    op=op.op,
                    inputs=list(op.inputs),
                    output=op.outputs[0],
                )
            )

        op_output_name = plan.ops[-1].output
        if op_output_name not in produced_outputs:
            plan.intermediate_buffers.append(
                _buffer_for_value(graph, op_output_name, kind="intermediate", source=op.name)
            )

    for output_name in graph.outputs:
        plan.output_buffers.append(_buffer_for_value(graph, output_name, kind="output"))

    return plan
