from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


def build_graph_runtime_plan(graph: GraphIR, target: str) -> GraphRuntimePlan:
    plan = GraphRuntimePlan(graph_name=graph.name, target=(target or "cuda").strip().lower())
    plan.memory_plan = dict(graph.metadata.get("memory_plan", {}))
    op_by_name: Dict[str, OpNode] = {op.name: op for op in graph.ops}
    consumed_relu = set()

    for input_name in graph.inputs:
        plan.input_buffers.append(_buffer_for_value(graph, input_name, kind="input"))

    produced_outputs = set(graph.outputs)
    for op in graph.ops:
        if op.name in consumed_relu:
            continue

        if op.op not in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
            plan.unsupported_ops.append(
                f"Runtime only supports Linear ops with optional fused ReLU; op '{op.name}' is '{op.op}'"
            )
            continue

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

        plan.ops.append(
            RuntimeOpPlan(
                graph_op=op.name,
                op=OpKind.LINEAR,
                inputs=list(op.inputs),
                output=output_name,
                weight_buffer=weight_name,
                bias_buffer=bias_name if op.attrs.get("bias") is not None else None,
                apply_relu=apply_relu,
            )
        )

        if output_name not in produced_outputs:
            plan.intermediate_buffers.append(_buffer_for_value(graph, output_name, kind="intermediate", source=op.name))

    for output_name in graph.outputs:
        plan.output_buffers.append(_buffer_for_value(graph, output_name, kind="output"))

    return plan
