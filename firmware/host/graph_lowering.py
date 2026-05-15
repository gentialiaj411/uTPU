from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from lowering_types import BlockedFCLoweringRequest


@dataclass
class PlannedOp:
    graph_op: str
    op: str
    status: str
    request: Optional[BlockedFCLoweringRequest] = None
    fused_activation: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class GraphCompilePlan:
    graph_name: str
    lowered_ops: List[PlannedOp] = field(default_factory=list)
    fallback_ops: List[PlannedOp] = field(default_factory=list)
    unsupported_ops: List[PlannedOp] = field(default_factory=list)

    @property
    def is_fully_supported(self) -> bool:
        return not self.fallback_ops and not self.unsupported_ops


def _as_int4_array(data: Any) -> np.ndarray:
    return np.clip(np.rint(np.asarray(data)), -8, 7).astype(np.int8)


def _activation_for(
    op: OpNode,
    graph: GraphIR,
    activation_values: Optional[Dict[str, Any]],
) -> tuple[np.ndarray, bool]:
    input_name = op.inputs[0]
    if activation_values and input_name in activation_values:
        return _as_int4_array(activation_values[input_name]).flatten(), False

    in_features = int(op.attrs["in_features"])
    input_value = graph.values.get(input_name)
    if input_value is not None and input_value.shape:
        if int(input_value.shape[-1]) != in_features:
            raise ValueError(
                f"Linear op '{op.name}' expected input last dim {in_features}, "
                f"got shape {input_value.shape}"
            )
    return np.zeros((in_features,), dtype=np.int8), True


def _only_relu_consumer(linear: OpNode, graph: GraphIR, op_by_name: Dict[str, OpNode]) -> Optional[OpNode]:
    output = graph.values.get(linear.outputs[0])
    if output is None or len(output.consumers) != 1:
        return None
    consumer = op_by_name.get(output.consumers[0])
    if consumer is not None and consumer.op == OpKind.RELU:
        return consumer
    return None


def plan_blocked_fc_graph(
    graph: GraphIR,
    array_size: int = 16,
    apply_quant: bool = True,
    activation_values: Optional[Dict[str, Any]] = None,
    weight_addr: int = 0x080,
    input_addr: int = 0x000,
    result_addr: int = 0x100,
) -> GraphCompilePlan:
    plan = GraphCompilePlan(graph_name=graph.name)
    op_by_name = {op.name: op for op in graph.ops}
    consumed_as_fused_relu = set()

    for op in graph.ops:
        if op.name in consumed_as_fused_relu:
            continue

        if op.op == OpKind.LINEAR:
            relu = _only_relu_consumer(op, graph, op_by_name)
            activations, placeholder_activation = _activation_for(op, graph, activation_values)
            request = BlockedFCLoweringRequest(
                weights_int4=_as_int4_array(op.attrs["weight"]),
                activations_int4=activations,
                out_features=int(op.attrs["out_features"]),
                in_features=int(op.attrs["in_features"]),
                array_size=int(array_size),
                apply_relu=relu is not None,
                apply_quant=bool(apply_quant),
                weight_addr=int(weight_addr),
                input_addr=int(input_addr),
                result_addr=int(result_addr),
            )
            notes = []
            if placeholder_activation:
                notes.append(
                    "activation_values did not bind this input; request uses zero int4 placeholder activations"
                )
            planned = PlannedOp(
                graph_op=op.name,
                op=op.op,
                status="lowered",
                request=request,
                fused_activation=relu.name if relu is not None else None,
                notes=notes,
            )
            plan.lowered_ops.append(planned)
            if relu is not None:
                consumed_as_fused_relu.add(relu.name)
            continue

        if op.op in {OpKind.RELU, OpKind.ADD, OpKind.VIEW}:
            plan.fallback_ops.append(
                PlannedOp(
                    graph_op=op.name,
                    op=op.op,
                    status="fallback",
                    notes=[f"{op.op} is represented in Graph IR but not lowered to blocked-FC in this milestone"],
                )
            )
            continue

        plan.unsupported_ops.append(
            PlannedOp(
                graph_op=op.name,
                op=op.op,
                status="unsupported",
                notes=[f"No graph lowering rule for op '{op.op}'"],
            )
        )

    return plan
