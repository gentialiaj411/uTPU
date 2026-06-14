from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from lowering_types import BatchedMatmulLoweringRequest, BlockedFCLoweringRequest, Conv2DIm2ColLoweringRequest


@dataclass
class PlannedOp:
    graph_op: str
    op: str
    status: str
    request: Optional[Any] = None
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


def _pack_int4(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    if flat.size % 2 == 1:
        flat = np.concatenate([flat, np.zeros((1,), dtype=np.int8)])
    lo = (flat[0::2] + 8).astype(np.uint8) & 0x0F
    hi = (flat[1::2] + 8).astype(np.uint8) & 0x0F
    return (lo | (hi << 4)).astype(np.uint8)


def _unpack_int4(packed: np.ndarray, size: int) -> np.ndarray:
    src = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.zeros((src.size * 2,), dtype=np.int8)
    out[0::2] = (src & 0x0F).astype(np.int8) - 8
    out[1::2] = ((src >> 4) & 0x0F).astype(np.int8) - 8
    return out[:size]


def quantize_weights_pass(graph: GraphIR, group_size: int = 64) -> GraphIR:
    quantized = copy.deepcopy(graph)
    for op in quantized.ops:
        if op.op not in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
            continue
        w = op.attrs.get("weight")
        if w is None:
            continue
        w_f = np.asarray(w, dtype=np.float32)
        out_features, in_features = int(w_f.shape[0]), int(w_f.shape[1])
        q = np.zeros_like(w_f, dtype=np.int8)
        num_groups = (in_features + group_size - 1) // group_size
        scales = np.ones((out_features, num_groups), dtype=np.float32)
        for o in range(out_features):
            for g in range(num_groups):
                s = g * group_size
                e = min(in_features, s + group_size)
                chunk = w_f[o, s:e]
                max_abs = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                scale = max(max_abs / 7.0, 1e-8)
                scales[o, g] = scale
                q[o, s:e] = np.clip(np.rint(chunk / scale), -8, 7).astype(np.int8)
        packed = _pack_int4(q)
        op.attrs["weight_fp32"] = w_f
        op.attrs["weight_int4_packed"] = packed
        op.attrs["weight_int4_shape"] = (out_features, in_features)
        op.attrs["weight_int4_scales"] = scales
        op.attrs["dtype_quant"] = "int4_g64"
        del op.attrs["weight"]
    return quantized


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
    target_backend: str = "cuda",
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

        if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
            if op.attrs.get("dtype_quant") == "int4_g64":
                packed = np.asarray(op.attrs["weight_int4_packed"], dtype=np.uint8)
                shape = tuple(op.attrs["weight_int4_shape"])
                q = _unpack_int4(packed, int(shape[0] * shape[1])).reshape(shape)
                scales = np.asarray(op.attrs["weight_int4_scales"], dtype=np.float32)
                deq = np.zeros(shape, dtype=np.float32)
                group_size = 64
                for o in range(shape[0]):
                    for g in range(scales.shape[1]):
                        s = g * group_size
                        e = min(shape[1], s + group_size)
                        deq[o, s:e] = q[o, s:e].astype(np.float32) * scales[o, g]
                int4_weights = _as_int4_array(deq)
            else:
                int4_weights = _as_int4_array(op.attrs["weight"])
            relu = _only_relu_consumer(op, graph, op_by_name)
            apply_relu = op.op == OpKind.LINEAR_RELU or relu is not None
            activations, placeholder_activation = _activation_for(op, graph, activation_values)
            request = BlockedFCLoweringRequest(
                weights_int4=int4_weights,
                activations_int4=activations,
                out_features=int(op.attrs["out_features"]),
                in_features=int(op.attrs["in_features"]),
                array_size=int(array_size),
                apply_relu=apply_relu,
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
                fused_activation=(relu.name if relu is not None else ("relu" if apply_relu else None)),
                notes=notes,
            )
            plan.lowered_ops.append(planned)
            if relu is not None:
                consumed_as_fused_relu.add(relu.name)
            continue

        if op.op == OpKind.CONV2D and str(target_backend).strip().lower() == "utpu":
            input_name = op.inputs[0]
            if activation_values is None or input_name not in activation_values:
                plan.fallback_ops.append(
                    PlannedOp(
                        graph_op=op.name,
                        op=op.op,
                        status="fallback",
                        notes=["conv2d lowering requires bound example activations for the current compiler path"],
                    )
                )
                continue
            plan.lowered_ops.append(
                PlannedOp(
                    graph_op=op.name,
                    op=op.op,
                    status="lowered",
                    request=Conv2DIm2ColLoweringRequest(
                        input_nchw=np.asarray(activation_values[input_name], dtype=np.float32),
                        weight_oihw=np.asarray(op.attrs["weight"], dtype=np.float32),
                        bias=None if op.attrs.get("bias") is None else np.asarray(op.attrs["bias"], dtype=np.float32),
                        stride=op.attrs.get("stride", 1),
                        padding=op.attrs.get("padding", 0),
                        dilation=op.attrs.get("dilation", 1),
                        groups=int(op.attrs.get("groups", 1)),
                        array_size=int(array_size),
                    ),
                    notes=[
                        "conv2d lowered through im2col into the batched blocked-FC GEMM datapath",
                        "current conv2d lowering is scoped to bias-free integer simulation cases",
                    ],
                )
            )
            continue

        if op.op == OpKind.BATCHED_MATMUL and str(target_backend).strip().lower() == "utpu":
            lhs_name, rhs_name = op.inputs[0], op.inputs[1]
            if activation_values is None or lhs_name not in activation_values or rhs_name not in activation_values:
                plan.fallback_ops.append(
                    PlannedOp(
                        graph_op=op.name,
                        op=op.op,
                        status="fallback",
                        notes=["batched_matmul lowering requires bound example tensors for both dynamic operands"],
                    )
                )
                continue
            lhs = np.asarray(activation_values[lhs_name], dtype=np.float32)
            rhs = np.asarray(activation_values[rhs_name], dtype=np.float32)
            if lhs.ndim < 2 or rhs.ndim < 2 or lhs.shape[:-2] != rhs.shape[:-2] or lhs.shape[-1] != rhs.shape[-2]:
                plan.unsupported_ops.append(
                    PlannedOp(
                        graph_op=op.name,
                        op=op.op,
                        status="unsupported",
                        notes=[f"incompatible batched_matmul shapes lhs={lhs.shape} rhs={rhs.shape}"],
                    )
                )
                continue
            plan.lowered_ops.append(
                PlannedOp(
                    graph_op=op.name,
                    op=op.op,
                    status="lowered",
                    request=BatchedMatmulLoweringRequest(
                        lhs_dynamic=np.clip(np.rint(lhs), -128, 127).astype(np.int8),
                        rhs_dynamic=np.clip(np.rint(rhs), -128, 127).astype(np.int8),
                        array_size=int(array_size),
                        apply_relu=False,
                        apply_quant=bool(apply_quant),
                        weight_addr=int(weight_addr),
                        input_addr=int(input_addr),
                        result_addr=int(result_addr),
                    ),
                    notes=[
                        "dynamic x dynamic batched_matmul lowered by reusing the streaming blocked-FC GEMM datapath",
                        "rhs batch slice is transposed into the runtime-stationary matrix; lhs rows are streamed as the activation batch",
                    ],
                )
            )
            continue

        if op.op in {
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
            OpKind.MAX_POOL2D,
            OpKind.ADAPTIVE_AVG_POOL2D,
        }:
            plan.fallback_ops.append(
                PlannedOp(
                    graph_op=op.name,
                    op=op.op,
                    status="fallback",
                    notes=[f"{op.op} is executed via graph-op runtime path, not blocked-FC lowering"],
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
