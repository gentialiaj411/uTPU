import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from graph_conv_ops import fold_conv_bn_weights
from graph_ir import GraphIR, OpKind, OpNode

_BACKEND_SUPPORTED_OPS: Dict[str, Set[str]] = {
    "cuda": {
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
    },
    "utpu": {OpKind.LINEAR, OpKind.LINEAR_RELU},
}


def supported_ops_for_backend(backend: str) -> Set[str]:
    target = (backend or "utpu").strip().lower()
    if target not in _BACKEND_SUPPORTED_OPS:
        raise BackendLegalityError(target, [{"op": target, "reason": "unknown_backend"}])
    return set(_BACKEND_SUPPORTED_OPS[target])


def is_op_supported_for_backend(op_kind: str, backend: str) -> bool:
    return op_kind in supported_ops_for_backend(backend)


class BackendLegalityError(ValueError):
    def __init__(self, backend: str, offending_ops: List[Dict[str, Any]]):
        self.backend = backend
        self.offending_ops = offending_ops
        details = json.dumps(offending_ops, sort_keys=True)
        super().__init__(f"backend_legality failed for backend='{backend}': {details}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "offending_ops": list(self.offending_ops),
        }


@dataclass
class PassRecord:
    pass_name: str
    before: Dict[str, Any]
    after: Dict[str, Any]


@dataclass
class PassPipelineResult:
    graph: GraphIR
    records: List[PassRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_name": self.graph.name,
            "passes": [
                {
                    "pass_name": record.pass_name,
                    "before": record.before,
                    "after": record.after,
                }
                for record in self.records
            ],
        }


def _clone_graph(graph: GraphIR) -> GraphIR:
    return copy.deepcopy(graph)


def _shape_from_view_args(args: Any) -> Any:
    if not args:
        return None
    first = args[0]
    if isinstance(first, (tuple, list)) and all(isinstance(v, int) for v in first):
        return tuple(int(v) for v in first)
    if all(isinstance(v, int) for v in args):
        return tuple(int(v) for v in args)
    return None


def _shape_from_permute_args(shape: Optional[Tuple[int, ...]], args: Any) -> Any:
    if shape is None:
        return None
    dims = None
    if args and isinstance(args[0], (tuple, list)):
        dims = tuple(int(v) for v in args[0])
    elif args:
        dims = tuple(int(v) for v in args)
    if dims is None or len(dims) != len(shape):
        return shape
    return tuple(shape[i] for i in dims)


def _graph_to_dict(graph: GraphIR) -> Dict[str, Any]:
    def clean_attr(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            out = {
                "shape": [int(d) for d in value.shape],
                "dtype": str(value.dtype),
                "size": int(value.size),
            }
            if value.size:
                out["min"] = float(np.min(value))
                out["max"] = float(np.max(value))
            return out
        if isinstance(value, (tuple, list)):
            return [clean_attr(v) for v in value]
        if isinstance(value, dict):
            return {str(k): clean_attr(v) for k, v in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return type(value).__name__

    return {
        "name": graph.name,
        "inputs": list(graph.inputs),
        "outputs": list(graph.outputs),
        "values": {
            name: {
                "shape": list(value.shape) if value.shape is not None else None,
                "dtype": value.dtype,
                "producer": value.producer,
                "consumers": list(value.consumers),
                "persistent": bool(getattr(value, "persistent", False)),
            }
            for name, value in graph.values.items()
        },
        "metadata": clean_attr(graph.metadata),
        "ops": [
            {
                "name": op.name,
                "op": op.op,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attrs": {k: clean_attr(v) for k, v in op.attrs.items()},
            }
            for op in graph.ops
        ],
    }


def _copy_value_metadata(src: GraphIR, dst: GraphIR, value_name: str) -> None:
    value = src.values.get(value_name)
    if value is None:
        dst.add_value(value_name)
        return
    dst.add_value(
        value_name,
        shape=value.shape,
        dtype=value.dtype,
        persistent=bool(getattr(value, "persistent", False)),
    )


def _rebuild_graph_with_ops(src: GraphIR, ops: List[OpNode]) -> GraphIR:
    dst = GraphIR(name=src.name)
    dst.inputs = list(src.inputs)
    dst.outputs = list(src.outputs)
    dst.metadata = copy.deepcopy(src.metadata)

    for input_name in dst.inputs:
        _copy_value_metadata(src, dst, input_name)

    for op in ops:
        for input_name in op.inputs:
            _copy_value_metadata(src, dst, input_name)
        for output_name in op.outputs:
            _copy_value_metadata(src, dst, output_name)
        dst.add_op(copy.deepcopy(op))

    for output_name in dst.outputs:
        _copy_value_metadata(src, dst, output_name)

    return dst


def shape_inference_pass(graph: GraphIR) -> GraphIR:
    inferred = _clone_graph(graph)

    for op in inferred.ops:
        if not op.outputs:
            continue
        output_name = op.outputs[0]
        out_value = inferred.get_value(output_name)

        if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
            in_value = inferred.get_value(op.inputs[0])
            in_shape = in_value.shape
            out_features = int(op.attrs["out_features"])
            if in_shape is not None and len(in_shape) > 0:
                out_shape = tuple(in_shape[:-1]) + (out_features,)
            else:
                out_shape = (out_features,)
            out_value.shape = out_shape
            out_value.dtype = in_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.RELU:
            in_value = inferred.get_value(op.inputs[0])
            out_value.shape = in_value.shape
            out_value.dtype = in_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.ADD:
            lhs = inferred.get_value(op.inputs[0])
            rhs = inferred.get_value(op.inputs[1])
            out_value.shape = lhs.shape or rhs.shape or out_value.shape
            out_value.dtype = lhs.dtype or rhs.dtype or out_value.dtype
            continue

        if op.op == OpKind.VIEW:
            in_value = inferred.get_value(op.inputs[0])
            view_shape = _shape_from_view_args(op.attrs.get("args", ()))
            out_value.shape = view_shape or in_value.shape or out_value.shape
            out_value.dtype = in_value.dtype or out_value.dtype
            continue
        if op.op == OpKind.PERMUTE:
            in_value = inferred.get_value(op.inputs[0])
            out_value.shape = _shape_from_permute_args(in_value.shape, op.attrs.get("args", ()))
            out_value.dtype = in_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.SOFTMAX:
            in_value = inferred.get_value(op.inputs[0])
            out_value.shape = in_value.shape
            out_value.dtype = in_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.LAYER_NORM:
            in_value = inferred.get_value(op.inputs[0])
            out_value.shape = in_value.shape
            out_value.dtype = in_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.SCALED_DOT_PRODUCT_ATTENTION:
            q_value = inferred.get_value(op.inputs[0])
            out_value.shape = q_value.shape or out_value.shape
            out_value.dtype = q_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.BATCHED_MATMUL:
            lhs = inferred.get_value(op.inputs[0])
            rhs = inferred.get_value(op.inputs[1])
            if lhs.shape is not None and rhs.shape is not None and len(lhs.shape) >= 2 and len(rhs.shape) >= 2:
                out_value.shape = tuple(lhs.shape[:-1]) + (int(rhs.shape[-1]),)
            else:
                out_value.shape = lhs.shape or rhs.shape or out_value.shape
            out_value.dtype = lhs.dtype or rhs.dtype or out_value.dtype
            continue

        if op.op in {OpKind.SCALE, OpKind.SCALED_SOFTMAX}:
            in_value = inferred.get_value(op.inputs[0])
            out_value.shape = in_value.shape
            out_value.dtype = in_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.CONV2D:
            in_value = inferred.get_value(op.inputs[0])
            in_shape = in_value.shape
            w = op.attrs.get("weight")
            if in_shape is not None and w is not None:
                _, _, h_in, w_in = in_shape
                _, _, kh, kw = np.asarray(w).shape
                stride = op.attrs.get("stride", 1)
                padding = op.attrs.get("padding", 0)
                if isinstance(padding, (tuple, list)):
                    pad_h = int(padding[0]) if len(padding) == 1 else int(padding[0])
                    pad_w = pad_h if len(padding) == 1 else int(padding[1])
                else:
                    pad_h = pad_w = int(padding)
                if isinstance(stride, (tuple, list)):
                    sh = int(stride[0]) if len(stride) == 1 else int(stride[0])
                    sw = sh if len(stride) == 1 else int(stride[1])
                else:
                    sh = sw = int(stride)
                h_out = (int(h_in) + 2 * pad_h - kh) // sh + 1
                w_out = (int(w_in) + 2 * pad_w - kw) // sw + 1
                c_out = int(np.asarray(w).shape[0])
                out_value.shape = tuple(in_shape[:-3]) + (c_out, h_out, w_out)
            out_value.dtype = in_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.MAX_POOL2D:
            in_value = inferred.get_value(op.inputs[0])
            in_shape = in_value.shape
            if in_shape is not None:
                _, _, h_in, w_in = in_shape
                k = op.attrs.get("kernel_size", 1)
                kh, kw = (int(k), int(k)) if not isinstance(k, (tuple, list)) else (int(k[0]), int(k[1]))
                stride = op.attrs.get("stride")
                if stride is None:
                    sh, sw = kh, kw
                elif not isinstance(stride, (tuple, list)):
                    sh = sw = int(stride)
                else:
                    sh = int(stride[0]) if len(stride) == 1 else int(stride[0])
                    sw = sh if len(stride) == 1 else int(stride[1])
                padding = op.attrs.get("padding", 0)
                if not isinstance(padding, (tuple, list)):
                    pad_h = pad_w = int(padding)
                else:
                    pad_h = int(padding[0]) if len(padding) == 1 else int(padding[0])
                    pad_w = pad_h if len(padding) == 1 else int(padding[1])
                h_pad = int(h_in) + 2 * pad_h
                w_pad = int(w_in) + 2 * pad_w
                h_out = (h_pad - kh) // sh + 1
                w_out = (w_pad - kw) // sw + 1
                if bool(op.attrs.get("ceil_mode", False)):
                    if (h_pad - kh) % sh != 0:
                        h_out += 1
                    if (w_pad - kw) % sw != 0:
                        w_out += 1
                out_value.shape = tuple(in_shape[:-2]) + (h_out, w_out)
            out_value.dtype = in_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.ADAPTIVE_AVG_POOL2D:
            in_value = inferred.get_value(op.inputs[0])
            in_shape = in_value.shape
            out_size = op.attrs.get("output_size", 1)
            if in_shape is not None:
                if not isinstance(out_size, (tuple, list)):
                    oh = ow = int(out_size)
                else:
                    oh = int(out_size[0]) if len(out_size) == 1 else int(out_size[0])
                    ow = oh if len(out_size) == 1 else int(out_size[1])
                out_value.shape = tuple(in_shape[:-2]) + (oh, ow)
            out_value.dtype = in_value.dtype or out_value.dtype
            continue

        if op.op == OpKind.BATCH_NORM:
            in_value = inferred.get_value(op.inputs[0])
            out_value.shape = in_value.shape
            out_value.dtype = in_value.dtype or out_value.dtype

    return inferred


def conv_bn_fusion_pass(graph: GraphIR) -> GraphIR:
    """Fold Conv2d + BatchNorm2d into a single Conv2d for inference graphs."""
    op_by_name: Dict[str, OpNode] = {op.name: op for op in graph.ops}
    skip: Set[str] = set()
    fused_ops: List[OpNode] = []

    def _consumers_of(value_name: str) -> List[str]:
        value = graph.values.get(value_name)
        if value is None:
            return []
        return list(value.consumers)

    for op in graph.ops:
        if op.name in skip:
            continue
        if op.op != OpKind.CONV2D:
            fused_ops.append(copy.deepcopy(op))
            continue

        conv_out = op.outputs[0]
        consumers = _consumers_of(conv_out)
        if len(consumers) != 1:
            fused_ops.append(copy.deepcopy(op))
            continue

        bn_op = op_by_name.get(consumers[0])
        if bn_op is None or bn_op.op != OpKind.BATCH_NORM or bn_op.inputs[0] != conv_out:
            fused_ops.append(copy.deepcopy(op))
            continue

        w_fold, b_fold = fold_conv_bn_weights(
            conv_weight=op.attrs["weight"],
            conv_bias=op.attrs.get("bias"),
            bn_weight=bn_op.attrs["weight"],
            bn_bias=bn_op.attrs["bias"],
            bn_running_mean=bn_op.attrs["running_mean"],
            bn_running_var=bn_op.attrs["running_var"],
            bn_eps=float(bn_op.attrs.get("eps", 1e-5)),
        )
        fused = copy.deepcopy(op)
        fused.attrs = dict(op.attrs)
        fused.attrs["weight"] = w_fold
        fused.attrs["bias"] = b_fold
        fused.attrs["bn_fused"] = True
        bn_out = bn_op.outputs[0]
        for downstream in graph.ops:
            if downstream.name in skip:
                continue
            downstream.inputs = [
                fused.outputs[0] if inp == bn_out else inp for inp in downstream.inputs
            ]
        fused_ops.append(fused)
        skip.add(bn_op.name)

    return _rebuild_graph_with_ops(graph, fused_ops)


def attention_decomposition_pass(graph: GraphIR) -> GraphIR:
    current = _clone_graph(graph)
    lowered_ops: List[OpNode] = []
    for op in current.ops:
        if op.op != OpKind.SCALED_DOT_PRODUCT_ATTENTION:
            lowered_ops.append(copy.deepcopy(op))
            continue
        q_name, k_name, v_name = op.inputs[0], op.inputs[1], op.inputs[2]
        mask_name = op.inputs[3] if len(op.inputs) > 3 else None
        k_shape = current.values.get(k_name).shape if k_name in current.values else None
        rank = len(k_shape) if k_shape is not None else 4
        if rank < 2:
            lowered_ops.append(copy.deepcopy(op))
            continue
        dims = list(range(rank))
        dims[-2], dims[-1] = dims[-1], dims[-2]

        k_t = f"{op.name}.k_t"
        scores = f"{op.name}.scores"
        scaled_scores = f"{op.name}.scaled_scores"
        masked_scores = f"{op.name}.masked_scores"
        probs = f"{op.name}.probs"
        out_name = op.outputs[0]

        lowered_ops.append(
            OpNode(
                name=f"{op.name}.permute_k",
                op=OpKind.PERMUTE,
                inputs=[k_name],
                outputs=[k_t],
                attrs={"target": "decompose_attention", "args": tuple(dims)},
            )
        )
        lowered_ops.append(
            OpNode(
                name=f"{op.name}.qk",
                op=OpKind.BATCHED_MATMUL,
                inputs=[q_name, k_t],
                outputs=[scores],
                attrs={"target": "decompose_attention"},
            )
        )
        lowered_ops.append(
            OpNode(
                name=f"{op.name}.scale",
                op=OpKind.SCALE,
                inputs=[scores],
                outputs=[scaled_scores],
                attrs={"target": "decompose_attention", "scale": float(op.attrs.get("scale", 1.0))},
            )
        )
        softmax_inputs = [scaled_scores]
        if mask_name is not None:
            lowered_ops.append(
                OpNode(
                    name=f"{op.name}.mask_add",
                    op=OpKind.ADD,
                    inputs=[scaled_scores, mask_name],
                    outputs=[masked_scores],
                    attrs={"target": "decompose_attention"},
                )
            )
            softmax_inputs = [masked_scores]
        lowered_ops.append(
            OpNode(
                name=f"{op.name}.softmax",
                op=OpKind.SOFTMAX,
                inputs=softmax_inputs,
                outputs=[probs],
                attrs={"target": "decompose_attention", "causal_mask": bool(op.attrs.get("causal_mask", False))},
            )
        )
        lowered_ops.append(
            OpNode(
                name=f"{op.name}.out",
                op=OpKind.BATCHED_MATMUL,
                inputs=[probs, v_name],
                outputs=[out_name],
                attrs={"target": "decompose_attention", "causal_mask": bool(op.attrs.get("causal_mask", False))},
            )
        )

    return _rebuild_graph_with_ops(current, lowered_ops)


def linear_relu_fusion_pass(graph: GraphIR) -> GraphIR:
    current = _clone_graph(graph)
    op_by_name = {op.name: op for op in current.ops}
    skip_ops: Set[str] = set()
    fused_ops: List[OpNode] = []

    for op in current.ops:
        if op.name in skip_ops:
            continue
        if op.op != OpKind.LINEAR:
            fused_ops.append(copy.deepcopy(op))
            continue

        out_name = op.outputs[0] if op.outputs else None
        out_value = current.values.get(out_name) if out_name is not None else None
        if out_value is None or len(out_value.consumers) != 1:
            fused_ops.append(copy.deepcopy(op))
            continue

        relu_name = out_value.consumers[0]
        relu = op_by_name.get(relu_name)
        if relu is None or relu.op != OpKind.RELU:
            fused_ops.append(copy.deepcopy(op))
            continue

        attrs = dict(op.attrs)
        attrs["fused_activation"] = "relu"
        fused_ops.append(
            OpNode(
                name=f"{op.name}_relu_fused",
                op=OpKind.LINEAR_RELU,
                inputs=list(op.inputs),
                outputs=list(relu.outputs),
                attrs=attrs,
            )
        )
        skip_ops.add(relu.name)

    return _rebuild_graph_with_ops(current, fused_ops)


def scale_softmax_fusion_pass(graph: GraphIR) -> GraphIR:
    current = _clone_graph(graph)
    op_by_name = {op.name: op for op in current.ops}
    skip_ops: Set[str] = set()
    fused_ops: List[OpNode] = []
    for op in current.ops:
        if op.name in skip_ops:
            continue
        if op.op != OpKind.SCALE:
            fused_ops.append(copy.deepcopy(op))
            continue
        out_name = op.outputs[0] if op.outputs else None
        out_value = current.values.get(out_name) if out_name is not None else None
        if out_value is None or len(out_value.consumers) != 1:
            fused_ops.append(copy.deepcopy(op))
            continue
        consumer = op_by_name.get(out_value.consumers[0])
        if consumer is None or consumer.op != OpKind.SOFTMAX:
            fused_ops.append(copy.deepcopy(op))
            continue
        fused_ops.append(
            OpNode(
                name=f"{op.name}_softmax_fused",
                op=OpKind.SCALED_SOFTMAX,
                inputs=list(op.inputs),
                outputs=list(consumer.outputs),
                attrs={
                    "scale": float(op.attrs.get("scale", 1.0)),
                    "causal_mask": bool(consumer.attrs.get("causal_mask", False)),
                },
            )
        )
        skip_ops.add(consumer.name)
    return _rebuild_graph_with_ops(current, fused_ops)


def dead_code_elimination_pass(graph: GraphIR) -> GraphIR:
    current = _clone_graph(graph)
    live_values: Set[str] = set(current.outputs)
    kept_reversed: List[OpNode] = []

    for op in reversed(current.ops):
        if any(out in live_values for out in op.outputs):
            kept_reversed.append(copy.deepcopy(op))
            for input_name in op.inputs:
                live_values.add(input_name)

    kept_ops = list(reversed(kept_reversed))
    return _rebuild_graph_with_ops(current, kept_ops)


def backend_legality_pass(graph: GraphIR, backend: str) -> GraphIR:
    target = (backend or "utpu").strip().lower()
    allowed = supported_ops_for_backend(target)

    lowered = _clone_graph(graph)
    lowered.metadata.setdefault("backend_legality", {})
    lowered.metadata["backend_legality"]["backend"] = target
    lowered.metadata["backend_legality"]["ops"] = []

    offending = []
    for op in lowered.ops:
        lowering_available = bool(op.op in allowed)
        lowered.metadata["backend_legality"]["ops"].append(
            {
                "name": op.name,
                "op": op.op,
                "lowering_available": lowering_available,
            }
        )
        op.attrs[f"{target}_lowering_available"] = lowering_available
        if not lowering_available:
            offending.append(
                {
                    "name": op.name,
                    "op": op.op,
                    "reason": f"unsupported_for_backend:{target}",
                }
            )

    if offending:
        raise BackendLegalityError(target, offending)
    return lowered


def _dtype_size_bytes(dtype: Optional[str]) -> int:
    if dtype is None:
        return 4
    name = str(dtype).lower()
    if "float64" in name or "int64" in name:
        return 8
    if "float16" in name or "bfloat16" in name or "int16" in name:
        return 2
    if "int8" in name or "uint8" in name or "bool" in name:
        return 1
    return 4


def _shape_nbytes(shape: Optional[Tuple[int, ...]], dtype: Optional[str]) -> int:
    if not shape:
        return 0
    count = 1
    for dim in shape:
        d = int(dim)
        if d < 0:
            return 0
        count *= d
    return int(count * _dtype_size_bytes(dtype))


def memory_planning_pass(graph: GraphIR) -> GraphIR:
    planned = _clone_graph(graph)
    op_index = {op.name: idx for idx, op in enumerate(planned.ops)}
    graph_end = len(planned.ops)
    logical_values: List[Dict[str, Any]] = []
    persistent_values: List[Dict[str, Any]] = []

    for name, value in planned.values.items():
        is_persistent = bool(getattr(value, "persistent", False))
        if name in planned.inputs:
            continue
        if value.producer is None and not is_persistent:
            continue
        first_def = int(op_index.get(value.producer, 0))
        consumer_indices = [op_index[c] for c in value.consumers if c in op_index]
        last_use = max(consumer_indices) if consumer_indices else first_def
        if name in planned.outputs:
            last_use = max(last_use, graph_end)
        size_bytes = _shape_nbytes(value.shape, value.dtype)
        record = {
            "value": name,
            "producer": value.producer,
            "consumers": list(value.consumers),
            "shape": list(value.shape) if value.shape is not None else None,
            "dtype": value.dtype,
            "size_bytes": int(size_bytes),
            "first_def": int(first_def),
            "last_use": int(last_use),
            "kind": "output" if name in planned.outputs else "intermediate",
            "persistent": is_persistent,
        }
        if record["persistent"]:
            persistent_values.append(record)
        else:
            logical_values.append(record)

    buffers: List[Dict[str, Any]] = []
    for logical in sorted(logical_values, key=lambda item: (item["first_def"], -item["size_bytes"], item["value"])):
        reusable = [
            buf for buf in buffers
            if int(buf["last_use"]) < int(logical["first_def"]) and int(buf["size_bytes"]) >= int(logical["size_bytes"])
        ]
        if reusable:
            chosen = min(reusable, key=lambda buf: (int(buf["size_bytes"]), str(buf["buffer"])))
        else:
            chosen = {
                "buffer": f"act_{len(buffers)}",
                "size_bytes": int(logical["size_bytes"]),
                "values": [],
                "first_def": int(logical["first_def"]),
                "last_use": int(logical["last_use"]),
            }
            buffers.append(chosen)

        chosen["size_bytes"] = max(int(chosen["size_bytes"]), int(logical["size_bytes"]))
        chosen["first_def"] = min(int(chosen["first_def"]), int(logical["first_def"]))
        chosen["last_use"] = max(int(chosen["last_use"]), int(logical["last_use"]))
        chosen["values"].append(logical["value"])
        logical["buffer"] = chosen["buffer"]

    kv_buffers: List[Dict[str, Any]] = []
    kv_cache_layout: Dict[str, Any] = {"layers": [], "total_kv_bytes": 0}
    persistent_offset = 0
    layer_map: Dict[int, Dict[str, Any]] = {}
    for item in sorted(persistent_values, key=lambda v: v["value"]):
        size = int(item["size_bytes"])
        kv_buffers.append(
            {
                "buffer": f"kv_{len(kv_buffers)}",
                "value": item["value"],
                "size_bytes": size,
                "offset_bytes": persistent_offset,
            }
        )
        item["buffer"] = kv_buffers[-1]["buffer"]
        item["offset_bytes"] = persistent_offset
        persistent_offset += size

        lname = item["value"].lower()
        layer_id = 0
        if "layer" in lname:
            suffix = lname.split("layer", 1)[1]
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if digits:
                layer_id = int(digits)
        entry = layer_map.setdefault(layer_id, {"layer": layer_id})
        if "k" in lname:
            entry["k"] = {
                "value": item["value"],
                "shape": item["shape"],
                "size_bytes": size,
                "offset_bytes": item["offset_bytes"],
                "stride_last_dim": item["shape"][-1] if item["shape"] else 0,
            }
        if "v" in lname:
            entry["v"] = {
                "value": item["value"],
                "shape": item["shape"],
                "size_bytes": size,
                "offset_bytes": item["offset_bytes"],
                "stride_last_dim": item["shape"][-1] if item["shape"] else 0,
            }

    kv_cache_layout["layers"] = [layer_map[k] for k in sorted(layer_map.keys())]
    kv_cache_layout["total_kv_bytes"] = int(sum(v["size_bytes"] for v in persistent_values))

    naive_persistent_bytes = int(sum(item["size_bytes"] for item in logical_values))
    planned_bytes = int(sum(buf["size_bytes"] for buf in buffers))
    reduction_pct = 0.0
    if naive_persistent_bytes > 0:
        reduction_pct = (1.0 - (float(planned_bytes) / float(naive_persistent_bytes))) * 100.0

    planned.metadata["memory_plan"] = {
        "method": "liveness_greedy_first_fit",
        "logical_value_count": int(len(logical_values)),
        "physical_buffer_count": int(len(buffers)),
        "naive_persistent_bytes": naive_persistent_bytes,
        "planned_peak_bytes": planned_bytes,
        "peak_memory_reduction_pct": float(reduction_pct),
        "values": logical_values + persistent_values,
        "buffers": buffers,
        "persistent_buffers": kv_buffers,
        "kv_cache_layout": kv_cache_layout,
        "activation_bytes_with_reuse": planned_bytes,
    }
    return planned


class GraphPassManager:
    def __init__(self, target_backend: str):
        self.target_backend = target_backend

    def run(self, graph: GraphIR) -> PassPipelineResult:
        current = _clone_graph(graph)
        records: List[PassRecord] = []
        passes = [
            ("shape_inference", shape_inference_pass),
            ("conv_bn_fusion", conv_bn_fusion_pass),
            ("shape_inference_post_fold", shape_inference_pass),
            ("attention_decomposition", attention_decomposition_pass),
            ("linear_relu_fusion", linear_relu_fusion_pass),
            ("scale_softmax_fusion", scale_softmax_fusion_pass),
            ("dead_code_elimination", dead_code_elimination_pass),
            ("memory_planning", memory_planning_pass),
            ("backend_legality", lambda g: backend_legality_pass(g, backend=self.target_backend)),
        ]

        for pass_name, fn in passes:
            before = _graph_to_dict(current)
            current = fn(current)
            after = _graph_to_dict(current)
            records.append(PassRecord(pass_name=pass_name, before=before, after=after))
        return PassPipelineResult(graph=current, records=records)


def write_pass_pipeline_dump(result: PassPipelineResult, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)
