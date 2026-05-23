from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from graph_ir import GraphIR, OpKind


class GraphReferenceInterpreterError(ValueError):
    pass


def _as_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _resolve_view_shape(raw_shape: Tuple[int, ...], size: int) -> Tuple[int, ...]:
    shape = list(int(v) for v in raw_shape)
    unknown = [i for i, d in enumerate(shape) if d == -1]
    if len(unknown) > 1:
        raise GraphReferenceInterpreterError(f"Invalid view shape with multiple -1 dims: {raw_shape}")
    if len(unknown) == 1:
        known_prod = 1
        for d in shape:
            if d != -1:
                known_prod *= d
        if known_prod == 0 or size % known_prod != 0:
            raise GraphReferenceInterpreterError(f"Cannot infer view shape {raw_shape} for size={size}")
        shape[unknown[0]] = size // known_prod
    return tuple(shape)


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / np.maximum(denom, 1e-12)


def _rms_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + float(eps))
    return x / rms


def _unpack_int4(packed: np.ndarray, size: int) -> np.ndarray:
    src = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.zeros((src.size * 2,), dtype=np.int8)
    out[0::2] = (src & 0x0F).astype(np.int8) - 8
    out[1::2] = ((src >> 4) & 0x0F).astype(np.int8) - 8
    return out[:size]


@dataclass
class GraphReferenceInterpreter:
    graph: GraphIR

    def run(self, *inputs: Any) -> Any:
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = tuple(inputs[0])
        if len(inputs) != len(self.graph.inputs):
            raise GraphReferenceInterpreterError(
                f"Expected {len(self.graph.inputs)} inputs, got {len(inputs)}"
            )

        values: Dict[str, np.ndarray] = {}
        for name, value in zip(self.graph.inputs, inputs):
            values[name] = _as_float32(value)

        for op in self.graph.ops:
            if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
                x = _as_float32(values[op.inputs[0]])
                if op.attrs.get("dtype_quant") == "int4_g64":
                    shape = tuple(op.attrs["weight_int4_shape"])
                    packed = np.asarray(op.attrs["weight_int4_packed"], dtype=np.uint8)
                    q = _unpack_int4(packed, int(shape[0] * shape[1])).reshape(shape).astype(np.float32)
                    scales = _as_float32(op.attrs["weight_int4_scales"])
                    w = np.zeros(shape, dtype=np.float32)
                    group_size = 64
                    for o in range(shape[0]):
                        for g in range(scales.shape[1]):
                            s = g * group_size
                            e = min(shape[1], s + group_size)
                            w[o, s:e] = q[o, s:e] * scales[o, g]
                else:
                    w = _as_float32(op.attrs["weight"])
                b = op.attrs.get("bias")
                b_arr = _as_float32(b) if b is not None else None

                if x.shape[-1] != int(op.attrs["in_features"]):
                    raise GraphReferenceInterpreterError(
                        f"Linear op '{op.name}' expected input last dim {op.attrs['in_features']}, got {x.shape}"
                    )
                y = np.matmul(x.astype(np.float32, copy=False), w.T.astype(np.float32, copy=False))
                if b_arr is not None:
                    y = y + b_arr
                if op.op == OpKind.LINEAR_RELU:
                    y = np.maximum(y, 0.0)
                values[op.outputs[0]] = y.astype(np.float32, copy=False)
                continue

            if op.op == OpKind.RELU:
                x = _as_float32(values[op.inputs[0]])
                values[op.outputs[0]] = np.maximum(x, 0.0).astype(np.float32, copy=False)
                continue

            if op.op == OpKind.ADD:
                lhs = _as_float32(values[op.inputs[0]])
                rhs = _as_float32(values[op.inputs[1]])
                values[op.outputs[0]] = (lhs + rhs).astype(np.float32, copy=False)
                continue

            if op.op == OpKind.VIEW:
                x = _as_float32(values[op.inputs[0]])
                raw = tuple(op.attrs.get("args", ()))
                if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                    raw = tuple(raw[0])
                if not raw:
                    raise GraphReferenceInterpreterError(f"View op '{op.name}' has no target shape args")
                shape = _resolve_view_shape(raw, int(x.size))
                values[op.outputs[0]] = np.reshape(x, shape).astype(np.float32, copy=False)
                continue
            if op.op == OpKind.PERMUTE:
                x = _as_float32(values[op.inputs[0]])
                raw = tuple(op.attrs.get("args", ()))
                if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                    raw = tuple(raw[0])
                values[op.outputs[0]] = np.transpose(x, axes=raw).astype(np.float32, copy=False)
                continue

            if op.op == OpKind.SOFTMAX:
                x = _as_float32(values[op.inputs[0]])
                if bool(op.attrs.get("causal_mask", False)):
                    tri = np.triu(np.ones((x.shape[-2], x.shape[-1]), dtype=bool), k=1)
                    x = np.where(tri, -1e9, x)
                values[op.outputs[0]] = _stable_softmax(x, axis=-1).astype(np.float32, copy=False)
                continue

            if op.op == OpKind.LAYER_NORM:
                x = _as_float32(values[op.inputs[0]])
                eps = float(op.attrs.get("eps", 1e-5))
                norm_kind = str(op.attrs.get("norm_kind", "rms_norm"))
                if norm_kind == "layer_norm":
                    mean = np.mean(x, axis=-1, keepdims=True)
                    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
                    y = (x - mean) / np.sqrt(var + eps)
                else:
                    y = _rms_norm(x, eps=eps)
                if op.attrs.get("weight") is not None:
                    y = y * _as_float32(op.attrs["weight"])
                if op.attrs.get("bias") is not None:
                    y = y + _as_float32(op.attrs["bias"])
                values[op.outputs[0]] = y.astype(np.float32, copy=False)
                continue

            if op.op == OpKind.SCALED_DOT_PRODUCT_ATTENTION:
                q = _as_float32(values[op.inputs[0]])
                k = _as_float32(values[op.inputs[1]])
                v = _as_float32(values[op.inputs[2]])
                mask = _as_float32(values[op.inputs[3]]) if len(op.inputs) > 3 else None

                if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
                    raise GraphReferenceInterpreterError(
                        f"Attention op '{op.name}' expects rank-4 [B,H,T,D] tensors, got "
                        f"{q.shape}, {k.shape}, {v.shape}"
                    )
                head_dim = int(op.attrs.get("head_dim", q.shape[-1]))
                scale = 1.0 / np.sqrt(float(head_dim))
                scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale

                causal = bool(op.attrs.get("causal_mask", False))
                if causal:
                    tq, tk = scores.shape[-2], scores.shape[-1]
                    tri = np.triu(np.ones((tq, tk), dtype=bool), k=1)
                    scores = np.where(tri[None, None, :, :], -1e9, scores)

                if mask is not None:
                    scores = scores + mask

                probs = _stable_softmax(scores, axis=-1).astype(np.float32, copy=False)
                out = np.matmul(probs, v)
                values[op.outputs[0]] = out.astype(np.float32, copy=False)
                continue

            if op.op == OpKind.BATCHED_MATMUL:
                lhs = _as_float32(values[op.inputs[0]])
                rhs = _as_float32(values[op.inputs[1]])
                values[op.outputs[0]] = np.matmul(lhs, rhs).astype(np.float32, copy=False)
                continue

            if op.op == OpKind.SCALE:
                x = _as_float32(values[op.inputs[0]])
                s = float(op.attrs.get("scale", 1.0))
                values[op.outputs[0]] = (x * s).astype(np.float32, copy=False)
                continue

            if op.op == OpKind.SCALED_SOFTMAX:
                x = _as_float32(values[op.inputs[0]])
                if len(op.inputs) > 1:
                    x = x + _as_float32(values[op.inputs[1]])
                s = float(op.attrs.get("scale", 1.0))
                if bool(op.attrs.get("causal_mask", False)):
                    tri = np.triu(np.ones((x.shape[-2], x.shape[-1]), dtype=bool), k=1)
                    x = np.where(tri, -1e9, x)
                values[op.outputs[0]] = _stable_softmax(x * s, axis=-1).astype(np.float32, copy=False)
                continue

            raise GraphReferenceInterpreterError(f"Unsupported op '{op.op}' in reference interpreter")

        outputs = [values[name] for name in self.graph.outputs]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def execute_graph_reference(graph: GraphIR, *inputs: Any) -> Any:
    return GraphReferenceInterpreter(graph).run(*inputs)
