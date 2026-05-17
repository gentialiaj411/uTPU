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

            raise GraphReferenceInterpreterError(f"Unsupported op '{op.op}' in reference interpreter")

        outputs = [values[name] for name in self.graph.outputs]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def execute_graph_reference(graph: GraphIR, *inputs: Any) -> Any:
    return GraphReferenceInterpreter(graph).run(*inputs)
