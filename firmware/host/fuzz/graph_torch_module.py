"""Build a traceable ``torch.nn.Module`` from a fuzzer ``GraphIR``.

Used by the optional TorchInductor differential oracle: ``torch.compile(...,
backend="inductor", fullgraph=True)`` needs an eager module, while the fuzzer
only materializes ``GraphIR`` + NumPy inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from graph_ir import GraphIR, OpKind

# Ops the fuzzer generator may emit and Inductor can lower on CUDA/CPU.
_INDUCTOR_SUPPORTED_OPS: Set[str] = frozenset(
    {
        OpKind.LINEAR,
        OpKind.LINEAR_RELU,
        OpKind.RELU,
        OpKind.ADD,
        OpKind.SCALE,
        OpKind.VIEW,
        OpKind.PERMUTE,
        OpKind.SOFTMAX,
        OpKind.LAYER_NORM,
        OpKind.BATCHED_MATMUL,
    }
)


def is_graph_inductor_compatible(graph: GraphIR) -> bool:
    """True when every op in ``graph`` is supported by :func:`build_torch_module_from_graph`."""
    for op in graph.ops:
        if op.op not in _INDUCTOR_SUPPORTED_OPS:
            return False
        if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU} and op.attrs.get("dtype_quant") == "int4_g64":
            return False
    return True


def build_torch_module_from_graph(graph: GraphIR):
    """Return an ``nn.Module`` whose ``forward(*graph.inputs)`` mirrors GraphIR semantics."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not is_graph_inductor_compatible(graph):
        raise ValueError(
            f"graph {graph.name!r} contains ops not supported for TorchInductor oracle"
        )

    class _GraphIRModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._graph = graph
            self._input_names: Tuple[str, ...] = tuple(graph.inputs)
            self._output_names: Tuple[str, ...] = tuple(graph.outputs)
            for idx, op in enumerate(graph.ops):
                if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
                    w = np.asarray(op.attrs["weight"], dtype=np.float32)
                    self.register_buffer(f"_w_{idx}", torch.from_numpy(w))
                    bias = op.attrs.get("bias")
                    if bias is not None:
                        b = np.asarray(bias, dtype=np.float32)
                        self.register_buffer(f"_b_{idx}", torch.from_numpy(b))
                elif op.op == OpKind.SCALE:
                    self.register_buffer(
                        f"_scale_{idx}",
                        torch.tensor(float(op.attrs.get("scale", 1.0)), dtype=torch.float32),
                    )
                elif op.op == OpKind.LAYER_NORM:
                    w = op.attrs.get("weight")
                    if w is not None:
                        self.register_buffer(
                            f"_ln_w_{idx}", torch.from_numpy(np.asarray(w, dtype=np.float32))
                        )
                    b = op.attrs.get("bias")
                    if b is not None:
                        self.register_buffer(
                            f"_ln_b_{idx}", torch.from_numpy(np.asarray(b, dtype=np.float32))
                        )

        def forward(self, *args: torch.Tensor) -> Any:
            if len(args) != len(self._input_names):
                raise ValueError(
                    f"expected {len(self._input_names)} inputs, got {len(args)}"
                )
            values: Dict[str, torch.Tensor] = {
                name: arg.to(dtype=torch.float32) for name, arg in zip(self._input_names, args)
            }
            for idx, op in enumerate(self._graph.ops):
                if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
                    x = values[op.inputs[0]]
                    w = getattr(self, f"_w_{idx}")
                    b = getattr(self, f"_b_{idx}", None)
                    y = F.linear(x, w, b)
                    if op.op == OpKind.LINEAR_RELU:
                        y = torch.relu(y)
                    values[op.outputs[0]] = y
                    continue
                if op.op == OpKind.RELU:
                    values[op.outputs[0]] = torch.relu(values[op.inputs[0]])
                    continue
                if op.op == OpKind.ADD:
                    values[op.outputs[0]] = values[op.inputs[0]] + values[op.inputs[1]]
                    continue
                if op.op == OpKind.SCALE:
                    values[op.outputs[0]] = values[op.inputs[0]] * getattr(self, f"_scale_{idx}")
                    continue
                if op.op == OpKind.VIEW:
                    x = values[op.inputs[0]]
                    raw = tuple(op.attrs.get("args", ()))
                    if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                        raw = tuple(raw[0])
                    values[op.outputs[0]] = x.reshape(raw)
                    continue
                if op.op == OpKind.PERMUTE:
                    x = values[op.inputs[0]]
                    raw = tuple(op.attrs.get("args", ()))
                    if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                        raw = tuple(raw[0])
                    values[op.outputs[0]] = x.permute(raw)
                    continue
                if op.op == OpKind.SOFTMAX:
                    x = values[op.inputs[0]]
                    if bool(op.attrs.get("causal_mask", False)):
                        tq, tk = x.shape[-2], x.shape[-1]
                        mask = torch.triu(
                            torch.ones((tq, tk), device=x.device, dtype=torch.bool), diagonal=1
                        )
                        x = x.masked_fill(mask, -1e9)
                    values[op.outputs[0]] = torch.softmax(x, dim=-1)
                    continue
                if op.op == OpKind.LAYER_NORM:
                    x = values[op.inputs[0]]
                    eps = float(op.attrs.get("eps", 1e-5))
                    norm_kind = str(op.attrs.get("norm_kind", "rms_norm"))
                    if norm_kind == "layer_norm":
                        mean = x.mean(dim=-1, keepdim=True)
                        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
                        y = (x - mean) / torch.sqrt(var + eps)
                    else:
                        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
                        y = x / rms
                    ln_w = getattr(self, f"_ln_w_{idx}", None)
                    if ln_w is not None:
                        y = y * ln_w
                    ln_b = getattr(self, f"_ln_b_{idx}", None)
                    if ln_b is not None:
                        y = y + ln_b
                    values[op.outputs[0]] = y
                    continue
                if op.op == OpKind.BATCHED_MATMUL:
                    values[op.outputs[0]] = torch.matmul(
                        values[op.inputs[0]], values[op.inputs[1]]
                    )
                    continue
                raise RuntimeError(f"unsupported op {op.op!r}")

            outs = [values[name] for name in self._output_names]
            if len(outs) == 1:
                return outs[0]
            return tuple(outs)

    return _GraphIRModule()
