import operator
from builtins import getattr as builtin_getattr
from typing import Any, Dict, Optional, Tuple

from graph_ir import GraphIR, OpKind, OpNode


class FXImportError(ValueError):
    pass


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        raise FXImportError(
            "PyTorch is required to import torch.fx graphs, but it is not available"
        ) from e
    return torch, nn, F


def _tensor_meta(node: Any) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    meta = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
    if meta is None:
        return None, None
    shape = tuple(int(d) for d in meta.shape) if getattr(meta, "shape", None) is not None else None
    dtype = str(meta.dtype) if getattr(meta, "dtype", None) is not None else None
    return shape, dtype


def _node_name(arg: Any) -> str:
    if hasattr(arg, "name"):
        return arg.name
    raise FXImportError(f"Expected FX node argument, got {arg!r}")


def _as_int(value: Any, context: str) -> int:
    try:
        return int(value)
    except Exception as e:
        raise FXImportError(f"Expected integer-like value for {context}, got {value!r}") from e


def _resolve_shape_meta_arg(arg: Any, node_to_value: Dict[Any, str], meta_values: Dict[Any, Any]) -> Any:
    if hasattr(arg, "op"):
        if arg in meta_values:
            return meta_values[arg]
        if arg in node_to_value:
            return _node_name(arg)
        raise FXImportError(f"Unsupported unresolved FX argument node '{arg.name}'")
    if isinstance(arg, list):
        return [_resolve_shape_meta_arg(v, node_to_value, meta_values) for v in arg]
    if isinstance(arg, tuple):
        return tuple(_resolve_shape_meta_arg(v, node_to_value, meta_values) for v in arg)
    return arg


def _normalize_view_args(node: Any, shape: Optional[Tuple[int, ...]], node_to_value: Dict[Any, str], meta_values: Dict[Any, Any]) -> Tuple[Any, ...]:
    if shape is not None:
        return tuple(int(d) for d in shape)

    resolved = [_resolve_shape_meta_arg(a, node_to_value, meta_values) for a in node.args[1:]]
    if len(resolved) == 1 and isinstance(resolved[0], (tuple, list)):
        resolved = list(resolved[0])
    return tuple(resolved)


def _single_node_arg(node: Any) -> str:
    if not node.args:
        raise FXImportError(f"Node '{node.name}' is missing its input argument")
    return _node_name(node.args[0])


def _linear_attrs(module: Any) -> Dict[str, Any]:
    weight = module.weight.detach().cpu().numpy()
    bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
    return {
        "module": module,
        "weight": weight,
        "bias": bias,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
    }


def _is_relu_function(target: Any, torch: Any, F: Any) -> bool:
    return target in {torch.relu, F.relu}


def _is_add_function(target: Any, torch: Any) -> bool:
    return target in {operator.add, torch.add}


def _is_view_method(target: Any) -> bool:
    return target in {"view", "reshape", "flatten"}


def _is_permute_method(target: Any) -> bool:
    return target in {"permute", "transpose"}


def _is_shape_plumbing_view_method(target: Any) -> bool:
    return target in {"unsqueeze", "squeeze", "contiguous"}


def _is_softmax_function(target: Any, torch: Any, F: Any) -> bool:
    return target in {torch.softmax, F.softmax}


def import_fx_graph_module(fx_module: Any, name: Optional[str] = None) -> GraphIR:
    torch, nn, F = _require_torch()
    modules = dict(fx_module.named_modules())
    graph = GraphIR(name=name or fx_module.__class__.__name__)
    node_to_value: Dict[Any, str] = {}
    meta_values: Dict[Any, Any] = {}

    for node in fx_module.graph.nodes:
        shape, dtype = _tensor_meta(node)

        if node.op == "placeholder":
            graph.inputs.append(node.name)
            graph.add_value(node.name, shape=shape, dtype=dtype)
            node_to_value[node] = node.name
            continue

        if node.op == "output":
            output_arg = node.args[0]
            outputs = output_arg if isinstance(output_arg, (tuple, list)) else (output_arg,)
            for out in outputs:
                graph.outputs.append(_node_name(out))
            continue

        if node.op == "call_module":
            module = modules.get(node.target)
            if isinstance(module, nn.Linear):
                input_name = _single_node_arg(node)
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.LINEAR,
                        inputs=[input_name],
                        outputs=[node.name],
                        attrs={**_linear_attrs(module), "target": str(node.target)},
                    )
                )
                node_to_value[node] = node.name
                continue
            if isinstance(module, nn.ReLU):
                input_name = _single_node_arg(node)
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.RELU,
                        inputs=[input_name],
                        outputs=[node.name],
                        attrs={"target": str(node.target), "inplace": bool(module.inplace)},
                    )
                )
                node_to_value[node] = node.name
                continue
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                input_name = _single_node_arg(node)
                weight = module.weight.detach().cpu().numpy() if getattr(module, "weight", None) is not None else None
                bias = module.bias.detach().cpu().numpy() if getattr(module, "bias", None) is not None else None
                norm_kind = "rms_norm" if isinstance(module, nn.RMSNorm) else "layer_norm"
                normalized_shape = tuple(int(d) for d in module.normalized_shape) if getattr(module, "normalized_shape", None) is not None else None
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.LAYER_NORM,
                        inputs=[input_name],
                        outputs=[node.name],
                        attrs={
                            "target": str(node.target),
                            "norm_kind": norm_kind,
                            "normalized_shape": normalized_shape,
                            "eps": float(module.eps) if module.eps is not None else 1e-5,
                            "weight": weight,
                            "bias": bias,
                        },
                    )
                )
                node_to_value[node] = node.name
                continue
            raise FXImportError(
                f"Unsupported call_module node '{node.name}' target '{node.target}' "
                f"of type {type(module).__name__}"
            )

        if node.op == "call_function":
            if node.target == operator.getitem:
                if len(node.args) < 2:
                    raise FXImportError(f"getitem node '{node.name}' needs base and index")
                base = _resolve_shape_meta_arg(node.args[0], node_to_value, meta_values)
                index = _resolve_shape_meta_arg(node.args[1], node_to_value, meta_values)
                try:
                    meta_values[node] = base[index]
                except Exception as e:
                    raise FXImportError(f"Failed to resolve getitem for node '{node.name}'") from e
                continue
            if node.target == builtin_getattr:
                if len(node.args) < 2:
                    raise FXImportError(f"getattr node '{node.name}' needs object and attr name")
                base_arg = node.args[0]
                attr_name = _resolve_shape_meta_arg(node.args[1], node_to_value, meta_values)
                if not isinstance(attr_name, str):
                    raise FXImportError(
                        f"getattr node '{node.name}' attr must resolve to string, got {attr_name!r}"
                    )
                if hasattr(base_arg, "op") and base_arg in node_to_value and attr_name == "shape":
                    value_name = node_to_value[base_arg]
                    tensor_shape = graph.values.get(value_name).shape if value_name in graph.values else None
                    if tensor_shape is None:
                        raise FXImportError(f"getattr(shape) node '{node.name}' requires known input shape")
                    meta_values[node] = tensor_shape
                else:
                    base = _resolve_shape_meta_arg(base_arg, node_to_value, meta_values)
                    try:
                        meta_values[node] = getattr(base, attr_name)
                    except Exception as e:
                        raise FXImportError(f"Failed to resolve getattr for node '{node.name}'") from e
                continue
            if _is_relu_function(node.target, torch, F):
                input_name = _single_node_arg(node)
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.RELU,
                        inputs=[input_name],
                        outputs=[node.name],
                        attrs={"target": str(node.target)},
                    )
                )
                node_to_value[node] = node.name
                continue
            if _is_add_function(node.target, torch):
                if len(node.args) < 2:
                    raise FXImportError(f"Add node '{node.name}' needs two input arguments")
                lhs = _node_name(node.args[0])
                rhs = _node_name(node.args[1])
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.ADD,
                        inputs=[lhs, rhs],
                        outputs=[node.name],
                        attrs={"target": str(node.target)},
                    )
                )
                node_to_value[node] = node.name
                continue
            if _is_softmax_function(node.target, torch, F):
                input_name = _single_node_arg(node)
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.SOFTMAX,
                        inputs=[input_name],
                        outputs=[node.name],
                        attrs={"target": str(node.target)},
                    )
                )
                node_to_value[node] = node.name
                continue
            if node.target == F.scaled_dot_product_attention:
                if len(node.args) < 3:
                    raise FXImportError(f"Attention node '{node.name}' requires q, k, v")
                q = _node_name(node.args[0])
                k = _node_name(node.args[1])
                v = _node_name(node.args[2])
                inputs = [q, k, v]
                if len(node.args) > 3 and hasattr(node.args[3], "name"):
                    inputs.append(_node_name(node.args[3]))
                else:
                    attn_mask_kw = node.kwargs.get("attn_mask")
                    if hasattr(attn_mask_kw, "name"):
                        inputs.append(_node_name(attn_mask_kw))
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.SCALED_DOT_PRODUCT_ATTENTION,
                        inputs=inputs,
                        outputs=[node.name],
                        attrs={
                            "causal_mask": bool(node.kwargs.get("is_causal", False)),
                            "scale": float(node.kwargs.get("scale")) if node.kwargs.get("scale") is not None else 1.0,
                        },
                    )
                )
                node_to_value[node] = node.name
                continue
            raise FXImportError(
                f"Unsupported call_function node '{node.name}' target '{node.target}'"
            )

        if node.op == "call_method":
            if node.target == "size":
                input_name = _single_node_arg(node)
                in_shape = graph.values.get(input_name).shape if input_name in graph.values else None
                if in_shape is None:
                    raise FXImportError(f"size node '{node.name}' requires known input shape")
                if len(node.args) > 1:
                    dim = _as_int(_resolve_shape_meta_arg(node.args[1], node_to_value, meta_values), f"{node.name}.dim")
                    meta_values[node] = int(in_shape[dim])
                else:
                    meta_values[node] = tuple(int(d) for d in in_shape)
                continue
            if _is_view_method(node.target):
                input_name = _single_node_arg(node)
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.VIEW,
                        inputs=[input_name],
                        outputs=[node.name],
                        attrs={
                            "target": str(node.target),
                            "args": _normalize_view_args(node, shape, node_to_value, meta_values),
                        },
                    )
                )
                node_to_value[node] = node.name
                continue
            if _is_shape_plumbing_view_method(node.target):
                input_name = _single_node_arg(node)
                if shape is None:
                    raise FXImportError(
                        f"{node.target} node '{node.name}' requires known output shape metadata"
                    )
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.VIEW,
                        inputs=[input_name],
                        outputs=[node.name],
                        attrs={
                            "target": str(node.target),
                            "args": tuple(int(d) for d in shape),
                        },
                    )
                )
                node_to_value[node] = node.name
                continue
            if _is_permute_method(node.target):
                input_name = _single_node_arg(node)
                graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
                if node.target == "transpose":
                    if len(node.args) < 3:
                        raise FXImportError(f"transpose node '{node.name}' missing dims")
                    dim0 = int(node.args[1])
                    dim1 = int(node.args[2])
                    rank = len(shape) if shape is not None else max(dim0, dim1) + 1
                    dims = list(range(rank))
                    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
                    args = tuple(dims)
                else:
                    args = tuple(node.args[1:]) if node.args[1:] else tuple(node.args)
                    if args and hasattr(args[0], "name"):
                        args = tuple(node.args[1:])
                graph.add_op(
                    OpNode(
                        name=node.name,
                        op=OpKind.PERMUTE,
                        inputs=[input_name],
                        outputs=[node.name],
                        attrs={"target": str(node.target), "args": args},
                    )
                )
                node_to_value[node] = node.name
                continue

        raise FXImportError(
            f"Unsupported FX node '{node.name}' with op '{node.op}' and target '{node.target}'"
        )

    return graph
