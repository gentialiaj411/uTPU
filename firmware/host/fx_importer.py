import operator
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


def import_fx_graph_module(fx_module: Any, name: Optional[str] = None) -> GraphIR:
    torch, nn, F = _require_torch()
    modules = dict(fx_module.named_modules())
    graph = GraphIR(name=name or fx_module.__class__.__name__)
    node_to_value: Dict[Any, str] = {}

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
            raise FXImportError(
                f"Unsupported call_module node '{node.name}' target '{node.target}' "
                f"of type {type(module).__name__}"
            )

        if node.op == "call_function":
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
            raise FXImportError(
                f"Unsupported call_function node '{node.name}' target '{node.target}'"
            )

        if node.op == "call_method" and _is_view_method(node.target):
            input_name = _single_node_arg(node)
            graph.add_value(node.name, shape=shape, dtype=dtype, producer=node.name)
            graph.add_op(
                OpNode(
                    name=node.name,
                    op=OpKind.VIEW,
                    inputs=[input_name],
                    outputs=[node.name],
                    attrs={"target": str(node.target), "args": tuple(node.args[1:])},
                )
            )
            node_to_value[node] = node.name
            continue

        raise FXImportError(
            f"Unsupported FX node '{node.name}' with op '{node.op}' and target '{node.target}'"
        )

    return graph
