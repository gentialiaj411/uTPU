import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from compiler_abstractions import BlockedFCProblem, build_blocked_fc_schedule, cuda_target_desc
from pytorch_compiler import PyTorchCompileResult, compile_mlp_model


def _shape_of(value: Any) -> Optional[List[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(d) for d in shape]


def _summarize_array(value: Any) -> Dict[str, Any]:
    arr = np.asarray(value)
    summary = {
        "shape": [int(d) for d in arr.shape],
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if arr.size:
        summary.update(
            {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )
    return summary


def _clean_attr(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _summarize_array(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_clean_attr(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean_attr(v) for k, v in value.items()}
    return type(value).__name__


def _fx_nodes(result: PyTorchCompileResult) -> List[Dict[str, Any]]:
    if result.fx_graph is None:
        return []
    nodes = []
    for node in result.fx_graph.graph.nodes:
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "args": [getattr(arg, "name", repr(arg)) for arg in node.args],
            }
        )
    return nodes


def _graph_ir(result: PyTorchCompileResult) -> Dict[str, Any]:
    graph = result.graph_ir
    if graph is None:
        return {}
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
            }
            for name, value in graph.values.items()
        },
        "ops": [
            {
                "name": op.name,
                "op": op.op,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attrs": {k: _clean_attr(v) for k, v in op.attrs.items() if k != "module"},
            }
            for op in graph.ops
        ],
    }


def _runtime_plan(result: PyTorchCompileResult) -> Dict[str, Any]:
    plan = result.runtime_plan
    if plan is None:
        return {}
    return {
        "graph_name": plan.graph_name,
        "target": plan.target,
        "executable": bool(plan.executable),
        "input_buffers": [asdict(v) for v in plan.input_buffers],
        "weight_buffers": [asdict(v) for v in plan.weight_buffers],
        "intermediate_buffers": [asdict(v) for v in plan.intermediate_buffers],
        "output_buffers": [asdict(v) for v in plan.output_buffers],
        "ops": [asdict(v) for v in plan.ops],
        "unsupported_ops": list(plan.unsupported_ops),
    }


def _compile_plan(result: PyTorchCompileResult) -> Dict[str, Any]:
    plan = result.plan
    if plan is None:
        return {}

    def planned(op):
        item = {
            "graph_op": op.graph_op,
            "op": op.op,
            "status": op.status,
            "fused_activation": op.fused_activation,
            "notes": list(op.notes),
        }
        if op.request is not None:
            req = op.request
            schedule = build_blocked_fc_schedule(
                problem=BlockedFCProblem(
                    out_features=req.out_features,
                    in_features=req.in_features,
                    array_size=req.array_size,
                ),
                target=cuda_target_desc(array_size=req.array_size),
            )
            item["request"] = {
                "out_features": int(req.out_features),
                "in_features": int(req.in_features),
                "array_size": int(req.array_size),
                "apply_relu": bool(req.apply_relu),
                "apply_quant": bool(req.apply_quant),
                "weights": _summarize_array(req.weights_int4),
                "activations": _summarize_array(req.activations_int4),
            }
            item["blocked_schedule"] = asdict(schedule)
        return item

    return {
        "graph_name": plan.graph_name,
        "is_fully_supported": bool(plan.is_fully_supported),
        "lowered_ops": [planned(op) for op in plan.lowered_ops],
        "fallback_ops": [planned(op) for op in plan.fallback_ops],
        "unsupported_ops": [planned(op) for op in plan.unsupported_ops],
    }


def _backend_ops(result: PyTorchCompileResult) -> List[Dict[str, Any]]:
    ops = []
    for op in result.backend_ops:
        lowering = {
            k: _clean_attr(v)
            for k, v in op.lowering.items()
            if k not in {"program", "kernel_source"}
        }
        lowering["program_bytes"] = len(op.lowering.get("program", b"")) if "program" in op.lowering else None
        lowering["kernel_source_bytes"] = len(op.lowering.get("kernel_source", "")) if "kernel_source" in op.lowering else None
        ops.append(
            {
                "graph_op": op.graph_op,
                "op": op.op,
                "target": op.target,
                "fused_activation": op.fused_activation,
                "notes": list(op.notes),
                "lowering": lowering,
            }
        )
    return ops


def inspect_compiled_mlp(
    model: Any,
    example_inputs: Any,
    array_size: int = 16,
) -> Dict[str, Any]:
    cuda = compile_mlp_model(model, example_inputs, target="cuda", array_size=array_size)
    utpu = compile_mlp_model(model, example_inputs, target="utpu", array_size=array_size)
    summary = {
        "scope": {
            "supported_path": "batch-1 MLP-style Linear/ReLU/Linear subset",
            "not_claimed": [
                "arbitrary PyTorch support",
                "transformer support",
                "production torch.compile backend",
                "physical board validation from Graph IR",
                "PyTorch/cuBLAS speedup",
            ],
        },
        "model_name": cuda.model_name,
        "example_input_shape": _shape_of(example_inputs),
        "cuda_summary": cuda.summary(),
        "utpu_summary": utpu.summary(),
        "fx_graph": _fx_nodes(cuda),
        "graph_ir": _graph_ir(cuda),
        "compile_plan": _compile_plan(cuda),
        "runtime_plan": _runtime_plan(cuda),
        "cuda_backend_ops": _backend_ops(cuda),
        "utpu_backend_ops": _backend_ops(utpu),
    }
    summary["derived"] = {
        "cuda_fallback_ops": list(summary["cuda_summary"].get("fallback_ops", [])),
        "cuda_unsupported_ops": list(summary["cuda_summary"].get("unsupported_ops", [])),
        "utpu_instruction_words_total": int(
            sum(
                op["lowering"].get("program_instruction_words", 0) or 0
                for op in summary["utpu_backend_ops"]
            )
        ),
        "utpu_all_lowered_ops_fit_instruction_bram": all(
            bool(op["lowering"].get("fits_instruction_bram", False))
            for op in summary["utpu_backend_ops"]
        )
        if summary["utpu_backend_ops"]
        else False,
    }
    return summary


def format_introspection_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("Compiler pipeline inspection")
    lines.append("============================")
    lines.append(f"model={report['model_name']}")
    lines.append(f"example_input_shape={report['example_input_shape']}")
    lines.append("")
    lines.append("Honest scope:")
    lines.append(f"- supported_path={report['scope']['supported_path']}")
    for item in report["scope"]["not_claimed"]:
        lines.append(f"- not_claimed={item}")
    lines.append("")
    lines.append("FX graph:")
    for node in report["fx_graph"]:
        lines.append(f"- {node['name']}: op={node['op']} target={node['target']} args={node['args']}")
    lines.append("")
    lines.append("Graph IR ops:")
    for op in report["graph_ir"].get("ops", []):
        lines.append(f"- {op['name']}: op={op['op']} inputs={op['inputs']} outputs={op['outputs']}")
    lines.append("")
    lines.append("Blocked lowering:")
    for op in report["compile_plan"].get("lowered_ops", []):
        sched = op.get("blocked_schedule", {})
        req = op.get("request", {})
        lines.append(
            f"- {op['graph_op']}: M={req.get('out_features')} K={req.get('in_features')} "
            f"out_blocks={sched.get('out_blocks')} in_blocks={sched.get('in_blocks')} "
            f"out_padded={sched.get('out_padded')} in_padded={sched.get('in_padded')} "
            f"fused_activation={op.get('fused_activation')}"
        )
    lines.append("")
    lines.append(f"fallback_ops={report['derived']['cuda_fallback_ops']}")
    lines.append(f"unsupported_ops={report['derived']['cuda_unsupported_ops']}")
    lines.append("")
    lines.append("CUDA backend ops:")
    for op in report["cuda_backend_ops"]:
        lowering = op["lowering"]
        lines.append(
            f"- {op['graph_op']}: mode={lowering.get('mode')} kernel={lowering.get('kernel_name')} "
            f"executable={lowering.get('executable_on_current_cuda_path')} blockers={lowering.get('blockers')}"
        )
    lines.append("")
    lines.append("uTPU ISA footprint:")
    for op in report["utpu_backend_ops"]:
        lowering = op["lowering"]
        lines.append(
            f"- {op['graph_op']}: words={lowering.get('program_instruction_words')} "
            f"block_ops={lowering.get('block_ops')} fits_bram={lowering.get('fits_instruction_bram')}"
        )
    lines.append(
        f"total_utpu_instruction_words={report['derived']['utpu_instruction_words_total']}"
    )
    lines.append(
        f"all_lowered_ops_fit_instruction_bram={report['derived']['utpu_all_lowered_ops_fit_instruction_bram']}"
    )
    return "\n".join(lines)


def write_introspection_json(report: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
