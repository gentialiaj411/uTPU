import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pytorch_compiler import compile_model


def _shape_of(value: Any) -> Optional[List[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _op_signature(op: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        op.get("name"),
        op.get("op"),
        tuple(op.get("inputs", [])),
        tuple(op.get("outputs", [])),
    )


def _format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _join_inline(values: Iterable[Any], empty: str = "none", sep: str = ", ") -> str:
    items = [str(value) for value in values if value is not None and str(value) != ""]
    if not items:
        return empty
    return sep.join(items)


def _arrow_chain(values: Iterable[Any], empty: str = "none") -> str:
    items = [str(value) for value in values if value is not None and str(value) != ""]
    if not items:
        return empty
    return " -> ".join(items)


def _format_delta(delta: Dict[str, Any]) -> str:
    if not delta:
        return "none"
    parts = []
    for key, value in delta.items():
        number = int(value)
        prefix = "+" if number > 0 else ""
        parts.append(f"{key} {prefix}{number}")
    return ", ".join(parts)


def _format_values_compact(values: Iterable[Any], empty: str = "none") -> str:
    items = [str(value) for value in values if value is not None and str(value) != ""]
    if not items:
        return empty
    if len(items) == 1:
        return items[0]
    return ", ".join(items)


def _html_escape(text: Any) -> str:
    value = str(text)
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _badge(status: str, label: Optional[str] = None) -> str:
    text = label or status
    safe = "".join(ch if ch.isalnum() else "-" for ch in status.lower()).strip("-") or "neutral"
    return f'<span class="badge badge-{safe}">{_html_escape(text)}</span>'


def _cuda_runtime_note(lowered_ops: Sequence[Dict[str, Any]]) -> Optional[str]:
    blockers: List[str] = []
    for op in lowered_ops:
        for blocker in op["lowering"].get("blockers", []) or []:
            text = str(blocker)
            if text not in blockers:
                blockers.append(text)
    if not blockers:
        return None
    return blockers[0]


def _fx_nodes(result: Any) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    if result.fx_graph is None:
        return nodes
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


def _graph_stage_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    ops = list(snapshot.get("ops", []))
    values = snapshot.get("values", {})
    return {
        "graph_name": snapshot.get("name"),
        "op_count": len(ops),
        "ops": ops,
        "op_kinds": [op.get("op") for op in ops],
        "value_count": len(values),
        "inputs": list(snapshot.get("inputs", [])),
        "outputs": list(snapshot.get("outputs", [])),
    }


def _pass_summary(record: Any) -> Dict[str, Any]:
    before_ops = list(record.before.get("ops", []))
    after_ops = list(record.after.get("ops", []))
    before_signatures = [_op_signature(op) for op in before_ops]
    after_signatures = [_op_signature(op) for op in after_ops]
    removed = [op["name"] for op in before_ops if _op_signature(op) not in set(after_signatures)]
    added = [op["name"] for op in after_ops if _op_signature(op) not in set(before_signatures)]
    before_counter = Counter(op.get("op") for op in before_ops)
    after_counter = Counter(op.get("op") for op in after_ops)
    delta = {
        key: int(after_counter.get(key, 0) - before_counter.get(key, 0))
        for key in sorted(set(before_counter) | set(after_counter))
        if int(after_counter.get(key, 0) - before_counter.get(key, 0)) != 0
    }
    metadata_changed = record.before.get("metadata", {}) != record.after.get("metadata", {})
    changed = bool(
        before_signatures != after_signatures
        or record.before.get("values", {}) != record.after.get("values", {})
        or metadata_changed
    )
    fused_ops = [op["name"] for op in after_ops if op.get("op") in {"linear_relu", "scaled_softmax"}]
    return {
        "pass_name": record.pass_name,
        "changed": changed,
        "before_op_count": len(before_ops),
        "after_op_count": len(after_ops),
        "added_ops": added,
        "removed_ops": removed,
        "op_kind_delta": delta,
        "fused_ops_present": fused_ops,
        "metadata_keys_after": sorted(record.after.get("metadata", {}).keys()),
    }


def _memory_plan_summary(memory_plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": memory_plan.get("method"),
        "logical_value_count": int(memory_plan.get("logical_value_count", 0) or 0),
        "physical_buffer_count": int(memory_plan.get("physical_buffer_count", 0) or 0),
        "naive_persistent_bytes": int(memory_plan.get("naive_persistent_bytes", 0) or 0),
        "planned_peak_bytes": int(memory_plan.get("planned_peak_bytes", 0) or 0),
        "peak_memory_reduction_pct": float(memory_plan.get("peak_memory_reduction_pct", 0.0) or 0.0),
        "activation_bytes_with_reuse": int(memory_plan.get("activation_bytes_with_reuse", 0) or 0),
        "buffers": [
            {
                "buffer": item.get("buffer"),
                "size_bytes": int(item.get("size_bytes", 0) or 0),
                "values": list(item.get("values", [])),
            }
            for item in memory_plan.get("buffers", [])
        ],
    }


def _runtime_plan_stage(result: Any) -> Dict[str, Any]:
    runtime_plan = result.runtime_plan
    memory_plan = runtime_plan.memory_plan if runtime_plan is not None else {}
    return {
        "target": result.target,
        "executable": bool(runtime_plan.executable) if runtime_plan is not None else False,
        "unsupported_ops": list(runtime_plan.unsupported_ops) if runtime_plan is not None else [],
        "input_buffers": len(runtime_plan.input_buffers) if runtime_plan is not None else 0,
        "weight_buffers": len(runtime_plan.weight_buffers) if runtime_plan is not None else 0,
        "intermediate_buffers": len(runtime_plan.intermediate_buffers) if runtime_plan is not None else 0,
        "output_buffers": len(runtime_plan.output_buffers) if runtime_plan is not None else 0,
        "ops": [
            {
                "graph_op": op.graph_op,
                "op": op.op,
                "inputs": list(op.inputs),
                "output": op.output,
                "apply_relu": bool(op.apply_relu),
                "cuda_schedule": dict(op.cuda_schedule) if op.cuda_schedule is not None else None,
                "cuda_schedule_provenance": dict(op.cuda_schedule_provenance)
                if op.cuda_schedule_provenance is not None
                else None,
            }
            for op in (runtime_plan.ops if runtime_plan is not None else [])
        ],
        "memory_plan": _memory_plan_summary(memory_plan),
    }


def _backend_stage(result: Any, stage_name: str) -> Dict[str, Any]:
    lowered_ops = []
    fallback_ops = list(result.plan.fallback_ops) if result.plan is not None else []
    unsupported_ops = list(result.plan.unsupported_ops) if result.plan is not None else []
    for op in result.backend_ops:
        lowering = dict(op.lowering)
        if "program" in lowering:
            lowering["program_bytes"] = len(lowering["program"])
            del lowering["program"]
        if "kernel_source" in lowering:
            lowering["kernel_source_bytes"] = len(lowering["kernel_source"] or "")
            del lowering["kernel_source"]
        lowered_ops.append(
            {
                "graph_op": op.graph_op,
                "op": op.op,
                "target": op.target,
                "fused_activation": op.fused_activation,
                "notes": list(op.notes),
                "lowering": lowering,
            }
        )

    executable_flags = [
        bool(op["lowering"].get("executable_on_current_cuda_path"))
        for op in lowered_ops
        if "executable_on_current_cuda_path" in op["lowering"]
    ]
    utpu_words = sum(int(op["lowering"].get("program_instruction_words", 0) or 0) for op in lowered_ops)
    return {
        "stage_name": stage_name,
        "target": result.target,
        "ok": bool(result.ok),
        "callable": bool(result.callable),
        "fully_lowered_to_backend": bool(result.fully_lowered_to_backend),
        "lowered_op_count": len(lowered_ops),
        "lowered_ops": lowered_ops,
        "fallback_ops": [{"graph_op": op.graph_op, "op": op.op, "notes": list(op.notes)} for op in fallback_ops],
        "unsupported_ops": [
            {"graph_op": op.graph_op, "op": op.op, "notes": list(op.notes)} for op in unsupported_ops
        ],
        "runtime_unsupported": list(result.runtime_plan.unsupported_ops) if result.runtime_plan is not None else [],
        "cuda_executable_on_current_machine": all(executable_flags) if executable_flags else None,
        "utpu_instruction_words_total": int(utpu_words),
    }


def build_compiler_pipeline_visual_report(
    model: Any,
    example_inputs: Any,
    *,
    array_size: int = 16,
) -> Dict[str, Any]:
    cuda_result = compile_model(model, example_inputs, target="cuda", array_size=array_size)
    utpu_result = compile_model(model, example_inputs, target="utpu", array_size=array_size)

    fx_nodes = _fx_nodes(cuda_result)
    imported_snapshot = cuda_result.pass_records[0].before if cuda_result.pass_records else {"ops": []}
    imported_graph = _graph_stage_from_snapshot(imported_snapshot)
    post_pass_graph = {
        "graph_name": cuda_result.graph_ir.name if cuda_result.graph_ir is not None else None,
        "op_count": len(cuda_result.graph_ir.ops) if cuda_result.graph_ir is not None else 0,
        "op_kinds": [op.op for op in (cuda_result.graph_ir.ops if cuda_result.graph_ir is not None else [])],
        "ops": [
            {
                "name": op.name,
                "op": op.op,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
            }
            for op in (cuda_result.graph_ir.ops if cuda_result.graph_ir is not None else [])
        ],
    }
    pass_summaries = [_pass_summary(record) for record in cuda_result.pass_records]
    changed_passes = [item["pass_name"] for item in pass_summaries if item["changed"]]

    cuda_stage = _backend_stage(cuda_result, "CUDA Backend Lowering")
    utpu_stage = _backend_stage(utpu_result, "uTPU ISA Lowering")
    cuda_runtime_note = _cuda_runtime_note(cuda_stage["lowered_ops"])

    caveats = [
        "This demo is a scoped compiler walkthrough for a tiny Linear -> ReLU -> Linear model, not a general PyTorch compiler claim.",
        "This report shows compiler lowering and ISA footprint only; board execution is not claimed by this artifact.",
        "uTPU graph-op support is intentionally narrow: only blocked-FC linear ops lower to the current ISA path; other op families stay unsupported on the uTPU backend.",
    ]
    if cuda_stage["cuda_executable_on_current_machine"] is False:
        caveats.append("CUDA kernels were lowered, but the current machine did not report a runnable CUDA runtime path for every lowered CUDA op.")
    if cuda_stage["fallback_ops"]:
        caveats.append("Some imported ops were not blocked-FC-lowered and rely on the graph-op runtime path or fallback handling.")

    report = {
        "title": "uTPU Compiler Pipeline Visual Report",
        "model_name": cuda_result.model_name,
        "example_input_shape": _shape_of(example_inputs),
        "array_size": int(array_size),
        "summary": {
            "cuda_ok": bool(cuda_result.ok),
            "cuda_callable_runtime_object": bool(cuda_result.callable),
            "cuda_executable_on_current_machine": cuda_stage["cuda_executable_on_current_machine"],
            "cuda_runtime_note": cuda_runtime_note,
            "utpu_ok": bool(utpu_result.ok),
            "utpu_instruction_words_total": int(utpu_stage["utpu_instruction_words_total"]),
            "changed_passes": changed_passes,
            "final_graph_ops": list(post_pass_graph["op_kinds"]),
        },
        "stages": [
            {
                "stage": "PyTorch Module",
                "facts": {
                    "model_class": cuda_result.model_name,
                    "module_repr": str(model),
                    "example_input_shape": _shape_of(example_inputs),
                },
            },
            {
                "stage": "torch.fx Graph",
                "facts": {
                    "node_count": len(fx_nodes),
                    "nodes": fx_nodes,
                    "node_targets": [node["target"] for node in fx_nodes],
                },
            },
            {
                "stage": "Graph IR Imported",
                "facts": imported_graph,
            },
            {
                "stage": "Pass Pipeline",
                "facts": {
                    "pass_count": len(pass_summaries),
                    "changed_passes": changed_passes,
                    "passes": pass_summaries,
                    "final_graph": post_pass_graph,
                },
            },
            {
                "stage": "Runtime Plan",
                "facts": _runtime_plan_stage(cuda_result),
            },
            {
                "stage": "CUDA Backend Lowering",
                "facts": cuda_stage,
            },
            {
                "stage": "uTPU ISA Lowering",
                "facts": utpu_stage,
            },
            {
                "stage": "Evidence / Limitations",
                "facts": {
                    "cuda_fallback_ops": cuda_stage["fallback_ops"],
                    "utpu_unsupported_ops": utpu_stage["unsupported_ops"],
                    "caveats": caveats,
                },
            },
        ],
    }
    return report


def format_visual_report_terminal(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    imported = report["stages"][2]["facts"]
    passes = report["stages"][3]["facts"]
    runtime_plan = report["stages"][4]["facts"]
    cuda = report["stages"][5]["facts"]
    utpu = report["stages"][6]["facts"]
    evidence = report["stages"][7]["facts"]

    lines = [
        "Visual compiler pipeline report",
        "===============================",
        f"model={report['model_name']}  example_input_shape={report['example_input_shape']}",
        "",
        "Pipeline:",
        f"- torch.fx nodes={report['stages'][1]['facts']['node_count']} targets={_join_inline(report['stages'][1]['facts']['node_targets'])}",
        f"- imported_graph_ir ops={_arrow_chain(imported['op_kinds'])}",
        f"- changed_passes={_join_inline(passes['changed_passes'])}",
        f"- final_graph_ops={_arrow_chain(summary['final_graph_ops'])}",
        "",
        "Runtime plan:",
        f"- executable={_format_bool(runtime_plan['executable'])} runtime_ops={_arrow_chain([op['op'] for op in runtime_plan['ops']])}",
        f"- memory_plan buffers={runtime_plan['memory_plan']['physical_buffer_count']} peak_bytes={runtime_plan['memory_plan']['planned_peak_bytes']} reduction_pct={runtime_plan['memory_plan']['peak_memory_reduction_pct']:.1f}",
        "",
        "CUDA backend:",
        f"- lowering=generated lowered_ops={cuda['lowered_op_count']} callable_runtime_object={_format_bool(cuda['callable'])}",
        f"- local_execution={'available' if cuda['cuda_executable_on_current_machine'] else 'unavailable'} reason={summary.get('cuda_runtime_note') or 'none'}",
        f"- fallback_ops={_join_inline([op['graph_op'] for op in cuda['fallback_ops']])} unsupported_ops={_join_inline([op['graph_op'] for op in cuda['unsupported_ops']])}",
        "",
        "uTPU backend:",
        f"- lowered_ops={utpu['lowered_op_count']} instruction_words_total={utpu['utpu_instruction_words_total']}",
        f"- unsupported_ops={_join_inline([op['graph_op'] for op in utpu['unsupported_ops']])}",
        "",
        "Caveats:",
    ]
    for caveat in evidence["caveats"]:
        lines.append(f"- {caveat}")
    return "\n".join(lines)


def _render_key_values(items: Sequence[Tuple[str, Any]]) -> str:
    rows = []
    for key, value in items:
        rows.append(
            f'<div class="kv"><span class="kv-key">{_html_escape(key)}</span><span class="kv-value">{_html_escape(value)}</span></div>'
        )
    return "".join(rows)


def _render_list(title: str, values: Iterable[Any], empty_label: str = "none") -> str:
    items = list(values)
    if not items:
        return f'<div class="mini-section"><h4>{_html_escape(title)}</h4><p class="muted">{_html_escape(empty_label)}</p></div>'
    rows = "".join(f"<li>{_html_escape(value)}</li>" for value in items)
    return f'<div class="mini-section"><h4>{_html_escape(title)}</h4><ul>{rows}</ul></div>'


def render_visual_report_html(report: Dict[str, Any]) -> str:
    module = report["stages"][0]["facts"]
    fx = report["stages"][1]["facts"]
    imported = report["stages"][2]["facts"]
    passes = report["stages"][3]["facts"]
    runtime = report["stages"][4]["facts"]
    cuda = report["stages"][5]["facts"]
    utpu = report["stages"][6]["facts"]
    evidence = report["stages"][7]["facts"]
    changed_passes = [item for item in passes["passes"] if item["changed"]]
    unchanged_passes = [item["pass_name"] for item in passes["passes"] if not item["changed"]]
    cuda_runtime_note = report["summary"].get("cuda_runtime_note")

    def render_pass_row(item: Dict[str, Any]) -> str:
        status = "changed" if item["changed"] else "no-op"
        details = []
        if item["op_kind_delta"]:
            details.append(_format_delta(item["op_kind_delta"]))
        if item["added_ops"]:
            details.append(f"added {_format_values_compact(item['added_ops'])}")
        if item["removed_ops"]:
            details.append(f"removed {_format_values_compact(item['removed_ops'])}")
        if not details:
            details.append("graph shape preserved")
        return (
            '<div class="pass-row">'
            f'<div class="pass-head"><span class="pass-name">{_html_escape(item["pass_name"])}</span>{_badge(status)}</div>'
            f'<div class="pass-body">{_html_escape(" | ".join(details))}</div>'
            "</div>"
        )

    def render_backend_rows(ops: Sequence[Dict[str, Any]], target: str) -> str:
        rows = []
        for op in ops:
            lowering = op["lowering"]
            badges = []
            if op.get("fused_activation"):
                badges.append(_badge("fused", f"fused {op['fused_activation']}"))
            if lowering.get("executable_on_current_cuda_path") is True:
                badges.append(_badge("lowered", "local execution"))
            elif lowering.get("mode") == "utpu_graph_op_unsupported":
                badges.append(_badge("unsupported"))
            elif target == "uTPU" and lowering.get("program_instruction_words") is not None:
                badges.append(_badge("lowered", "ISA emitted"))
                badges.append(_badge("sim-only", "sim/RTL path"))
                if lowering.get("fits_instruction_bram") is True:
                    badges.append(_badge("fused", "fits BRAM"))
            elif lowering.get("mode"):
                badges.append(_badge("lowered"))
            summary_items = [
                f"graph_op={op['graph_op']}",
                f"kind={op['op']}",
            ]
            mode_value = lowering.get("mode")
            if mode_value and target != "uTPU":
                summary_items.append(f"mode={mode_value}")
            if lowering.get("kernel_name"):
                summary_items.append(f"kernel={lowering['kernel_name']}")
            if lowering.get("program_instruction_words") is not None:
                summary_items.append(f"words={lowering.get('program_instruction_words')}")
            if lowering.get("block_ops") is not None:
                summary_items.append(f"block_ops={lowering.get('block_ops')}")
            if target == "CUDA" and lowering.get("executable_on_current_cuda_path") is False:
                summary_items.append("local_execution=unavailable")
            blockers = lowering.get("blockers") or []
            blocker_html = ""
            if blockers:
                label = "CUDA unavailable" if target == "CUDA" else "blockers"
                blocker_html = f'<div class="muted">{_html_escape(label)}: {_html_escape(_format_values_compact(blockers))}</div>'
            rows.append(
                '<div class="backend-row">'
                f'<div class="backend-head"><span>{_html_escape(target)}</span>{"".join(badges)}</div>'
                f'<div class="backend-body">{_html_escape(" | ".join(summary_items))}</div>'
                f"{blocker_html}"
                "</div>"
            )
        return "".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_html_escape(report['title'])}</title>
  <style>
    :root {{
      --bg: #f7fafc;
      --panel: #ffffff;
      --ink: #15202b;
      --muted: #5f6b76;
      --border: #d9e2ec;
      --accent: #0f766e;
      --accent-soft: #e6fffb;
      --blue: #1d4ed8;
      --blue-soft: #e8f0ff;
      --warn: #9a6700;
      --warn-soft: #fff4d6;
      --bad: #b42318;
      --bad-soft: #fde7e7;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.06) 0, transparent 28%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.06) 0, transparent 26%),
        linear-gradient(180deg, #fbfdff 0%, #f4f8fb 100%);
      color: var(--ink);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 24px 40px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
      margin-bottom: 20px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .hero-main {{
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.05;
      font-weight: 700;
    }}
    .lede {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
    }}
    .hero-pipeline {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
      align-items: stretch;
    }}
    .hero-step {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
      position: relative;
      min-height: 96px;
    }}
    .hero-step::after {{
      content: "→";
      position: absolute;
      right: -9px;
      top: 50%;
      transform: translateY(-50%);
      color: #8ca0b3;
      font-size: 22px;
      font-weight: 700;
    }}
    .hero-step:last-child::after {{
      content: "";
    }}
    .hero-step-title {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .hero-step-main {{
      display: block;
      font-size: 17px;
      font-weight: 700;
      line-height: 1.2;
      margin-bottom: 6px;
    }}
    .hero-step-sub {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.35;
    }}
    .hero-side {{
      padding: 18px;
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .kv-grid {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .kv {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      background: #f9fbfd;
    }}
    .kv-key {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .kv-value {{
      display: block;
      font-size: 15px;
      font-weight: 600;
      word-break: break-word;
    }}
    .pipeline {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
      margin-bottom: 20px;
    }}
    .stage {{
      padding: 16px;
      min-height: 138px;
      position: relative;
    }}
    .stage::after {{
      content: "→";
      position: absolute;
      right: -11px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--muted);
      font-size: 24px;
      font-weight: 700;
    }}
    .stage:last-child::after {{
      content: "";
    }}
    .stage h3 {{
      margin: 0 0 12px;
      font-size: 17px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid transparent;
      margin-left: 6px;
      white-space: nowrap;
    }}
    .badge-changed, .badge-lowered, .badge-runnable {{
      background: var(--accent-soft);
      color: var(--accent);
      border-color: #a7f3d0;
    }}
    .badge-no-op, .badge-sim-only {{
      background: var(--blue-soft);
      color: var(--blue);
      border-color: #bfd3ff;
    }}
    .badge-fused {{
      background: var(--warn-soft);
      color: var(--warn);
      border-color: #f0d7aa;
    }}
    .badge-unsupported, .badge-fallback {{
      background: var(--bad-soft);
      color: var(--bad);
      border-color: #ebbbbb;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 18px;
    }}
    .section {{
      padding: 18px;
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 20px;
    }}
    .section h4 {{
      margin: 0 0 8px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .pass-row, .backend-row {{
      border-top: 1px solid var(--border);
      padding: 12px 0;
    }}
    .pass-row:first-child, .backend-row:first-child {{
      border-top: 0;
      padding-top: 0;
    }}
    .pass-head, .backend-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
      flex-wrap: wrap;
    }}
    .pass-name {{
      font-weight: 700;
      font-size: 15px;
    }}
    .pass-body, .backend-body {{
      font-size: 13px;
      line-height: 1.45;
      color: var(--ink);
      word-break: break-word;
    }}
    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .mini-section {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: #f9fbfd;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.5;
      font-size: 13px;
    }}
    .compact-note {{
      border: 1px dashed var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      color: var(--muted);
      margin-top: 14px;
      line-height: 1.45;
      background: #f9fbfd;
    }}
    .code-block {{
      margin-top: 14px;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #f7fafc;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .footer-notes {{
      margin-top: 18px;
      padding: 18px;
    }}
    .footer-notes li {{
      margin-bottom: 8px;
    }}
    @media (max-width: 1100px) {{
      .hero, .grid {{
        grid-template-columns: 1fr;
      }}
      .hero-pipeline {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .hero-step::after {{
        content: "";
      }}
      .pipeline {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .stage::after {{
        content: "";
      }}
    }}
    @media (max-width: 720px) {{
      .page {{
        padding: 20px 14px 28px;
      }}
      h1 {{
        font-size: 28px;
      }}
      .pipeline, .kv-grid, .mini-grid, .hero-pipeline {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="panel hero-main">
        <h1>{_html_escape(report['title'])}</h1>
        <p class="lede">
          Actual compiler run for <strong>{_html_escape(report['model_name'])}</strong> on a tiny
          <strong>Linear -&gt; ReLU -&gt; Linear</strong> flow. The first viewport is intentionally pipeline-first:
          FX import, Graph IR fusion, runtime planning, then CUDA and uTPU lowering with scope boundaries intact.
        </p>
        <div class="hero-pipeline">
          <div class="hero-step">
            <span class="hero-step-title">PyTorch FX</span>
            <span class="hero-step-main">{_html_escape(_arrow_chain(fx["node_targets"]))}</span>
            <span class="hero-step-sub">traced nodes={_html_escape(fx["node_count"])}</span>
          </div>
          <div class="hero-step">
            <span class="hero-step-title">Graph IR</span>
            <span class="hero-step-main">{_html_escape(_arrow_chain(imported["op_kinds"]))}</span>
            <span class="hero-step-sub">imported op count={_html_escape(imported["op_count"])}</span>
          </div>
          <div class="hero-step">
            <span class="hero-step-title">Pass Pipeline</span>
            <span class="hero-step-main">{_html_escape(_arrow_chain(report["summary"]["final_graph_ops"]))}</span>
            <span class="hero-step-sub">changed: {_html_escape(_join_inline(report["summary"]["changed_passes"]))}</span>
          </div>
          <div class="hero-step">
            <span class="hero-step-title">Backends</span>
            <span class="hero-step-main">CUDA + uTPU</span>
            <span class="hero-step-sub">ISA words={_html_escape(report["summary"]["utpu_instruction_words_total"])}</span>
          </div>
        </div>
      </div>
      <div class="panel hero-side">
        <div class="kv-grid">
          {_render_key_values([
              ("Input shape", report["example_input_shape"]),
              ("Final Graph IR ops", _arrow_chain(report["summary"]["final_graph_ops"])),
              ("Changed passes", _join_inline(report["summary"]["changed_passes"])),
              ("uTPU ISA words", report["summary"]["utpu_instruction_words_total"]),
          ])}
        </div>
        <div class="compact-note">
          <strong>CUDA lowering</strong>: generated<br />
          <strong>Local execution</strong>: {_html_escape("available" if cuda["cuda_executable_on_current_machine"] else "unavailable")}<br />
          {_html_escape(cuda_runtime_note or "No local CUDA runtime issue reported.")}
        </div>
      </div>
    </section>

    <section class="pipeline">
      <div class="panel stage">
        <h3>PyTorch Module</h3>
        <div class="muted">class={_html_escape(module["model_class"])}</div>
        <div class="muted">input_shape={_html_escape(module["example_input_shape"])}</div>
        <div class="muted">module structure moved to details below for screenshot clarity</div>
      </div>
      <div class="panel stage">
        <h3>torch.fx Graph</h3>
        <div class="muted">nodes={_html_escape(fx["node_count"])}</div>
        <div class="muted">targets={_html_escape(_arrow_chain(fx["node_targets"]))}</div>
      </div>
      <div class="panel stage">
        <h3>Graph IR Imported</h3>
        <div class="muted">op_count={_html_escape(imported["op_count"])}</div>
        <div class="muted">ops={_html_escape(_arrow_chain(imported["op_kinds"]))}</div>
      </div>
      <div class="panel stage">
        <h3>Pass Pipeline</h3>
        <div class="muted">changed={_html_escape(_join_inline(passes["changed_passes"]))}</div>
        <div class="muted">final_ops={_html_escape(_arrow_chain(passes["final_graph"]["op_kinds"]))}</div>
      </div>
      <div class="panel stage">
        <h3>Runtime Plan</h3>
        <div class="muted">runtime_ops={_html_escape(_arrow_chain([op["op"] for op in runtime["ops"]]))}</div>
        <div class="muted">peak_bytes={_html_escape(runtime["memory_plan"]["planned_peak_bytes"])}</div>
      </div>
      <div class="panel stage">
        <h3>CUDA Lowering</h3>
        <div class="muted">lowered_ops={_html_escape(cuda["lowered_op_count"])}</div>
        <div class="muted">local_execution={_html_escape("available" if cuda["cuda_executable_on_current_machine"] else "unavailable")}</div>
      </div>
      <div class="panel stage">
        <h3>uTPU ISA Lowering</h3>
        <div class="muted">lowered_ops={_html_escape(utpu["lowered_op_count"])}</div>
        <div class="muted">words={_html_escape(utpu["utpu_instruction_words_total"])}</div>
      </div>
      <div class="panel stage">
        <h3>Evidence / Limits</h3>
        <div class="muted">fallback_ops={_html_escape([item["graph_op"] for item in cuda["fallback_ops"]])}</div>
        <div class="muted">utpu_unsupported={_html_escape([item["graph_op"] for item in utpu["unsupported_ops"]])}</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel section">
        <h2>Pass Details</h2>
        {"".join(render_pass_row(item) for item in changed_passes)}
        <div class="compact-note">
          <strong>Unchanged passes</strong>: {_html_escape(_join_inline(unchanged_passes))}
        </div>
      </div>
      <div class="panel section">
        <h2>Runtime Plan</h2>
        <div class="kv-grid">
          {_render_key_values([
              ("Executable", "yes" if runtime["executable"] else "no"),
              ("Input buffers", runtime["input_buffers"]),
              ("Weight buffers", runtime["weight_buffers"]),
              ("Intermediate buffers", runtime["intermediate_buffers"]),
              ("Output buffers", runtime["output_buffers"]),
              ("Peak bytes", runtime["memory_plan"]["planned_peak_bytes"]),
          ])}
        </div>
        <div class="mini-grid">
          {_render_list("Runtime ops", [f'{op["graph_op"]}: {op["op"]}' for op in runtime["ops"]])}
          {_render_list("Memory buffers", [f'{buf["buffer"]}: {buf["size_bytes"]} B -> {_format_values_compact(buf["values"])}' for buf in runtime["memory_plan"]["buffers"]])}
        </div>
        <div class="code-block">{_html_escape(module["module_repr"])}</div>
      </div>
    </section>

    <section class="grid" style="margin-top: 18px;">
      <div class="panel section">
        <h2>CUDA Backend Lowering</h2>
        {render_backend_rows(cuda["lowered_ops"], "CUDA")}
        <div class="mini-grid">
          {_render_list("Fallback ops", [item["graph_op"] for item in cuda["fallback_ops"]])}
          {_render_list("Unsupported ops", [item["graph_op"] for item in cuda["unsupported_ops"]])}
        </div>
      </div>
      <div class="panel section">
        <h2>uTPU ISA Lowering</h2>
        {render_backend_rows(utpu["lowered_ops"], "uTPU")}
        <div class="mini-grid">
          {_render_list("uTPU unsupported", [item["graph_op"] for item in utpu["unsupported_ops"]])}
          {_render_list("uTPU runtime unsupported", utpu["runtime_unsupported"])}
        </div>
      </div>
    </section>

    <section class="panel footer-notes">
      <h2>Evidence / Limitations</h2>
      <ul>
        {"".join(f"<li>{_html_escape(item)}</li>" for item in evidence["caveats"])}
      </ul>
    </section>
  </div>
</body>
</html>
"""


def write_visual_report(report: Dict[str, Any], json_path: str, html_path: str) -> None:
    for path in (json_path, html_path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(render_visual_report_html(report))
