import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import memory_planning_pass, shape_inference_pass


REPORT_JSON_PATH = Path("build/reports/memory_plan_report.json")
REPORT_MD_PATH = Path("build/reports/memory_plan_report.md")


def _linear(name: str, input_name: str, output_name: str, in_features: int, out_features: int) -> OpNode:
    return OpNode(
        name=name,
        op=OpKind.LINEAR,
        inputs=[input_name],
        outputs=[output_name],
        attrs={
            "weight": np.ones((out_features, in_features), dtype=np.float32),
            "bias": None,
            "in_features": int(in_features),
            "out_features": int(out_features),
        },
    )


def build_sample_graph() -> GraphIR:
    graph = GraphIR(name="memory_plan_liveness_sample")
    graph.inputs = ["x"]
    graph.outputs = ["b", "c"]
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(_linear("fc1", "x", "a", 4, 4))
    graph.add_op(_linear("fc2", "a", "b", 4, 4))
    graph.add_op(_linear("fc3", "x", "c", 4, 4))
    return graph


def generate_report(output_json: Path = REPORT_JSON_PATH, output_md: Path = REPORT_MD_PATH) -> dict:
    graph = memory_planning_pass(shape_inference_pass(build_sample_graph()))
    memory_plan = graph.metadata["memory_plan"]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "graph_name": graph.name,
        "op_count": len(graph.ops),
        "memory_plan": memory_plan,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Memory Plan Report",
        "",
        f"- timestamp_utc: {report['timestamp_utc']}",
        f"- graph_name: {report['graph_name']}",
        f"- method: {memory_plan['method']}",
        f"- logical_value_count: {memory_plan['logical_value_count']}",
        f"- physical_buffer_count: {memory_plan['physical_buffer_count']}",
        f"- naive_persistent_bytes: {memory_plan['naive_persistent_bytes']}",
        f"- planned_peak_bytes: {memory_plan['planned_peak_bytes']}",
        f"- peak_memory_reduction_pct: {memory_plan['peak_memory_reduction_pct']:.2f}%",
        "",
        "| value | buffer | first_def | last_use | size_bytes |",
        "|---|---|---:|---:|---:|",
    ]
    for value in memory_plan["values"]:
        lines.append(
            f"| {value['value']} | {value['buffer']} | {value['first_def']} | "
            f"{value['last_use']} | {value['size_bytes']} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = generate_report()
    print(json.dumps(report["memory_plan"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
