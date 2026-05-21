import json
import os

from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import memory_planning_pass


def test_kv_cache_planning_layout_and_reuse():
    num_layers = 3
    num_heads = 4
    head_dim = 16
    max_seq_len = 32
    shape = (1, num_heads, max_seq_len, head_dim)

    graph = GraphIR(name="kv_cache_memory_plan")
    graph.inputs = ["x"]
    graph.outputs = ["o2"]
    graph.add_value("x", shape=(1, num_heads * head_dim), dtype="float16")

    for i in range(num_layers):
        k_name = f"kv_layer{i}_k"
        v_name = f"kv_layer{i}_v"
        graph.add_value(k_name, shape=shape, dtype="float16", persistent=True)
        graph.add_value(v_name, shape=shape, dtype="float16", persistent=True)

    graph.add_op(
        OpNode(
            name="fc0",
            op=OpKind.LINEAR,
            inputs=["x"],
            outputs=["o0"],
            attrs={"in_features": num_heads * head_dim, "out_features": num_heads * head_dim},
        )
    )
    graph.add_op(
        OpNode(
            name="fc1",
            op=OpKind.LINEAR,
            inputs=["o0"],
            outputs=["o1"],
            attrs={"in_features": num_heads * head_dim, "out_features": num_heads * head_dim},
        )
    )
    graph.add_op(
        OpNode(
            name="fc2",
            op=OpKind.LINEAR,
            inputs=["o1"],
            outputs=["o2"],
            attrs={"in_features": num_heads * head_dim, "out_features": num_heads * head_dim},
        )
    )

    planned = memory_planning_pass(graph)
    plan = planned.metadata["memory_plan"]
    kv_layout = plan.get("kv_cache_layout")
    assert kv_layout is not None
    assert len(kv_layout["layers"]) == num_layers
    for layer in kv_layout["layers"]:
        assert "k" in layer and "v" in layer
        assert tuple(layer["k"]["shape"]) == shape
        assert tuple(layer["v"]["shape"]) == shape

    buffers = plan["buffers"]
    assert len(buffers) < plan["logical_value_count"]

    bytes_per_elem = 2
    per_tensor = num_heads * head_dim * max_seq_len * bytes_per_elem
    bytes_per_layer = per_tensor * 2
    total_kv_bytes = bytes_per_layer * num_layers

    report_path = os.path.join("build", "reports", "kv_cache_plan_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_layers": num_layers,
                "bytes_per_layer": bytes_per_layer,
                "total_kv_bytes": total_kv_bytes,
                "max_seq_len": max_seq_len,
                "activation_bytes_with_reuse": plan["activation_bytes_with_reuse"],
            },
            f,
            indent=2,
            sort_keys=True,
        )
