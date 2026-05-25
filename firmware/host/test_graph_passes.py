import importlib.util

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import (
    BackendLegalityError,
    GraphPassManager,
    backend_legality_pass,
    dead_code_elimination_pass,
    linear_relu_fusion_pass,
    memory_planning_pass,
    shape_inference_pass,
)


def _linear(name: str, input_name: str, output_name: str, in_features: int, out_features: int) -> OpNode:
    return OpNode(
        name=name,
        op=OpKind.LINEAR,
        inputs=[input_name],
        outputs=[output_name],
        attrs={
            "weight": np.ones((out_features, in_features), dtype=np.float32),
            "bias": None,
            "in_features": in_features,
            "out_features": out_features,
        },
    )


def test_shape_inference_pass_propagates_linear_and_relu_shapes():
    graph = GraphIR(name="shape_infer")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(_linear("fc1", "x", "h", 4, 3))
    graph.add_op(OpNode(name="relu1", op=OpKind.RELU, inputs=["h"], outputs=["y"]))

    inferred = shape_inference_pass(graph)

    assert inferred.values["h"].shape == (1, 3)
    assert inferred.values["y"].shape == (1, 3)
    assert inferred.values["h"].dtype == "torch.float32"
    assert inferred.values["y"].dtype == "torch.float32"


def test_linear_relu_fusion_pass_creates_fused_op():
    graph = GraphIR(name="fuse")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(_linear("fc1", "x", "h", 4, 3))
    graph.add_op(OpNode(name="relu1", op=OpKind.RELU, inputs=["h"], outputs=["y"]))

    fused = linear_relu_fusion_pass(graph)

    assert [op.op for op in fused.ops] == [OpKind.LINEAR_RELU]
    assert fused.ops[0].outputs == ["y"]
    assert fused.values["y"].producer == fused.ops[0].name


def test_dead_code_elimination_pass_removes_orphan_ops():
    graph = GraphIR(name="dce")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(_linear("fc_live", "x", "h", 4, 3))
    graph.add_op(OpNode(name="relu_live", op=OpKind.RELU, inputs=["h"], outputs=["y"]))
    graph.add_op(_linear("fc_dead", "x", "dead", 4, 2))

    pruned = dead_code_elimination_pass(graph)

    assert [op.name for op in pruned.ops] == ["fc_live", "relu_live"]
    assert "dead" not in pruned.values


def test_backend_legality_pass_reports_offending_ops():
    graph = GraphIR(name="legality")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(OpNode(name="bad1", op="unsupported_op", inputs=["x"], outputs=["y"]))

    try:
        backend_legality_pass(graph, backend="cuda")
        raise AssertionError("Expected backend_legality_pass to raise")
    except BackendLegalityError as e:
        payload = e.to_dict()
        assert payload["backend"] == "cuda"
        assert len(payload["offending_ops"]) == 1
        assert payload["offending_ops"][0]["name"] == "bad1"
        assert payload["offending_ops"][0]["op"] == "unsupported_op"


def test_memory_planning_pass_reuses_non_overlapping_buffers():
    graph = GraphIR(name="memory_reuse")
    graph.inputs = ["x"]
    graph.outputs = ["b", "c"]
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(_linear("fc1", "x", "a", 4, 4))
    graph.add_op(_linear("fc2", "a", "b", 4, 4))
    graph.add_op(_linear("fc3", "x", "c", 4, 4))

    inferred = shape_inference_pass(graph)
    planned = memory_planning_pass(inferred)
    memory_plan = planned.metadata["memory_plan"]

    assert memory_plan["logical_value_count"] == 3
    assert memory_plan["physical_buffer_count"] == 2
    assert memory_plan["planned_peak_bytes"] < memory_plan["naive_persistent_bytes"]
    values = {item["value"]: item for item in memory_plan["values"]}
    assert values["a"]["buffer"] == values["c"]["buffer"]
    assert values["b"]["buffer"] != values["a"]["buffer"]


def test_full_pass_pipeline_on_sample_mlp():
    if importlib.util.find_spec("torch") is None:
        return
    import torch
    import torch.nn as nn
    from torch.fx.passes.shape_prop import ShapeProp

    from fx_importer import import_fx_graph_module

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(torch.randn(1, 4))
    graph = import_fx_graph_module(gm, name="pipeline_mlp")
    graph.add_op(_linear("dead_linear", graph.inputs[0], "dead", 4, 2))

    result = GraphPassManager(target_backend="cuda").run(graph)
    final_graph = result.graph

    assert [record.pass_name for record in result.records] == [
        "shape_inference",
        "conv_bn_fusion",
        "shape_inference_post_fold",
        "attention_decomposition",
        "linear_relu_fusion",
        "scale_softmax_fusion",
        "dead_code_elimination",
        "memory_planning",
        "backend_legality",
    ]
    assert any(op.op == OpKind.LINEAR_RELU for op in final_graph.ops)
    assert all(op.op != OpKind.RELU for op in final_graph.ops)
    assert "dead" not in final_graph.values
    assert "memory_plan" in final_graph.metadata
