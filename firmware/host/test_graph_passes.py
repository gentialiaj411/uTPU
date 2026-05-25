import importlib.util

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import (
    BackendLegalityError,
    CONV_BN_FUSION_RULE,
    DEFAULT_FUSION_RULES,
    FusionEngine,
    FusionRewrite,
    FusionRule,
    GraphPassManager,
    LINEAR_RELU_FUSION_RULE,
    SCALE_SOFTMAX_FUSION_RULE,
    backend_legality_pass,
    conv_bn_fusion_pass,
    dead_code_elimination_pass,
    linear_relu_fusion_pass,
    memory_planning_pass,
    scale_softmax_fusion_pass,
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


# ---------------------------------------------------------------------------
# FusionEngine: rule registry + legality predicates (Phase 2).
# ---------------------------------------------------------------------------


def _build_linear_relu_graph(multi_consumer: bool = False) -> GraphIR:
    graph = GraphIR(name="lr")
    graph.inputs = ["x"]
    graph.outputs = ["y"] if not multi_consumer else ["y", "h"]
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(_linear("fc1", "x", "h", 4, 3))
    graph.add_op(OpNode(name="relu1", op=OpKind.RELU, inputs=["h"], outputs=["y"]))
    return graph


def _build_scale_softmax_graph() -> GraphIR:
    graph = GraphIR(name="ss")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(
        OpNode(name="scale1", op=OpKind.SCALE, inputs=["x"], outputs=["s"], attrs={"scale": 0.125})
    )
    graph.add_op(
        OpNode(name="softmax1", op=OpKind.SOFTMAX, inputs=["s"], outputs=["y"], attrs={"causal_mask": True})
    )
    return graph


def _build_conv_bn_graph() -> GraphIR:
    graph = GraphIR(name="cbn")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_value("x", shape=(1, 3, 8, 8), dtype="torch.float32")
    weight = np.ones((4, 3, 3, 3), dtype=np.float32)
    graph.add_op(
        OpNode(
            name="conv1",
            op=OpKind.CONV2D,
            inputs=["x"],
            outputs=["c"],
            attrs={"weight": weight, "bias": None, "stride": 1, "padding": 1},
        )
    )
    graph.add_op(
        OpNode(
            name="bn1",
            op=OpKind.BATCH_NORM,
            inputs=["c"],
            outputs=["y"],
            attrs={
                "weight": np.ones((4,), dtype=np.float32),
                "bias": np.zeros((4,), dtype=np.float32),
                "running_mean": np.zeros((4,), dtype=np.float32),
                "running_var": np.ones((4,), dtype=np.float32),
                "eps": 1e-5,
            },
        )
    )
    return graph


def test_fusion_engine_applies_linear_relu_rule_and_logs():
    graph = _build_linear_relu_graph()
    result = FusionEngine([LINEAR_RELU_FUSION_RULE]).run(graph)

    assert [op.op for op in result.graph.ops] == [OpKind.LINEAR_RELU]
    assert result.graph.ops[0].outputs == ["y"]
    assert result.graph.outputs == ["y"]
    assert len(result.applications) == 1
    app = result.applications[0]
    assert app.rule_name == "linear_relu_fusion"
    assert app.producer_name == "fc1"
    assert app.consumer_name == "relu1"


def test_fusion_engine_applies_scale_softmax_rule_and_preserves_causal_mask():
    graph = _build_scale_softmax_graph()
    result = FusionEngine([SCALE_SOFTMAX_FUSION_RULE]).run(graph)

    assert [op.op for op in result.graph.ops] == [OpKind.SCALED_SOFTMAX]
    fused = result.graph.ops[0]
    assert fused.outputs == ["y"]
    assert fused.attrs["scale"] == 0.125
    assert fused.attrs["causal_mask"] is True
    assert len(result.applications) == 1
    assert result.applications[0].rule_name == "scale_softmax_fusion"


def test_fusion_engine_folds_conv_bn_into_conv_and_rewires_downstream():
    graph = _build_conv_bn_graph()
    result = FusionEngine([CONV_BN_FUSION_RULE]).run(graph)

    assert [op.op for op in result.graph.ops] == [OpKind.CONV2D]
    fused = result.graph.ops[0]
    assert fused.name == "conv1"
    assert fused.outputs == ["c"]
    assert fused.attrs.get("bn_fused") is True
    # graph output 'y' (BN output) must be rewired to conv output 'c'
    assert result.graph.outputs == ["c"]
    assert "y" not in result.graph.values
    assert len(result.applications) == 1
    app = result.applications[0]
    assert app.consumer_name == "bn1"
    assert app.aliased_values == {"y": "c"}


def test_fusion_engine_rejects_multi_consumer_producer():
    """Producer with more than one consumer must not be fused (legality
    predicate: single-consumer enforced by engine before rule's legality
    callback)."""
    graph = _build_linear_relu_graph(multi_consumer=True)
    # Add a second consumer of 'h' so the LINEAR has 2 consumers.
    graph.add_op(OpNode(name="relu2", op=OpKind.RELU, inputs=["h"], outputs=["h_alt"]))
    graph.outputs = ["y", "h_alt"]

    result = FusionEngine([LINEAR_RELU_FUSION_RULE]).run(graph)

    op_kinds = [op.op for op in result.graph.ops]
    assert OpKind.LINEAR in op_kinds
    assert OpKind.LINEAR_RELU not in op_kinds
    assert result.applications == []


def test_fusion_engine_rejects_consumer_with_wrong_op_kind():
    graph = GraphIR(name="no_match")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(_linear("fc1", "x", "h", 4, 3))
    # Consumer is ADD not RELU → rule should not fire.
    graph.add_op(OpNode(name="add1", op=OpKind.ADD, inputs=["h", "x"], outputs=["y"]))

    result = FusionEngine([LINEAR_RELU_FUSION_RULE]).run(graph)
    assert [op.op for op in result.graph.ops] == [OpKind.LINEAR, OpKind.ADD]
    assert result.applications == []


def test_fusion_engine_legality_predicate_can_veto_match():
    """A rule's legality predicate may veto an otherwise-matching pair;
    the engine must respect it."""

    def _veto(producer, consumer, graph):
        return "vetoed_for_test"

    custom = FusionRule(
        name="veto_rule",
        producer_op=OpKind.LINEAR,
        consumer_op=OpKind.RELU,
        legality=_veto,
        rewrite=lambda p, c, g: FusionRewrite(new_ops=()),
        description="always vetoes",
    )
    result = FusionEngine([custom]).run(_build_linear_relu_graph())
    assert [op.op for op in result.graph.ops] == [OpKind.LINEAR, OpKind.RELU]
    assert result.applications == []


def test_fusion_engine_runs_multiple_rules_in_registration_order():
    """When rule A fires first, rule B sees the post-A graph. Chains that
    require A-then-B should still resolve."""
    graph = _build_linear_relu_graph()
    # Also add a scale+softmax tail so both rules have something to fuse.
    graph.outputs = ["y2"]
    graph.add_op(
        OpNode(name="scale1", op=OpKind.SCALE, inputs=["y"], outputs=["sc"], attrs={"scale": 2.0})
    )
    graph.add_op(OpNode(name="softmax1", op=OpKind.SOFTMAX, inputs=["sc"], outputs=["y2"], attrs={}))

    result = FusionEngine(list(DEFAULT_FUSION_RULES)).run(graph)
    op_kinds = [op.op for op in result.graph.ops]
    assert OpKind.LINEAR_RELU in op_kinds
    assert OpKind.SCALED_SOFTMAX in op_kinds
    assert OpKind.LINEAR not in op_kinds
    assert OpKind.RELU not in op_kinds
    assert OpKind.SOFTMAX not in op_kinds
    rule_names = {app.rule_name for app in result.applications}
    assert rule_names == {"linear_relu_fusion", "scale_softmax_fusion"}


def test_pass_wrappers_match_engine_output_byte_identically():
    """Public pass wrappers must produce graphs identical to the engine
    invoked with the same rule."""
    for graph_builder, rule, pass_fn in [
        (_build_linear_relu_graph, LINEAR_RELU_FUSION_RULE, linear_relu_fusion_pass),
        (_build_scale_softmax_graph, SCALE_SOFTMAX_FUSION_RULE, scale_softmax_fusion_pass),
        (_build_conv_bn_graph, CONV_BN_FUSION_RULE, conv_bn_fusion_pass),
    ]:
        graph = graph_builder()
        via_engine = FusionEngine([rule]).run(graph_builder()).graph
        via_pass = pass_fn(graph)
        assert [op.name for op in via_pass.ops] == [op.name for op in via_engine.ops]
        assert [op.op for op in via_pass.ops] == [op.op for op in via_engine.ops]
        assert [op.inputs for op in via_pass.ops] == [op.inputs for op in via_engine.ops]
        assert [op.outputs for op in via_pass.ops] == [op.outputs for op in via_engine.ops]
        assert via_pass.outputs == via_engine.outputs
        assert set(via_pass.values.keys()) == set(via_engine.values.keys())


def test_default_fusion_rules_ordering_is_stable():
    """Lock the canonical fusion-rule order so downstream pass ordering
    (relative to memory_planning and backend_legality) is preserved."""
    assert [r.name for r in DEFAULT_FUSION_RULES] == [
        "conv_bn_fusion",
        "linear_relu_fusion",
        "scale_softmax_fusion",
    ]
