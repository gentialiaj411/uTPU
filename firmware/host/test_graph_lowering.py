import importlib.util

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_lowering import plan_blocked_fc_graph


def _manual_linear_relu_graph():
    graph = GraphIR(name="manual_linear_relu")
    graph.inputs.append("x")
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_value("fc", shape=(1, 3), dtype="torch.float32", producer="fc")
    graph.add_op(
        OpNode(
            name="fc",
            op=OpKind.LINEAR,
            inputs=["x"],
            outputs=["fc"],
            attrs={
                "weight": np.array(
                    [
                        [1, -1, 2, 0],
                        [0, 1, -2, 3],
                        [2, 2, 2, 2],
                    ],
                    dtype=np.float32,
                ),
                "bias": None,
                "in_features": 4,
                "out_features": 3,
            },
        )
    )
    graph.add_value("relu", shape=(1, 3), dtype="torch.float32", producer="relu")
    graph.add_op(
        OpNode(
            name="relu",
            op=OpKind.RELU,
            inputs=["fc"],
            outputs=["relu"],
        )
    )
    graph.outputs.append("relu")
    return graph


def test_linear_relu_plans_blocked_fc_request():
    graph = _manual_linear_relu_graph()
    plan = plan_blocked_fc_graph(
        graph,
        array_size=16,
        activation_values={"x": np.array([1, 2, 3, 4], dtype=np.int8)},
    )

    assert len(plan.lowered_ops) == 1
    assert not plan.fallback_ops
    lowered = plan.lowered_ops[0]
    assert lowered.graph_op == "fc"
    assert lowered.fused_activation == "relu"
    assert lowered.request is not None
    assert lowered.request.out_features == 3
    assert lowered.request.in_features == 4
    assert lowered.request.array_size == 16
    assert lowered.request.apply_relu is True
    assert lowered.request.apply_quant is True
    assert lowered.request.weights_int4.shape == (3, 4)
    assert lowered.request.activations_int4.tolist() == [1, 2, 3, 4]


def test_unfused_relu_is_fallback():
    graph = GraphIR(name="standalone_relu")
    graph.inputs.append("x")
    graph.add_value("x", shape=(1, 4), dtype="torch.float32")
    graph.add_op(OpNode(name="relu", op=OpKind.RELU, inputs=["x"], outputs=["relu"]))

    plan = plan_blocked_fc_graph(graph)

    assert len(plan.lowered_ops) == 0
    assert len(plan.fallback_ops) == 1
    assert plan.fallback_ops[0].op == OpKind.RELU
    assert plan.is_fully_supported is False


def test_fx_linear_lowering_when_torch_available():
    if importlib.util.find_spec("torch") is None:
        print("test_graph_lowering FX path: SKIP (PyTorch not installed)")
        return

    import torch
    import torch.nn as nn
    from torch.fx.passes.shape_prop import ShapeProp

    from fx_importer import import_fx_graph_module

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(torch.randn(1, 4))
    graph = import_fx_graph_module(gm, name="fx_linear_relu")
    plan = plan_blocked_fc_graph(graph, activation_values={graph.inputs[0]: np.ones(4, dtype=np.int8)})

    assert len(plan.lowered_ops) == 1
    assert plan.lowered_ops[0].request.out_features == 3
    assert plan.lowered_ops[0].request.in_features == 4
    assert plan.lowered_ops[0].request.apply_relu is True


def run_all():
    test_linear_relu_plans_blocked_fc_request()
    test_unfused_relu_is_fallback()
    test_fx_linear_lowering_when_torch_available()
    print("test_graph_lowering: PASS")


if __name__ == "__main__":
    run_all()
