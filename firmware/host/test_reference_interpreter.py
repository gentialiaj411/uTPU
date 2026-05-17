import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_reference_interpreter import GraphReferenceInterpreter, execute_graph_reference


def _tiny_graph() -> GraphIR:
    graph = GraphIR(name="tiny_ref")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_value("x", shape=(1, 2), dtype="torch.float32")
    graph.add_op(
        OpNode(
            name="fc1_relu",
            op=OpKind.LINEAR_RELU,
            inputs=["x"],
            outputs=["h"],
            attrs={
                "weight": np.array([[1.0, -1.0], [2.0, 0.0]], dtype=np.float32),
                "bias": np.array([0.5, -1.0], dtype=np.float32),
                "in_features": 2,
                "out_features": 2,
            },
        )
    )
    graph.add_op(
        OpNode(
            name="fc2",
            op=OpKind.LINEAR,
            inputs=["h"],
            outputs=["y"],
            attrs={
                "weight": np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
                "bias": np.array([0.0, 0.25], dtype=np.float32),
                "in_features": 2,
                "out_features": 2,
            },
        )
    )
    return graph


def test_reference_interpreter_matches_hand_computed_output():
    graph = _tiny_graph()
    x = np.array([[3.0, 1.0]], dtype=np.float32)
    y = execute_graph_reference(graph, x)

    expected = np.array([[7.5, 5.25]], dtype=np.float32)
    assert np.array_equal(y, expected)


def test_reference_interpreter_is_deterministic_for_same_input():
    graph = _tiny_graph()
    interpreter = GraphReferenceInterpreter(graph)
    x = np.array([[3.0, 1.0]], dtype=np.float32)

    y1 = interpreter.run(x)
    y2 = interpreter.run(x)
    assert np.array_equal(y1, y2)
