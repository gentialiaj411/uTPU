import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_lowering import plan_blocked_fc_graph
from graph_reference_interpreter import GraphReferenceInterpreter
from requantization import RequantParams
from utpu_batched_matmul_lowering import DEFAULT_BMM_CFG, batched_matmul_int_oracle, simulate_lowered_batched_matmul_utpu


def _bmm_graph(lhs: np.ndarray, rhs: np.ndarray) -> GraphIR:
    graph = GraphIR(name="bmm_case")
    graph.inputs = ["lhs", "rhs"]
    graph.outputs = ["y"]
    graph.add_value("lhs", shape=tuple(lhs.shape), dtype="torch.float32")
    graph.add_value("rhs", shape=tuple(rhs.shape), dtype="torch.float32")
    graph.add_op(
        OpNode(
            name="bmm",
            op=OpKind.BATCHED_MATMUL,
            inputs=["lhs", "rhs"],
            outputs=["y"],
            attrs={},
        )
    )
    return graph


def test_utpu_batched_matmul_lowering_matches_reference_for_small_case():
    lhs = np.array(
        [
            [[1, -2, 3, 0], [0, 1, -1, 2]],
            [[-1, 2, 0, 1], [2, -1, 1, 0]],
        ],
        dtype=np.int8,
    )
    rhs = np.array(
        [
            [[1, 0, -1], [0, 1, 2], [1, -1, 0], [2, 0, 1]],
            [[-1, 1, 0], [1, 0, 1], [0, 2, -1], [1, 1, 0]],
        ],
        dtype=np.int8,
    )
    requant = RequantParams(multiplier=3, right_shift=1, enable=True)
    result = simulate_lowered_batched_matmul_utpu(lhs, rhs, cfg=DEFAULT_BMM_CFG, requant_params=requant)
    expected = batched_matmul_int_oracle(lhs, rhs, cfg=DEFAULT_BMM_CFG, requant_params=requant)
    assert np.array_equal(result["output"], expected)
    assert result["all_programs_bit_exact_vs_oracle"] is True
    assert result["program_count"] == 2


def test_plan_blocked_fc_graph_emits_utpu_batched_matmul_lowering_request():
    lhs = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 4.0
    rhs = (np.arange(40, dtype=np.float32).reshape(2, 4, 5) % 7) - 3.0
    graph = _bmm_graph(lhs, rhs)
    plan = plan_blocked_fc_graph(
        graph,
        array_size=16,
        activation_values={"lhs": lhs, "rhs": rhs},
        target_backend="utpu",
    )
    assert len(plan.lowered_ops) == 1
    assert not plan.fallback_ops
    lowered = plan.lowered_ops[0]
    assert lowered.op == OpKind.BATCHED_MATMUL
    assert "dynamic x dynamic" in " ".join(lowered.notes)


def test_attention_graph_plan_emits_batched_matmul_backend_lowering():
    x = np.arange(64, dtype=np.float32).reshape(2, 4, 8) / 10.0
    graph = GraphIR(name="manual_attention_core")
    graph.inputs = ["x"]
    graph.outputs = ["ctx"]
    graph.add_value("x", shape=(2, 4, 8), dtype="torch.float32")
    for name in ("q_proj", "k_proj", "v_proj"):
        graph.add_op(
            OpNode(
                name=name,
                op=OpKind.LINEAR,
                inputs=["x"],
                outputs=[f"{name}_out"],
                attrs={
                    "weight": np.ones((8, 8), dtype=np.float32),
                    "bias": None,
                    "in_features": 8,
                    "out_features": 8,
                },
            )
        )
    graph.add_op(OpNode("q_view", OpKind.VIEW, ["q_proj_out"], ["q_view"], attrs={"args": (2, 4, 2, 4)}))
    graph.add_op(OpNode("k_view", OpKind.VIEW, ["k_proj_out"], ["k_view"], attrs={"args": (2, 4, 2, 4)}))
    graph.add_op(OpNode("v_view", OpKind.VIEW, ["v_proj_out"], ["v_view"], attrs={"args": (2, 4, 2, 4)}))
    graph.add_op(OpNode("q_perm", OpKind.PERMUTE, ["q_view"], ["q_perm"], attrs={"args": (0, 2, 1, 3)}))
    graph.add_op(OpNode("k_perm", OpKind.PERMUTE, ["k_view"], ["k_perm"], attrs={"args": (0, 2, 1, 3)}))
    graph.add_op(OpNode("v_perm", OpKind.PERMUTE, ["v_view"], ["v_perm"], attrs={"args": (0, 2, 1, 3)}))
    graph.add_op(OpNode("k_t", OpKind.PERMUTE, ["k_perm"], ["k_t"], attrs={"args": (0, 1, 3, 2)}))
    graph.add_op(OpNode("qk", OpKind.BATCHED_MATMUL, ["q_perm", "k_t"], ["scores"], attrs={}))
    graph.add_op(OpNode("softmax", OpKind.SOFTMAX, ["scores"], ["probs"], attrs={}))
    graph.add_op(OpNode("av", OpKind.BATCHED_MATMUL, ["probs", "v_perm"], ["ctx_heads"], attrs={}))
    graph.add_op(OpNode("ctx_perm", OpKind.PERMUTE, ["ctx_heads"], ["ctx_perm"], attrs={"args": (0, 2, 1, 3)}))
    graph.add_op(OpNode("ctx_flat", OpKind.VIEW, ["ctx_perm"], ["ctx_flat"], attrs={"args": (2, 4, 8)}))
    graph.add_op(
        OpNode(
            name="out_proj",
            op=OpKind.LINEAR,
            inputs=["ctx_flat"],
            outputs=["out_proj_out"],
            attrs={
                "weight": np.ones((8, 8), dtype=np.float32),
                "bias": None,
                "in_features": 8,
                "out_features": 8,
            },
        )
    )
    graph.add_op(OpNode("ctx_out", OpKind.VIEW, ["out_proj_out"], ["ctx"], attrs={"args": (2, 4, 8)}))
    plan = plan_blocked_fc_graph(
        graph,
        array_size=16,
        activation_values=GraphReferenceInterpreter(graph).run_with_intermediates(x),
        target_backend="utpu",
    )
    bmm_ops = [op for op in plan.lowered_ops if op.op == OpKind.BATCHED_MATMUL]
    assert len(bmm_ops) == 2
    for op in bmm_ops:
        assert "rhs batch slice is transposed into the runtime-stationary matrix" in " ".join(op.notes)
