import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_lowering import plan_blocked_fc_graph
from graph_passes import supported_ops_for_backend
from graph_reference_interpreter import GraphReferenceInterpreter
from pytorch_compiler import compile_model
from requantization import RequantParams
from utpu_conv2d_lowering import conv2d_im2col_int_oracle, simulate_lowered_conv2d_utpu


def _conv_graph():
    x = np.array(
        [[[[1, 0, -1, 1], [0, 1, 1, 0], [-1, 1, 0, 1], [1, 0, 1, -1]]]],
        dtype=np.float32,
    )
    w = np.array(
        [
            [[[1, 0, -1], [0, 1, 0], [1, 0, -1]]],
            [[[0, 1, 0], [1, -1, 1], [0, 1, 0]]],
        ],
        dtype=np.float32,
    )
    graph = GraphIR(name="conv_case")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_value("x", shape=tuple(x.shape), dtype="torch.float32")
    graph.add_op(
        OpNode(
            name="conv",
            op=OpKind.CONV2D,
            inputs=["x"],
            outputs=["y"],
            attrs={
                "weight": w,
                "bias": None,
                "stride": (1, 1),
                "padding": (1, 1),
                "dilation": (1, 1),
                "groups": 1,
            },
        )
    )
    return graph, x


def test_utpu_supported_ops_include_conv2d_and_batched_matmul():
    ops = supported_ops_for_backend("utpu")
    assert OpKind.CONV2D in ops
    assert OpKind.BATCHED_MATMUL in ops


def test_plan_blocked_fc_graph_emits_utpu_conv_lowering_request():
    graph, x = _conv_graph()
    plan = plan_blocked_fc_graph(
        graph,
        array_size=16,
        activation_values={"x": x},
        target_backend="utpu",
    )
    assert len(plan.lowered_ops) == 1
    assert not plan.fallback_ops
    assert plan.lowered_ops[0].op == OpKind.CONV2D
    assert "im2col" in " ".join(plan.lowered_ops[0].notes)


def test_utpu_conv2d_lowering_matches_reference_for_small_case():
    graph, x = _conv_graph()
    conv = graph.ops[0]
    result = simulate_lowered_conv2d_utpu(
        x,
        conv.attrs["weight"],
        bias=None,
        stride=conv.attrs["stride"],
        padding=conv.attrs["padding"],
        dilation=conv.attrs["dilation"],
        groups=conv.attrs["groups"],
        requant_params=RequantParams(multiplier=1, right_shift=0, enable=True),
    )
    expected = np.asarray(GraphReferenceInterpreter(graph).run(x), dtype=np.float32)
    assert np.array_equal(result["output"], expected)
    assert result["all_programs_bit_exact_vs_oracle"] is True


def test_compile_model_utpu_conv_emits_conv_backend_lowering():
    import torch
    import torch.nn as nn

    class TinyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = TinyConv()
    with torch.no_grad():
        model.conv.weight.copy_(
            torch.tensor(
                [
                    [[[1, 0, -1], [0, 1, 0], [1, 0, -1]]],
                    [[[0, 1, 0], [1, -1, 1], [0, 1, 0]]],
                ],
                dtype=torch.float32,
            )
        )
    x = torch.tensor([[[[1, 0, -1, 1], [0, 1, 1, 0], [-1, 1, 0, 1], [1, 0, 1, -1]]]], dtype=torch.float32)
    compiled = compile_model(model.eval(), x, target="utpu")
    conv_ops = [op for op in compiled.backend_ops if op.op == OpKind.CONV2D]
    assert len(conv_ops) == 1
    assert conv_ops[0].lowering["mode"] == "utpu_conv2d_im2col"
    assert conv_ops[0].lowering["program_count"] >= 1


def test_utpu_conv2d_lowering_supports_non_identity_requant_with_leaky_relu():
    x = np.array([[[[-4, -2, 0, 2], [1, -1, 1, -1], [2, 0, -2, 1], [0, 1, 0, -1]]]], dtype=np.float32)
    w = np.array([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]], dtype=np.float32)
    result = simulate_lowered_conv2d_utpu(
        x,
        w,
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        apply_relu=True,
        requant_params=RequantParams(multiplier=3, right_shift=1, enable=True),
    )
    expected = conv2d_im2col_int_oracle(
        x,
        w,
        stride=(1, 1),
        padding=(1, 1),
        groups=1,
        apply_relu=True,
        requant_params=RequantParams(multiplier=3, right_shift=1, enable=True),
    )
    assert np.array_equal(result["output"], expected)
