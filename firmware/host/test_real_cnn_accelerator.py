import numpy as np

from graph_ir import OpKind
from requantization import RequantParams
from run_real_cnn_accelerator import (
    QuantizedLayer,
    QuantizedModel,
    _build_shared_graph,
    _run_integer_oracle_graph,
)


def _dummy_model() -> QuantizedModel:
    conv1_w = np.ones((16, 3, 3, 3), dtype=np.int8)
    conv2_w = np.ones((32, 16, 3, 3), dtype=np.int8)
    conv3_w = np.ones((64, 32, 3, 3), dtype=np.int8)
    conv4_w = np.ones((64, 64, 3, 3), dtype=np.int8)
    fc_w = np.ones((10, 64 * 8 * 8), dtype=np.int8)
    rq = RequantParams(multiplier=3, right_shift=1, enable=True)
    return QuantizedModel(
        input_scale=0.1,
        logits_scale=0.2,
        conv1=QuantizedLayer("conv1", conv1_w, 0.1, 0.2, rq, stride=(1, 1), padding=(1, 1), apply_relu=True),
        conv2=QuantizedLayer("conv2", conv2_w, 0.1, 0.2, rq, stride=(2, 2), padding=(1, 1), apply_relu=True),
        conv3=QuantizedLayer("conv3", conv3_w, 0.1, 0.2, rq, stride=(2, 2), padding=(1, 1), apply_relu=True),
        conv4=QuantizedLayer("conv4", conv4_w, 0.1, 0.2, rq, stride=(1, 1), padding=(1, 1), apply_relu=True),
        fc=QuantizedLayer("fc", fc_w, 0.1, 0.2, rq, apply_relu=False),
    )


def test_shared_graph_contains_four_convs_view_linear():
    graph = _build_shared_graph(_dummy_model())
    assert [op.op for op in graph.ops] == [
        OpKind.CONV2D,
        OpKind.CONV2D,
        OpKind.CONV2D,
        OpKind.CONV2D,
        OpKind.VIEW,
        OpKind.LINEAR,
    ]
    assert graph.ops[0].attrs["requant"]["multiplier"] == 3
    assert graph.ops[-1].attrs["out_features"] == 10


def test_integer_oracle_graph_handles_non_identity_linear_requant():
    graph = _build_shared_graph(_dummy_model())
    x = np.zeros((1, 3, 32, 32), dtype=np.int8)
    logits = _run_integer_oracle_graph(graph, [x])
    assert logits.shape == (1, 10)
    assert logits.dtype == np.int8
