import json
import os

import numpy as np
import pytest

from cuda_blocked_fc_backend import CUDAGraphOpExecutor
from graph_ir import GraphIR, OpKind, OpNode
from graph_reference_interpreter import execute_graph_reference


def _build_attention_graph() -> GraphIR:
    graph = GraphIR(name="attention_op_validation")
    graph.inputs = ["q", "k", "v", "mask"]
    graph.outputs = ["norm_out"]

    graph.add_op(
        OpNode(
            name="attn",
            op=OpKind.SCALED_DOT_PRODUCT_ATTENTION,
            inputs=["q", "k", "v", "mask"],
            outputs=["attn_out"],
            attrs={"num_heads": 4, "head_dim": 16, "causal_mask": False},
        )
    )
    graph.add_op(
        OpNode(
            name="softmax",
            op=OpKind.SOFTMAX,
            inputs=["attn_out"],
            outputs=["prob_out"],
            attrs={},
        )
    )
    graph.add_op(
        OpNode(
            name="rmsnorm",
            op=OpKind.LAYER_NORM,
            inputs=["prob_out"],
            outputs=["norm_out"],
            attrs={"eps": 1e-5},
        )
    )
    return graph


def test_attention_reference_vs_cuda():
    rng = np.random.default_rng(42)
    b, h, t, d = 2, 4, 8, 16
    q = rng.standard_normal((b, h, t, d), dtype=np.float32)
    k = rng.standard_normal((b, h, t, d), dtype=np.float32)
    v = rng.standard_normal((b, h, t, d), dtype=np.float32)
    mask = np.zeros((b, h, t, t), dtype=np.float32)

    graph = _build_attention_graph()
    ref = execute_graph_reference(graph, q, k, v, mask)
    cuda_result = CUDAGraphOpExecutor(device="cuda").run(graph, q, k, v, mask)
    cuda_available = bool(cuda_result.get("executed", False))

    max_abs_error = None
    if not cuda_available:
        pytest.skip(f"CUDA path unavailable: {cuda_result.get('reason')}")

    out = cuda_result["outputs"]
    max_abs_error = float(np.max(np.abs(ref - out)))
    np.testing.assert_allclose(ref, out, atol=1e-3, rtol=1e-3)

    report_path = os.path.join("build", "reports", "attention_op_validation.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "shapes_tested": [[b, h, t, d]],
                "max_abs_error": max_abs_error,
                "cuda_available": cuda_available,
            },
            f,
            indent=2,
            sort_keys=True,
        )
