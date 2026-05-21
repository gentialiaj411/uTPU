import json
import os

import numpy as np

from cuda_blocked_fc_backend import CUDAGraphOpExecutor
from graph_ir import GraphIR, OpKind, OpNode
from graph_lowering import quantize_weights_pass
from graph_reference_interpreter import execute_graph_reference


def test_int4_quantization_parity():
    rng = np.random.default_rng(7)
    in_features = 128
    out_features = 96
    x = rng.standard_normal((2, in_features), dtype=np.float32)
    w = rng.standard_normal((out_features, in_features), dtype=np.float32) * 0.5
    b = rng.standard_normal((out_features,), dtype=np.float32) * 0.1

    graph = GraphIR(name="int4_quant")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_op(
        OpNode(
            name="fc",
            op=OpKind.LINEAR,
            inputs=["x"],
            outputs=["y"],
            attrs={
                "weight": w,
                "bias": b,
                "in_features": in_features,
                "out_features": out_features,
            },
        )
    )

    qgraph = quantize_weights_pass(graph, group_size=64)
    ref = execute_graph_reference(qgraph, x)
    cuda = CUDAGraphOpExecutor(device="cuda").run(qgraph, x)
    assert cuda.get("executed", False), cuda.get("reason")
    out = cuda["outputs"]
    np.testing.assert_allclose(ref, out, atol=5e-3, rtol=5e-3)
    max_abs = float(np.max(np.abs(ref - out)))

    fp16_bytes = int(w.astype(np.float16).nbytes)
    int4_bytes = int(np.asarray(qgraph.ops[0].attrs["weight_int4_packed"]).nbytes + np.asarray(qgraph.ops[0].attrs["weight_int4_scales"]).nbytes)
    reduction = 100.0 * (1.0 - (float(int4_bytes) / float(fp16_bytes)))

    report_path = os.path.join("build", "reports", "int4_quantization_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_weight_bytes_fp16": fp16_bytes,
                "model_weight_bytes_int4": int4_bytes,
                "reduction_percent": reduction,
                "parity_max_abs_error": max_abs,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    assert reduction >= 70.0
