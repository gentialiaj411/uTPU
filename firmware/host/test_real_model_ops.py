import numpy as np
import pytest

from graph_conv_ops import (
    adaptive_avg_pool2d_nchw_numpy,
    conv2d_nchw_numpy,
    fold_conv_bn_weights,
    max_pool2d_nchw_numpy,
)
from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import GraphPassManager, conv_bn_fusion_pass
from graph_reference_interpreter import execute_graph_reference


def _torch_conv(x, w, b, stride, padding, groups):
    import torch
    import torch.nn.functional as F

    xt = torch.from_numpy(x)
    wt = torch.from_numpy(w)
    bt = None if b is None else torch.from_numpy(b)
    with torch.no_grad():
        yt = F.conv2d(xt, wt, bt, stride=stride, padding=padding, groups=groups)
    return yt.numpy().astype(np.float32)


def test_conv2d_numpy_matches_torch():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 3, 8, 8), dtype=np.float32)
    w = rng.standard_normal((4, 3, 3, 3), dtype=np.float32)
    b = rng.standard_normal((4,), dtype=np.float32)
    ours = conv2d_nchw_numpy(x, w, bias=b, stride=1, padding=1, groups=1)
    ref = _torch_conv(x, w, b, stride=1, padding=1, groups=1)
    assert np.allclose(ours, ref, rtol=1e-4, atol=1e-4)


def test_max_pool_numpy_matches_torch():
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 4, 9, 9), dtype=np.float32)
    ours = max_pool2d_nchw_numpy(x, kernel_size=3, stride=2, padding=1)
    with torch.no_grad():
        ref = F.max_pool2d(torch.from_numpy(x), 3, stride=2, padding=1).numpy()
    assert np.allclose(ours, ref, rtol=1e-5, atol=1e-5)


def test_adaptive_avg_pool_numpy_matches_torch():
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(13)
    x = rng.standard_normal((1, 8, 7, 7), dtype=np.float32)
    ours = adaptive_avg_pool2d_nchw_numpy(x, output_size=(1, 1))
    with torch.no_grad():
        ref = F.adaptive_avg_pool2d(torch.from_numpy(x), (1, 1)).numpy()
    assert np.allclose(ours, ref, rtol=1e-5, atol=1e-5)


def test_conv_bn_fold_pass_and_reference_add_residual():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 3, 5, 5), dtype=np.float32)
    w = rng.standard_normal((3, 3, 3, 3), dtype=np.float32)
    b_conv = rng.standard_normal((3,), dtype=np.float32)
    gamma = np.abs(rng.standard_normal((3,), dtype=np.float32)) + 0.1
    beta = rng.standard_normal((3,), dtype=np.float32)
    mean = rng.standard_normal((3,), dtype=np.float32)
    var = np.abs(rng.standard_normal((3,), dtype=np.float32)) + 0.1

    graph = GraphIR(name="tiny_conv_bn_add")
    graph.inputs = ["x"]
    graph.outputs = ["out"]
    graph.add_value("x", shape=tuple(x.shape))
    graph.add_op(
        OpNode(
            name="conv",
            op=OpKind.CONV2D,
            inputs=["x"],
            outputs=["conv_out"],
            attrs={
                "weight": w,
                "bias": b_conv,
                "stride": (1, 1),
                "padding": (1, 1),
                "groups": 1,
            },
        )
    )
    graph.add_value("conv_out", shape=(1, 3, 5, 5), producer="conv")
    graph.add_op(
        OpNode(
            name="bn",
            op=OpKind.BATCH_NORM,
            inputs=["conv_out"],
            outputs=["bn_out"],
            attrs={
                "weight": gamma,
                "bias": beta,
                "running_mean": mean,
                "running_var": var,
                "eps": 1e-5,
            },
        )
    )
    graph.add_value("bn_out", shape=(1, 3, 5, 5), producer="bn")
    graph.values["bn_out"].consumers.append("add")
    graph.add_op(
        OpNode(name="add", op=OpKind.ADD, inputs=["x", "bn_out"], outputs=["out"], attrs={})
    )
    graph.add_value("out", shape=(1, 3, 5, 5), producer="add")

    folded = conv_bn_fusion_pass(graph)
    assert all(op.op != OpKind.BATCH_NORM for op in folded.ops)
    out = execute_graph_reference(folded, x)
    w_fold, b_fold = fold_conv_bn_weights(w, b_conv, gamma, beta, mean, var, 1e-5)
    ref = conv2d_nchw_numpy(x, w_fold, bias=b_fold, stride=1, padding=1) + x
    assert np.allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_resnet18_import_passes_legality():
    pytest.importorskip("torchvision")
    import torch
    import torchvision.models as models
    from torch.fx.passes.shape_prop import ShapeProp

    from fx_importer import import_fx_graph_module
    from graph_passes import GraphPassManager

    model = models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(x)
    graph = import_fx_graph_module(gm, name="ResNet")
    result = GraphPassManager(target_backend="cuda").run(graph)
    assert not any(op.op == OpKind.BATCH_NORM for op in result.graph.ops)
