import importlib.util


def _require_torch_or_skip():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch
    import torch.nn as nn

    return torch, nn


class TinyMLP:
    @staticmethod
    def make(nn):
        return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def test_fx_import_produces_graph_ir_with_at_least_two_ops_for_two_layer_mlp():
    torch_info = _require_torch_or_skip()
    if torch_info is None:
        return
    torch, nn = torch_info
    from fx_importer import import_fx_graph_module

    model = TinyMLP.make(nn).eval()
    gm = torch.fx.symbolic_trace(model)
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(gm).propagate(torch.randn(1, 4))
    graph = import_fx_graph_module(gm, name="tiny_mlp_smoke")
    assert len(graph.ops) >= 2


def test_graph_lowering_produces_exactly_two_blocked_fc_requests_for_tiny_mlp():
    torch_info = _require_torch_or_skip()
    if torch_info is None:
        return
    torch, nn = torch_info
    from pytorch_compiler import compile_mlp_model

    model = TinyMLP.make(nn).eval()
    result = compile_mlp_model(model, torch.randn(1, 4), target="utpu", array_size=16)
    assert result.plan is not None
    assert len(result.plan.lowered_ops) == 2


def test_create_backend_lowerer_cuda_returns_object():
    from backend_lowering import create_backend_lowerer

    lowerer = create_backend_lowerer("cuda")
    assert lowerer is not None


def test_create_backend_lowerer_utpu_returns_object():
    from backend_lowering import create_backend_lowerer

    lowerer = create_backend_lowerer("utpu")
    assert lowerer is not None
