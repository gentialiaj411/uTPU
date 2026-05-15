import importlib.util


def _require_torch_or_skip():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch
    import torch.nn as nn
    from torch.fx.passes.shape_prop import ShapeProp
    return torch, nn, ShapeProp


def _trace_with_shapes(model, example):
    import torch

    gm = torch.fx.symbolic_trace(model)
    from torch.fx.passes.shape_prop import ShapeProp

    if isinstance(example, tuple):
        ShapeProp(gm).propagate(*example)
    else:
        ShapeProp(gm).propagate(example)
    return gm


def test_import_simple_mlp():
    torch_info = _require_torch_or_skip()
    if torch_info is None:
        return
    torch, nn, _ = torch_info

    from fx_importer import import_fx_graph_module
    from graph_ir import OpKind

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    gm = _trace_with_shapes(model, torch.randn(1, 4))
    graph = import_fx_graph_module(gm, name="tiny_mlp")

    assert [op.op for op in graph.ops] == [OpKind.LINEAR, OpKind.RELU, OpKind.LINEAR]
    assert graph.values[graph.inputs[0]].shape == (1, 4)
    assert graph.ops[0].attrs["in_features"] == 4
    assert graph.ops[0].attrs["out_features"] == 3
    assert graph.values[graph.ops[0].outputs[0]].shape == (1, 3)
    assert graph.values[graph.ops[0].outputs[0]].dtype == "torch.float32"
    assert graph.ops[1].op == OpKind.RELU


def test_import_add_function():
    torch_info = _require_torch_or_skip()
    if torch_info is None:
        return
    torch, nn, _ = torch_info

    from fx_importer import import_fx_graph_module
    from graph_ir import OpKind

    class AddModel(nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    gm = _trace_with_shapes(AddModel(), (torch.randn(1, 4), torch.randn(1, 4)))
    graph = import_fx_graph_module(gm, name="add_model")

    assert len(graph.ops) == 1
    assert graph.ops[0].op == OpKind.ADD
    assert graph.values[graph.ops[0].outputs[0]].shape == (1, 4)


def test_unsupported_op_fails_clearly():
    torch_info = _require_torch_or_skip()
    if torch_info is None:
        return
    torch, nn, _ = torch_info

    from fx_importer import FXImportError, import_fx_graph_module

    class SigmoidModel(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    gm = _trace_with_shapes(SigmoidModel(), torch.randn(1, 4))
    try:
        import_fx_graph_module(gm, name="bad_model")
        raise AssertionError("Expected unsupported sigmoid import to fail")
    except FXImportError as e:
        assert "Unsupported call_function node" in str(e)
        assert "sigmoid" in str(e)


def run_all():
    if importlib.util.find_spec("torch") is None:
        print("test_fx_importer: SKIP (PyTorch not installed)")
        return
    test_import_simple_mlp()
    test_import_add_function()
    test_unsupported_op_fails_clearly()
    print("test_fx_importer: PASS")


if __name__ == "__main__":
    run_all()
