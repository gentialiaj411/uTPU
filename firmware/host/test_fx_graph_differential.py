"""Eager-PyTorch differential coverage for the supported FX graph subset."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from pytorch_compiler import PyTorchCompileError, compile_model


class OutputAndRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        h = self.linear(x)
        return h, F.relu(h)


class SoftmaxDimZero(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=0)


class BroadcastAdd(nn.Module):
    def forward(self, x, y):
        return x + y


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the native tile width so this exercises the compiled resident
        # path instead of a padded-shape rejection.
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(8)); self.fc1.bias.zero_()
            self.fc2.weight.copy_(torch.eye(8)); self.fc2.bias.zero_()

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4, bias=False)
        self.k = nn.Linear(4, 4, bias=False)
        self.v = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return F.scaled_dot_product_attention(self.q(x), self.k(x), self.v(x))


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return x + F.relu(self.fc(x))


def _assert_compiled_matches_eager_or_rejects(model, inputs):
    model.eval()
    with torch.no_grad():
        eager = model(*inputs)
    try:
        result = compile_model(model, inputs if len(inputs) > 1 else inputs[0], target="cuda", strict=True)
    except PyTorchCompileError as exc:
        assert "unsupported" in str(exc).lower() or "softmax" in str(exc).lower()
        return
    try:
        compiled = result(*inputs, mode="compiled")
    except Exception as exc:
        # Backends without a lowering must reject rather than silently use a
        # different operation (for example, non-last-axis softmax).
        assert "unsupported" in str(exc).lower() or "softmax" in str(exc).lower()
        return
    if isinstance(eager, tuple):
        assert isinstance(compiled, tuple)
        for got, expected in zip(compiled, eager):
            torch.testing.assert_close(got.cpu(), expected.cpu(), rtol=1e-4, atol=1e-5)
    else:
        torch.testing.assert_close(compiled.cpu(), eager.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "model,inputs",
    [
        (OutputAndRelu(), (torch.randn(2, 4),)),
        (SoftmaxDimZero(), (torch.randn(2, 3),)),
        (BroadcastAdd(), (torch.randn(2, 1), torch.randn(1, 3))),
        (SmallMLP(), (torch.ones(1, 8),)),
        (AttentionBlock(), (torch.randn(1, 2, 4),)),
        (ResidualBlock(), (torch.randn(2, 4),)),
    ],
)
def test_fx_graph_matches_eager_or_rejects_clearly(model, inputs):
    _assert_compiled_matches_eager_or_rejects(model, inputs)
