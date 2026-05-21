import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_compiler import compile_mlp_model


class TinyTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, seq_len: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.seq_len = seq_len
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim, bias=False)

    def forward(self, x):
        n1 = self.norm1(x)
        q = self.q_proj(n1).view(1, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(n1).view(1, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(n1).view(1, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        attn = attn.permute(0, 2, 1, 3).reshape(1, self.seq_len, self.hidden_dim)
        x = x + self.o_proj(attn)
        n2 = self.norm2(x)
        return x + self.fc2(F.relu(self.fc1(n2)))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.detach().cpu() - b.detach().cpu())).item())


def test_transformer_graph_compiles_without_runtime_unsupported_ops():
    torch.manual_seed(7)
    model = TinyTransformerBlock(hidden_dim=32, num_heads=4, mlp_dim=64, seq_len=8).eval()
    x = torch.randn(1, 8, 32)
    compiled = compile_mlp_model(model, x, target="cuda")

    assert compiled.import_error is None
    assert compiled.runtime_plan is not None
    assert compiled.runtime_plan.unsupported_ops == []
    assert compiled.plan is not None
    assert compiled.plan.unsupported_ops == []
    assert compiled.ok is True


def test_transformer_parity_matrix_small_shapes():
    torch.manual_seed(9)
    shape_matrix = [
        {"seq_len": 8, "hidden_dim": 32, "num_heads": 4, "mlp_dim": 64},
        {"seq_len": 12, "hidden_dim": 48, "num_heads": 6, "mlp_dim": 96},
    ]
    max_errors = []

    with torch.no_grad():
        for cfg in shape_matrix:
            model = TinyTransformerBlock(
                hidden_dim=cfg["hidden_dim"],
                num_heads=cfg["num_heads"],
                mlp_dim=cfg["mlp_dim"],
                seq_len=cfg["seq_len"],
            ).eval()
            x = torch.randn(1, cfg["seq_len"], cfg["hidden_dim"])
            compiled = compile_mlp_model(model, x, target="cuda")
            y_compiled = compiled(x, mode="compiled")
            y_ref = model(x)
            err = _max_abs(y_compiled, y_ref)
            max_errors.append(err)
            assert err <= 1e-4

    assert max(max_errors) <= 1e-4
