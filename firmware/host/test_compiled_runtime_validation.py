import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from compiled_runtime import CompiledRuntimeError
from pytorch_compiler import compile_model


class TinyTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, seq_len: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.seq_len = seq_len
        self.norm1 = nn.RMSNorm(hidden_dim, elementwise_affine=False)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm2 = nn.RMSNorm(hidden_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim, bias=False)

    def forward(self, x, attn_mask=None):
        n1 = self.norm1(x)
        q = self.q_proj(n1).view(1, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(n1).view(1, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(n1).view(1, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        attn = attn.permute(0, 2, 1, 3).reshape(1, self.seq_len, self.hidden_dim)
        x = x + self.o_proj(attn)
        n2 = self.norm2(x)
        return x + self.fc2(F.relu(self.fc1(n2)))


def _compile_case(seq_len=8, hidden_dim=32, num_heads=4, mlp_dim=64):
    torch.manual_seed(11)
    model = TinyTransformerBlock(hidden_dim, num_heads, mlp_dim, seq_len).eval()
    x = torch.randn(1, seq_len, hidden_dim)
    mask = torch.zeros((1, num_heads, seq_len, seq_len), dtype=torch.float32)
    compiled = compile_model(model, (x, mask), target="cuda")
    return compiled, x, mask


def test_runtime_validation_rejects_invalid_permute_axes():
    compiled, x, mask = _compile_case()
    permute_op = next(op for op in compiled.graph_ir.ops if op.op == "permute")
    original = tuple(permute_op.attrs.get("args", ()))
    permute_op.attrs["args"] = (0, 1, 1, 3)
    with pytest.raises(CompiledRuntimeError, match="Invalid permute"):
        compiled((x, mask), mode="compiled")
    permute_op.attrs["args"] = original


def test_runtime_validation_rejects_invalid_attention_mask_rank():
    compiled, x, _ = _compile_case()
    bad_mask = torch.zeros((1, 4, 8, 8, 1), dtype=torch.float32)
    with pytest.raises(CompiledRuntimeError, match="mask rank must be 2/3/4"):
        compiled((x, bad_mask), mode="compiled")


def test_runtime_validation_rejects_invalid_attention_mask_shape():
    compiled, x, _ = _compile_case()
    bad_mask = torch.zeros((1, 4, 7, 8), dtype=torch.float32)
    with pytest.raises(CompiledRuntimeError, match="must end with"):
        compiled((x, bad_mask), mode="compiled")
