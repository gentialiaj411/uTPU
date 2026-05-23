import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOST_DIR = os.path.join(REPO_ROOT, "firmware", "host")
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from pytorch_compiler import compile_model
from cuda_blocked_fc_backend import detect_cuda_environment


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 32, num_heads: int = 4, mlp_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.fc1 = nn.Linear(d_model, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, d_model, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        n1 = self.norm1(x)
        q = self.q_proj(n1).view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(n1).view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(n1).view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, scale=1.0)
        attn = attn.permute(0, 2, 1, 3).reshape(b, t, self.d_model)
        x = x + self.o_proj(attn)
        n2 = self.norm2(x)
        return x + self.fc2(F.relu(self.fc1(n2)))


def main() -> int:
    env = detect_cuda_environment()
    if not env.runtime_available:
        print(json.dumps({"error": "CUDA runtime unavailable", "reason": env.reason}, indent=2))
        return 1
    torch.manual_seed(1234)
    model = TinyTransformerBlock().eval()
    x = torch.randn(2, 16, 32)
    compiled = compile_model(model, (x,), target="cuda")
    y = compiled((x,), mode="compiled")
    y_ref = model(x)
    max_abs = float(torch.max(torch.abs(y - y_ref)).item())
    max_rel = float(torch.max(torch.abs(y - y_ref) / torch.clamp(torch.abs(y_ref), min=1e-8)).item())
    payload = {
        "summary": compiled.summary(),
        "execution_report": compiled.execution_report(),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
    }
    out_dir = Path("build/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "compile_transformer_block_report.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
