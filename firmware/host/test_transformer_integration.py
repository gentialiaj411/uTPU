import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cuda_blocked_fc_backend import detect_cuda_environment
from pytorch_compiler import compile_model


class AttentionOnlyBlock(nn.Module):
    def __init__(self, d: int, h: int, norm_kind: str = "rms", causal: bool = False):
        super().__init__()
        self.d = d
        self.h = h
        self.hd = d // h
        norm_cls = nn.LayerNorm if norm_kind == "layer" else nn.RMSNorm
        self.norm = norm_cls(d, elementwise_affine=False)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.causal = bool(causal)

    def forward(self, x):
        b, t, _ = x.shape
        n = self.norm(x)
        q = self.q(n).view(b, t, self.h, self.hd).permute(0, 2, 1, 3)
        k = self.k(n).view(b, t, self.h, self.hd).permute(0, 2, 1, 3)
        v = self.v(n).view(b, t, self.h, self.hd).permute(0, 2, 1, 3)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=1.0)
        y = y.permute(0, 2, 1, 3).reshape(b, t, self.d)
        return x + self.o(y)


class FullTransformerBlock(nn.Module):
    def __init__(self, d: int, h: int, mlp: int, norm_kind: str = "rms", causal: bool = False):
        super().__init__()
        self.attn = AttentionOnlyBlock(d, h, norm_kind=norm_kind, causal=causal)
        norm_cls = nn.LayerNorm if norm_kind == "layer" else nn.RMSNorm
        self.norm2 = norm_cls(d, elementwise_affine=False)
        self.fc1 = nn.Linear(d, mlp, bias=False)
        self.fc2 = nn.Linear(mlp, d, bias=False)

    def forward(self, x):
        y = self.attn(x)
        n2 = self.norm2(y)
        return y + self.fc2(F.relu(self.fc1(n2)))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.detach().cpu() - b.detach().cpu())).item())


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.abs(a.detach().cpu() - b.detach().cpu())
    den = torch.clamp(torch.abs(b.detach().cpu()), min=1e-8)
    return float(torch.max(num / den).item())


@pytest.mark.parametrize(
    "cfg",
    [
        {"B": 1, "T": 4, "D": 8, "H": 2, "MLP": 16, "norm_kind": "rms", "causal": False, "block": "attn"},
        {"B": 1, "T": 4, "D": 8, "H": 2, "MLP": 16, "norm_kind": "rms", "causal": True, "block": "attn"},
        {"B": 2, "T": 16, "D": 32, "H": 4, "MLP": 64, "norm_kind": "layer", "causal": False, "block": "full"},
        {"B": 2, "T": 16, "D": 32, "H": 4, "MLP": 64, "norm_kind": "rms", "causal": True, "block": "full"},
    ],
)
def test_transformer_compiled_cuda_no_fallback_parity(cfg):
    env = detect_cuda_environment()
    if not env.runtime_available:
        pytest.skip(f"CUDA runtime unavailable: {env.reason}")

    torch.manual_seed(1234)
    if cfg["block"] == "attn":
        model = AttentionOnlyBlock(cfg["D"], cfg["H"], norm_kind=cfg["norm_kind"], causal=cfg["causal"]).eval()
    else:
        model = FullTransformerBlock(cfg["D"], cfg["H"], cfg["MLP"], norm_kind=cfg["norm_kind"], causal=cfg["causal"]).eval()
    x = torch.randn(cfg["B"], cfg["T"], cfg["D"])
    compiled = compile_model(model, (x,), target="cuda")
    assert compiled.ok
    y_compiled = compiled((x,), mode="compiled")
    y_ref = model(x)
    max_abs = _max_abs(y_compiled, y_ref)
    max_rel = _max_rel(y_compiled, y_ref)
    report = compiled.execution_report()
    assert report["pytorch_fallback_ops"] == 0
    assert report["numpy_fallback_ops"] == 0
    assert report["fallback_ops"] == []
    assert max_abs <= 3e-3
    assert max_rel <= 3e-2

    out_dir = Path("build/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transformer_parity_report.json"
    payload = {"shape": cfg, "max_abs_error": max_abs, "max_rel_error": max_rel, "report": report}
    items = []
    if out_path.exists():
        try:
            items = json.loads(out_path.read_text(encoding="utf-8")).get("cases", [])
        except Exception:
            items = []
    items.append(payload)
    out_path.write_text(json.dumps({"seed": 1234, "cases": items}, indent=2), encoding="utf-8")
