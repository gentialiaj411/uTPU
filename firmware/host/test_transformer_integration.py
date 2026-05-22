import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_compiler import compile_model


class TinyTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        seq_len: int,
        causal: bool = False,
        affine_norm: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.seq_len = seq_len
        self.norm1 = nn.RMSNorm(hidden_dim, elementwise_affine=bool(affine_norm))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm2 = nn.RMSNorm(hidden_dim, elementwise_affine=bool(affine_norm))
        self.fc1 = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim, bias=False)
        self.causal = bool(causal)

    def forward(self, x, attn_mask=None):
        n1 = self.norm1(x)
        q = self.q_proj(n1).view(1, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(n1).view(1, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(n1).view(1, self.seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.causal)
        attn = attn.permute(0, 2, 1, 3).reshape(1, self.seq_len, self.hidden_dim)
        x = x + self.o_proj(attn)
        n2 = self.norm2(x)
        return x + self.fc2(F.relu(self.fc1(n2)))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.detach().cpu() - b.detach().cpu())).item())


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.abs(a.detach().cpu() - b.detach().cpu())
    den = torch.clamp(torch.abs(b.detach().cpu()), min=1e-8)
    return float(torch.max(num / den).item())


def _max_rel_nonzero_ref(a: torch.Tensor, b: torch.Tensor, ref_floor: float = 1e-3) -> float:
    num = torch.abs(a.detach().cpu() - b.detach().cpu())
    ref = torch.abs(b.detach().cpu())
    mask = ref >= ref_floor
    if not torch.any(mask):
        return 0.0
    den = torch.clamp(ref[mask], min=ref_floor)
    return float(torch.max(num[mask] / den).item())


def _write_parity_report(cases):
    report_dir = Path("build/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for c in cases if c["within_tolerance"])
    payload = {
        "total_cases": len(cases),
        "passed_cases": passed,
        "pass_rate": (passed / len(cases)) if cases else 0.0,
        "cases": cases,
    }
    out = report_dir / "transformer_parity_report.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_transformer_graph_compiles_without_runtime_unsupported_ops():
    torch.manual_seed(7)
    model = TinyTransformerBlock(hidden_dim=32, num_heads=4, mlp_dim=64, seq_len=8).eval()
    x = torch.randn(1, 8, 32)
    attn_mask = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    compiled = compile_model(model, (x, attn_mask), target="cuda")

    assert compiled.import_error is None
    assert compiled.runtime_plan is not None
    assert compiled.runtime_plan.unsupported_ops == []
    assert compiled.plan is not None
    assert compiled.plan.unsupported_ops == []
    assert compiled.ok is True


def test_transformer_parity_matrix_small_shapes():
    torch.manual_seed(9)
    shape_matrix = [
        {"seq_len": 8, "hidden_dim": 32, "num_heads": 4, "mlp_dim": 64, "causal": False, "mask_variant": "4d_full", "affine_norm": False},
        {"seq_len": 16, "hidden_dim": 32, "num_heads": 4, "mlp_dim": 64, "causal": False, "mask_variant": "2d_broadcast", "affine_norm": False},
        {"seq_len": 24, "hidden_dim": 48, "num_heads": 6, "mlp_dim": 96, "causal": False, "mask_variant": "3d_batch", "affine_norm": False},
        {"seq_len": 32, "hidden_dim": 64, "num_heads": 8, "mlp_dim": 128, "causal": False, "mask_variant": "4d_head_broadcast", "affine_norm": False},
        {"seq_len": 48, "hidden_dim": 64, "num_heads": 8, "mlp_dim": 128, "causal": False, "mask_variant": "4d_full", "affine_norm": False},
        {"seq_len": 16, "hidden_dim": 32, "num_heads": 4, "mlp_dim": 64, "causal": False, "mask_variant": "4d_full", "affine_norm": True},
    ]
    tol_abs = 1e-4
    tol_rel = 3e-3
    tol_rel_any = 1e-2
    report_cases = []

    with torch.no_grad():
        for cfg in shape_matrix:
            model = TinyTransformerBlock(
                hidden_dim=cfg["hidden_dim"],
                num_heads=cfg["num_heads"],
                mlp_dim=cfg["mlp_dim"],
                seq_len=cfg["seq_len"],
                causal=cfg["causal"],
                affine_norm=cfg["affine_norm"],
            ).eval()
            x = torch.randn(1, cfg["seq_len"], cfg["hidden_dim"])
            seq_len = cfg["seq_len"]
            num_heads = cfg["num_heads"]
            if cfg["mask_variant"] == "2d_broadcast":
                attn_mask = torch.zeros((seq_len, seq_len), dtype=torch.float32)
            elif cfg["mask_variant"] == "3d_batch":
                attn_mask = torch.zeros((1, seq_len, seq_len), dtype=torch.float32)
            elif cfg["mask_variant"] == "4d_head_broadcast":
                attn_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
            else:
                attn_mask = torch.zeros((1, num_heads, seq_len, seq_len), dtype=torch.float32)
            if attn_mask is not None:
                attn_mask[..., -1] = -1e4
            compiled = compile_model(model, (x, attn_mask), target="cuda")
            y_compiled = compiled((x, attn_mask), mode="compiled")
            y_ref = model(x, attn_mask=attn_mask)
            max_abs = _max_abs(y_compiled, y_ref)
            max_rel = _max_rel(y_compiled, y_ref)
            max_rel_nonzero_ref = _max_rel_nonzero_ref(y_compiled, y_ref, ref_floor=1e-3)
            case_ok = max_abs <= tol_abs and max_rel_nonzero_ref <= tol_rel and max_rel <= tol_rel_any
            report_cases.append(
                {
                    "config": cfg,
                    "max_abs_error": max_abs,
                    "max_rel_error": max_rel,
                    "max_rel_error_nonzero_ref": max_rel_nonzero_ref,
                    "within_tolerance": case_ok,
                }
            )
            assert case_ok, (
                f"cfg={cfg} max_abs={max_abs} max_rel={max_rel} "
                f"max_rel_nonzero_ref={max_rel_nonzero_ref}"
            )

    _write_parity_report(report_cases)
