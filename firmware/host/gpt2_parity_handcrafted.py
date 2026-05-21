import json
import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

from cuda_blocked_fc_backend import CUDABackendLowerer, CUDAGraphOpExecutor
from fx_importer import import_fx_graph_module
from graph_lowering import plan_blocked_fc_graph, quantize_weights_pass
from graph_passes import (
    backend_legality_pass,
    dead_code_elimination_pass,
    linear_relu_fusion_pass,
    memory_planning_pass,
    shape_inference_pass,
)


@dataclass
class ModelConfig:
    layers: int = 4
    hidden_dim: int = 256
    num_heads: int = 4
    head_dim: int = 64
    vocab_size: int = 1024
    max_seq_len: int = 128
    mlp_dim: int = 1536


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.rms1 = nn.RMSNorm(cfg.hidden_dim)
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.o_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.rms2 = nn.RMSNorm(cfg.hidden_dim)
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.mlp_dim)
        self.fc2 = nn.Linear(cfg.mlp_dim, cfg.hidden_dim)
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.max_seq_len = cfg.max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n1 = self.rms1(x)
        q = self.q_proj(n1).view(1, self.max_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(n1).view(1, self.max_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(n1).view(1, self.max_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).reshape(1, self.max_seq_len, self.num_heads * self.head_dim)
        x = x + self.o_proj(attn)
        n2 = self.rms2(x)
        mlp = self.fc2(F.relu(self.fc1(n2)))
        return x + mlp


class HandcraftedTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.vocab_size, cfg.hidden_dim)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.layers)])
        self.final_rms = nn.RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x_onehot: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x_onehot)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_rms(x)
        return self.lm_head(x)


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _run_pipeline(graph):
    graph = shape_inference_pass(graph)
    graph = linear_relu_fusion_pass(graph)
    graph = dead_code_elimination_pass(graph)
    graph = quantize_weights_pass(graph, group_size=64)
    graph = memory_planning_pass(graph)
    graph = backend_legality_pass(graph, backend="cuda")
    return graph


def _one_hot_tokens(tokens: List[int], vocab_size: int, seq_len: int) -> np.ndarray:
    x = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
    for i, tok in enumerate(tokens):
        x[0, i, tok] = 1.0
    return x


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    model = HandcraftedTransformer(cfg).to(device).eval()

    prompt_tokens = [11, 23, 47, 89, 13, 377, 41, 9]
    tokens_to_generate = 8
    start_len = len(prompt_tokens)

    example = torch.from_numpy(_one_hot_tokens(prompt_tokens, cfg.vocab_size, cfg.max_seq_len)).to(device)
    compile_t0 = time.perf_counter()
    traced = torch.fx.symbolic_trace(model)
    ShapeProp(traced).propagate(example)
    graph = import_fx_graph_module(traced, name="handcrafted_transformer_4l")
    graph = _run_pipeline(graph)
    compile_s = time.perf_counter() - compile_t0

    lowered = plan_blocked_fc_graph(graph, array_size=16, apply_quant=True, activation_values=None)
    lowerer = CUDABackendLowerer()
    _ = [lowerer.lower_blocked_fc(op.request) for op in lowered.lowered_ops if op.request is not None]

    exec_cuda = CUDAGraphOpExecutor(device="cuda" if torch.cuda.is_available() else "cpu")
    pt_tokens = list(prompt_tokens)
    utpu_tokens = list(prompt_tokens)
    max_abs_error = 0.0

    with torch.no_grad():
        for _step in range(tokens_to_generate):
            pt_in = torch.from_numpy(_one_hot_tokens(pt_tokens, cfg.vocab_size, cfg.max_seq_len)).to(device)
            pt_logits = model(pt_in)
            pt_next = int(torch.argmax(pt_logits[0, len(pt_tokens) - 1]).item())
            pt_tokens.append(pt_next)

            ut_in = _one_hot_tokens(utpu_tokens, cfg.vocab_size, cfg.max_seq_len)
            cuda_out = exec_cuda.run(graph, ut_in)
            if not cuda_out.get("executed", False):
                raise RuntimeError(f"CUDA graph execution failed: {cuda_out.get('reason')}")
            ut_logits = np.asarray(cuda_out["outputs"], dtype=np.float32)
            ref_slice = pt_logits.detach().cpu().numpy().astype(np.float32)
            max_abs_error = max(max_abs_error, float(np.max(np.abs(ref_slice - ut_logits))))
            ut_next = int(np.argmax(ut_logits[0, len(utpu_tokens) - 1]))
            utpu_tokens.append(ut_next)

    pt_generated = pt_tokens[start_len:]
    ut_generated = utpu_tokens[start_len:]
    matches = sum(int(a == b) for a, b in zip(pt_generated, ut_generated))
    match_rate = float(matches / tokens_to_generate)

    report = {
        "model_params": _count_params(model),
        "layers": cfg.layers,
        "hidden_dim": cfg.hidden_dim,
        "tokens_compared": tokens_to_generate,
        "token_match_rate": match_rate,
        "int4_enabled": True,
        "parity_max_abs_error": max_abs_error,
        "compile_time_s": compile_s,
        "pytorch_tokens": pt_generated,
        "utpu_tokens": ut_generated,
    }
    path = os.path.join("build", "reports", "transformer_parity.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(path)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
