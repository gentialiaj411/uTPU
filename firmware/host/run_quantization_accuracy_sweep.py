import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

from cuda_blocked_fc_backend import CUDAGraphOpExecutor
from fx_importer import import_fx_graph_module
from graph_lowering import quantize_weights_pass
from graph_reference_interpreter import execute_graph_reference
from graph_passes import (
    backend_legality_pass,
    dead_code_elimination_pass,
    linear_relu_fusion_pass,
    memory_planning_pass,
    shape_inference_pass,
)

SEED = 20260523
N_PROMPTS = 20
K_TOKENS = 16
MODEL_NAME = "handcrafted_transformer_4l"
REPORT_PATH = os.path.join("build", "reports", "quantization_accuracy_sweep.json")


class ModelConfig:
    layers = 4
    hidden_dim = 256
    num_heads = 4
    head_dim = 64
    vocab_size = 1024
    max_seq_len = 128
    mlp_dim = 1536


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


def _prompt_bank() -> List[List[int]]:
    return [
        [11, 23, 47, 89, 13, 377, 41, 9],
        [5, 19, 22, 35, 77, 101, 88, 66],
        [99, 12, 300, 44, 20, 18, 17, 16],
        [2, 4, 8, 16, 32, 64, 128, 256],
        [511, 510, 509, 508, 507, 506, 505, 504],
        [3, 14, 15, 92, 65, 35, 89, 79],
        [1, 1, 2, 3, 5, 8, 13, 21],
        [42, 43, 44, 45, 46, 47, 48, 49],
        [7, 70, 700, 70, 7, 700, 70, 7],
        [255, 0, 255, 0, 255, 0, 255, 0],
        [120, 121, 122, 123, 124, 125, 126, 127],
        [900, 901, 902, 903, 904, 905, 906, 907],
        [13, 26, 39, 52, 65, 78, 91, 104],
        [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
        [17, 34, 68, 136, 272, 544, 33, 66],
        [73, 19, 73, 19, 73, 19, 73, 19],
        [31, 62, 93, 124, 155, 186, 217, 248],
        [400, 401, 450, 451, 500, 501, 550, 551],
        [9, 81, 729, 81, 9, 81, 729, 9],
        [0, 1023, 0, 1023, 0, 1023, 0, 1023],
    ]


def _hash_outputs(outputs: List[List[int]]) -> str:
    payload = json.dumps(outputs, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = ModelConfig()
    prompts = _prompt_bank()
    if len(prompts) != N_PROMPTS:
        raise RuntimeError(f"Expected {N_PROMPTS} prompts, got {len(prompts)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandcraftedTransformer(cfg).to(device).eval()

    example = torch.from_numpy(_one_hot_tokens(prompts[0], cfg.vocab_size, cfg.max_seq_len)).to(device)
    traced = torch.fx.symbolic_trace(model)
    ShapeProp(traced).propagate(example)
    graph = import_fx_graph_module(traced, name=MODEL_NAME)
    graph = _run_pipeline(graph)
    executor = CUDAGraphOpExecutor(device="cuda" if torch.cuda.is_available() else "cpu")
    int4_execution_backend = "cuda_graph_executor"

    fp32_outputs: List[List[int]] = []
    int4_outputs: List[List[int]] = []
    per_prompt_top1_match: List[float] = []

    with torch.no_grad():
        for prompt in prompts:
            fp32_tokens = list(prompt)
            int4_tokens = list(prompt)
            fp32_generated: List[int] = []
            int4_generated: List[int] = []
            matches = 0

            for _ in range(K_TOKENS):
                fp32_in = torch.from_numpy(_one_hot_tokens(fp32_tokens, cfg.vocab_size, cfg.max_seq_len)).to(device)
                fp32_logits = model(fp32_in)
                fp32_next = int(torch.argmax(fp32_logits[0, len(fp32_tokens) - 1]).item())
                fp32_tokens.append(fp32_next)
                fp32_generated.append(fp32_next)

                int4_in = _one_hot_tokens(int4_tokens, cfg.vocab_size, cfg.max_seq_len)
                int4_out = executor.run(graph, int4_in)
                if int4_out.get("executed", False):
                    int4_logits = np.asarray(int4_out["outputs"], dtype=np.float32)
                else:
                    int4_execution_backend = "graph_reference_interpreter_fallback"
                    int4_logits = np.asarray(execute_graph_reference(graph, int4_in), dtype=np.float32)
                int4_next = int(np.argmax(int4_logits[0, len(int4_tokens) - 1]))
                int4_tokens.append(int4_next)
                int4_generated.append(int4_next)

                if fp32_next == int4_next:
                    matches += 1

            fp32_outputs.append(fp32_generated)
            int4_outputs.append(int4_generated)
            per_prompt_top1_match.append(float(matches / K_TOKENS))

    aggregate_top1_match_rate = float(sum(per_prompt_top1_match) / len(per_prompt_top1_match))
    histogram: Dict[str, int] = {}
    for rate in per_prompt_top1_match:
        bucket = f"{rate:.4f}"
        histogram[bucket] = histogram.get(bucket, 0) + 1

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_name": MODEL_NAME,
        "n_prompts": N_PROMPTS,
        "k_tokens_per_prompt": K_TOKENS,
        "seed": SEED,
        "prompt_token_sequences": prompts,
        "fp32_outputs_hash": _hash_outputs(fp32_outputs),
        "int4_outputs_hash": _hash_outputs(int4_outputs),
        "aggregate_top1_match_rate": aggregate_top1_match_rate,
        "per_prompt_top1_match": per_prompt_top1_match,
        "per_prompt_top1_match_histogram": histogram,
        "quantization_config": {
            "enabled": True,
            "weight_bits": 4,
            "group_size": 64,
            "path": "quantize_weights_pass",
            "execution_backend": int4_execution_backend,
        },
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(REPORT_PATH)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
