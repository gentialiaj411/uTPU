"""Float-enabled diagnostic harness; never imported by the integer runtime."""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch

from tools.llama2c.reference import Llama2CModel
from tools.llama2c.torch_oracle import load_torch_model

from .export import export_model
from .runtime import IntegerLlama


def _qdq_weight(value, qmax=127):
    peak = torch.amax(torch.abs(value), dim=1, keepdim=True)
    scale = torch.where(peak == 0, torch.ones_like(peak), peak / qmax)
    return torch.clamp(torch.round(value / scale), -qmax, qmax) * scale


def _fq(value, scale):
    return torch.clamp(torch.round(value / scale), -128, 127) * scale


class DiagnosticModel:
    """Vectorized fake-quant model used only to rank component contributions."""

    def __init__(self, reference, scales, device, *, weights_quant=True,
                 activations_quant=True, float_component=None, dynamic_activations=False):
        self.c, self.s, self.device = reference.config, scales, device
        self.aq = activations_quant
        self.float_component = float_component
        self.dynamic_activations = dynamic_activations
        self.runq_style = float_component == "runq_style"
        tensor = lambda value: torch.from_numpy(np.asarray(value).copy()).float().to(device)
        weight = lambda value: _qdq_weight(tensor(value)) if weights_quant else tensor(value)
        self.embedding_float = tensor(reference.token_embedding)
        self.embedding = (self.embedding_float if dynamic_activations else
                          (_fq(self.embedding_float, scales["x"]) if weights_quant or activations_quant else self.embedding_float))
        self.rms_att, self.rms_ffn = tensor(reference.rms_att), tensor(reference.rms_ffn)
        self.rms_final = tensor(reference.rms_final)
        self.layers = []
        for i in range(self.c.n_layers):
            self.layers.append({name: weight(getattr(reference, name)[i]) for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3")})
        self.wcls = weight(reference.wcls)
        luts = json.loads((Path(__file__).with_name("luts") / "luts.json").read_text(encoding="utf-8"))
        self.rope_cos = torch.tensor(luts["rope_cos_q15"]["values"], device=device).float() / 32768.0
        self.rope_sin = torch.tensor(luts["rope_sin_q15"]["values"], device=device).float() / 32768.0
        self.silu_lut = torch.tensor(luts["silu_i8"]["values"], device=device).float() * scales["silu"]
        # The ablation freezes the pre-fix 1/16, [-8, 0] exp table even after
        # the production LUT is regenerated at higher resolution.
        self.exp_lut = torch.round(torch.exp(-torch.arange(129, device=device).float() / 16.0) * 32767.0) / 32767.0

    def q(self, value, name):
        if not self.aq:
            return value
        if self.float_component == f"activation_{name}" or (self.float_component == "activation_qkv" and name in ("q", "k", "v")) or (self.float_component == "activation_ffn" and name in ("h1", "h3", "hidden")):
            return value
        if self.dynamic_activations:
            groups = []
            group_size = int(self.dynamic_activations) if not isinstance(self.dynamic_activations, bool) else value.shape[-1]
            if name == "logits":
                group_size = value.shape[-1]
            for begin in range(0, value.shape[-1], group_size):
                block = value[..., begin:begin + group_size]
                peak = torch.amax(torch.abs(block), dim=-1, keepdim=True)
                scale = torch.where(peak == 0, torch.ones_like(peak), peak / 127.0)
                groups.append(torch.clamp(torch.round(block / scale), -128, 127) * scale)
            return torch.cat(groups, dim=-1)
        return _fq(value, self.s[name])

    def rms(self, x, gamma):
        if self.float_component == "rmsnorm" or not self.aq or self.dynamic_activations:
            out = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * gamma
        else:
            code = torch.clamp(torch.round(x / self.s["x"]), -128, 127)
            gamma_q14 = torch.clamp(torch.round(gamma * 16384.0), -32768, 32767) / 16384.0
            out = code * torch.rsqrt(torch.mean(code * code, dim=-1, keepdim=True) + 1.0) * gamma_q14
        return self.q(out, "norm")

    def rope(self, value, positions, scale):
        shape = value.shape
        pair = value.reshape(shape[:-1] + (-1, 2))
        if self.float_component == "rope_lut" or not self.aq:
            index = torch.arange(pair.shape[-2], device=self.device) % (self.c.head_size // 2)
            angle = positions[:, None] / (10000.0 ** (2.0 * index / self.c.head_size))
            cos, sin = torch.cos(angle), torch.sin(angle)
        else:
            index = torch.arange(pair.shape[-2], device=self.device) % self.rope_cos.shape[1]
            cos, sin = self.rope_cos[positions[:, None], index], self.rope_sin[positions[:, None], index]
        out = torch.stack((pair[..., 0] * cos - pair[..., 1] * sin,
                           pair[..., 0] * sin + pair[..., 1] * cos), dim=-1).reshape(shape)
        return _fq(out, scale) if self.aq else out

    def __call__(self, tokens):
        if self.runq_style:
            return self._runq(tokens)
        c, s = self.c, self.s
        batch, t = tokens.shape
        positions = torch.arange(t, device=self.device)
        x = self.q(self.embedding[tokens], "x")
        causal = torch.triu(torch.ones((t, t), dtype=torch.bool, device=self.device), 1)
        for i, w in enumerate(self.layers):
            residual = x
            norm = self.rms(x, self.rms_att[i])
            q_real, k_real, v_real = norm @ w["wq"].T, norm @ w["wk"].T, norm @ w["wv"].T
            q = self.rope(self.q(q_real, "q"), positions, s["q"])
            k_input = k_real if self.float_component == "kv_cache_quant" else self.q(k_real, "k")
            v = v_real if self.float_component == "kv_cache_quant" else self.q(v_real, "v")
            k = self.rope(k_input, positions, s["k"])
            q = q.view(batch, t, c.n_heads, c.head_size).transpose(1, 2)
            k = k.view(batch, t, c.n_kv_heads, c.head_size).transpose(1, 2).repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
            v = v.view(batch, t, c.n_kv_heads, c.head_size).transpose(1, 2).repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
            score = q @ k.transpose(-2, -1) / math.sqrt(c.head_size)
            score = _fq(score, 1.0 / 16.0) if self.aq else score
            score = score.masked_fill(causal, float("-inf"))
            if self.float_component == "softmax_exp_lut" or not self.aq:
                probability = torch.softmax(score, dim=-1)
            else:
                delta_code = torch.clamp(torch.round((score - torch.amax(score, dim=-1, keepdim=True)) * 16.0), -128, 0).long()
                exponent = self.exp_lut[-delta_code]
                exponent = exponent.masked_fill(causal, 0.0)
                probability = exponent / torch.sum(exponent, dim=-1, keepdim=True)
            if self.float_component != "attention_prob_q15" and self.aq:
                probability = torch.round(probability * 32767.0) / 32767.0
            av = (probability @ v).transpose(1, 2).reshape(batch, t, c.dim)
            av = self.q(av, "v")
            x = self.q(residual + av @ w["wo"].T, "x")
            residual = x
            norm = self.rms(x, self.rms_ffn[i])
            h1, h3 = self.q(norm @ w["w1"].T, "h1"), self.q(norm @ w["w3"].T, "h3")
            if self.float_component == "silu_lut" or not self.aq:
                silu = torch.nn.functional.silu(h1)
            else:
                code = torch.clamp(torch.round(h1 / s["h1"]), -128, 127).long()
                silu = self.silu_lut[code + 128]
            hidden = self.q(silu * h3, "hidden")
            x = self.q(residual + hidden @ w["w2"].T, "x")
        norm = self.rms(x, self.rms_final)
        logits = norm @ self.wcls.T
        if self.aq and self.float_component != "logits_quant":
            logits = self.q(logits, "logits")
        return logits

    def _runq(self, tokens):
        c = self.c
        batch, t = tokens.shape
        positions = torch.arange(t, device=self.device)
        x = self.embedding_float[tokens]
        causal = torch.triu(torch.ones((t, t), dtype=torch.bool, device=self.device), 1)
        for i, w in enumerate(self.layers):
            residual = x
            norm = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * self.rms_att[i]
            norm = self.q(norm, "norm")
            q, k, v = norm @ w["wq"].T, norm @ w["wk"].T, norm @ w["wv"].T
            shape_q = q.view(batch, t, c.n_heads, c.head_size)
            shape_k = k.view(batch, t, c.n_kv_heads, c.head_size)
            index_q = torch.arange(c.n_heads * c.head_size // 2, device=self.device) % (c.head_size // 2)
            index_k = torch.arange(c.n_kv_heads * c.head_size // 2, device=self.device) % (c.head_size // 2)
            angle_q = positions[:, None] / (10000.0 ** (2.0 * index_q / c.head_size))
            angle_k = positions[:, None] / (10000.0 ** (2.0 * index_k / c.head_size))
            def rotate(value, angle):
                pair = value.reshape(batch, t, -1, 2)
                return torch.stack((pair[..., 0] * torch.cos(angle) - pair[..., 1] * torch.sin(angle),
                                    pair[..., 0] * torch.sin(angle) + pair[..., 1] * torch.cos(angle)), -1).reshape(value.shape)
            q, k = rotate(q, angle_q), rotate(k, angle_k)
            q = q.view(batch, t, c.n_heads, c.head_size).transpose(1, 2)
            k = k.view(batch, t, c.n_kv_heads, c.head_size).transpose(1, 2).repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
            v = v.view(batch, t, c.n_kv_heads, c.head_size).transpose(1, 2).repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
            score = (q @ k.transpose(-2, -1) / math.sqrt(c.head_size)).masked_fill(causal, float("-inf"))
            av = (torch.softmax(score, -1) @ v).transpose(1, 2).reshape(batch, t, c.dim)
            x = residual + self.q(av, "v") @ w["wo"].T
            residual = x
            norm = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * self.rms_ffn[i]
            norm = self.q(norm, "norm")
            h1, h3 = norm @ w["w1"].T, norm @ w["w3"].T
            hidden = torch.nn.functional.silu(h1) * h3
            x = residual + self.q(hidden, "hidden") @ w["w2"].T
        norm = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5) * self.rms_final
        return self.q(norm, "norm") @ self.wcls.T


def metric_accumulator():
    return {"nll": 0.0, "kl": 0.0, "matches": 0, "tokens": 0}


def add_metrics(acc, float_logits, candidate_logits, targets):
    float_logp = torch.log_softmax(float_logits.double(), dim=-1)
    candidate_logp = torch.log_softmax(candidate_logits.double(), dim=-1)
    probs = torch.exp(float_logp)
    acc["nll"] += float((-candidate_logp.gather(-1, targets[..., None]).sum()).cpu())
    acc["kl"] += float(torch.sum(probs * (float_logp - candidate_logp)).cpu())
    acc["matches"] += int(torch.sum(torch.argmax(float_logits, dim=-1) == torch.argmax(candidate_logits, dim=-1)).cpu())
    acc["tokens"] += targets.numel()


def finish(acc):
    mean = acc["nll"] / acc["tokens"]
    return {"eval_tokens": acc["tokens"], "mean_nll": mean, "perplexity": math.exp(mean),
            "teacher_forced_top1_agreement_pct": 100.0 * acc["matches"] / acc["tokens"],
            "logit_kl_float_to_candidate": acc["kl"] / acc["tokens"]}


def run_ablation(reference, scales, ids, lut_dir, device="cuda"):
    float_model = load_torch_model(reference, device)
    variants = [
        ("float", DiagnosticModel(reference, scales, device, weights_quant=False, activations_quant=False)),
        ("int8_weights_only", DiagnosticModel(reference, scales, device, weights_quant=True, activations_quant=False)),
        ("quantized_activations_only", DiagnosticModel(reference, scales, device, weights_quant=False, activations_quant=True)),
        ("full_int8_diagnostic", DiagnosticModel(reference, scales, device)),
    ]
    for component in ("activation_quant", "rmsnorm", "rope_lut", "silu_lut", "softmax_exp_lut", "kv_cache_quant", "attention_prob_q15", "logits_quant"):
        if component == "activation_quant":
            model = DiagnosticModel(reference, scales, device, weights_quant=True, activations_quant=False)
        else:
            model = DiagnosticModel(reference, scales, device, float_component=component)
        variants.append((f"full_int8_swap_{component}_to_float", model))
    accumulators = {name: metric_accumulator() for name, _ in variants}
    seq_len = reference.config.seq_len
    windows = [ids[start:min(start + seq_len + 1, len(ids))] for start in range(0, len(ids) - 1, seq_len)]
    batches = []
    for length in sorted({len(item) for item in windows}, reverse=True):
        same = [item for item in windows if len(item) == length]
        batches.extend(same[i:i + 8] for i in range(0, len(same), 8))
    with torch.inference_mode():
        for chunks in batches:
            x = torch.tensor([item[:-1] for item in chunks], dtype=torch.long, device=device)
            y = torch.tensor([item[1:] for item in chunks], dtype=torch.long, device=device)
            float_logits = float_model(x, y)
            for name, model in variants:
                add_metrics(accumulators[name], float_logits, model(x), y)
    rows = [{"name": name, **finish(accumulators[name])} for name, _ in variants]
    full = next(row for row in rows if row["name"] == "full_int8_diagnostic")
    for row in rows:
        row["kl_reduction_vs_full_int8"] = full["logit_kl_float_to_candidate"] - row["logit_kl_float_to_candidate"]
    ranking = sorted((row for row in rows if row["name"].startswith("full_int8_swap_")),
                     key=lambda row: row["kl_reduction_vs_full_int8"], reverse=True)
    return rows, [row["name"] for row in ranking]


def main():
    from tools.llama2c.metrics import tokenize_eval
    from tools.llama2c.tokenizer import Tokenizer

    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "build/reports/numerics_v2_ablation_pre_fix.json")
    args = parser.parse_args()
    base = root / "models/llama2c"
    reference = Llama2CModel(base / "stories260K/stories260K.bin")
    scales = json.loads((Path(__file__).with_name("calibration_260k.json")).read_text(encoding="utf-8"))["activation_scales"]
    ids = tokenize_eval(Tokenizer(base / "stories260K/tok512.bin", 512), base / "eval_slice.txt")
    rows, ranking = run_ablation(reference, scales, ids, Path(__file__).with_name("luts"))
    result = {"method": "separate float-enabled fake-quant diagnostic; production integer runtime unchanged",
              "rows": rows, "ranking_by_kl_reduction": ranking}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
