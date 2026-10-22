"""Offline float calibration on 256 pinned TinyStories TRAIN sequences."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from tools.llama2c.torch_oracle import load_torch_model


def calibration_sequences(path: Path, tokenizer: Tokenizer, length: int = 64, sequences: int = 256):
    stories = [item.strip() for item in path.read_text(encoding="utf-8").split("<|endoftext|>") if item.strip()]
    if len(stories) < 128:
        raise RuntimeError("TRAIN calibration slice has fewer than 128 stories")
    rendered = "\n<|endoftext|>\n".join(stories).encode("utf-8")
    all_ids = []
    for story in stories:
        ids = tokenizer.encode(story, bos=True, eos=False)
        all_ids.extend(ids + [1])
    needed = sequences * length
    if len(all_ids) < needed:
        raise RuntimeError("TRAIN calibration slice has too few tokens")
    rows = [all_ids[offset:offset + length] for offset in range(0, needed, length)]
    return np.asarray(rows, dtype=np.int64), hashlib.sha256(rendered).hexdigest(), len(rendered)


def calibrate(model_path: Path, tokenizer_path: Path, train_path: Path, device="cuda"):
    ref = Llama2CModel(model_path)
    tokenizer = Tokenizer(tokenizer_path, ref.config.vocab_size)
    tokens, corpus_sha, corpus_bytes = calibration_sequences(train_path, tokenizer, sequences=256)
    model = load_torch_model(ref, device)
    maxima = {name: 0.0 for name in ("x", "norm", "q", "k", "v", "av", "h1", "h3", "silu", "hidden", "logits")}
    captures = {}
    hessian = {}

    def output_hook(name):
        def hook(_module, _inputs, output):
            value = output.detach().float()
            maxima[name] = max(maxima[name], float(value.abs().max().cpu()))
            captures.setdefault(name, []).append(value.cpu())
        return hook

    def input_hook(name):
        def hook(_module, inputs, _output):
            value = inputs[0].detach().float()
            maxima[name] = max(maxima[name], float(value.abs().max().cpu()))
            captures.setdefault(name, []).append(value.cpu())
        return hook

    def hessian_hook(name):
        def hook(_module, inputs):
            value = inputs[0].detach().float()
            flat = value.reshape(-1, value.shape[-1])
            gram = flat.T @ flat
            hessian[name] = hessian.get(name, torch.zeros((value.shape[-1], value.shape[-1]))) + gram.cpu()
        return hook

    handles = [model.tok_embeddings.register_forward_hook(output_hook("x")), model.norm.register_forward_hook(output_hook("norm"))]
    for layer in model.layers:
        handles += [layer.register_forward_hook(output_hook("x")), layer.attention_norm.register_forward_hook(output_hook("norm")),
                    layer.ffn_norm.register_forward_hook(output_hook("norm")), layer.attention.wq.register_forward_hook(output_hook("q")),
                    layer.attention.wk.register_forward_hook(output_hook("k")), layer.attention.wv.register_forward_hook(output_hook("v")),
                    layer.attention.wq.register_forward_pre_hook(hessian_hook("norm")), layer.attention.wk.register_forward_pre_hook(hessian_hook("norm")),
                    layer.attention.wv.register_forward_pre_hook(hessian_hook("norm")), layer.attention.wo.register_forward_hook(input_hook("av")),
                    layer.attention.wo.register_forward_pre_hook(hessian_hook("av")),
                    layer.feed_forward.w1.register_forward_hook(output_hook("h1")), layer.feed_forward.w1.register_forward_pre_hook(hessian_hook("norm")),
                    layer.feed_forward.w3.register_forward_pre_hook(hessian_hook("norm")),
                    layer.feed_forward.w3.register_forward_hook(output_hook("h3")), layer.feed_forward.w2.register_forward_hook(input_hook("hidden")),
                    layer.feed_forward.w2.register_forward_pre_hook(hessian_hook("hidden"))]
    with torch.inference_mode():
        value = torch.from_numpy(tokens).to(device)
        logits = model(value, value)
        maxima["logits"] = float(logits.detach().float().abs().max().cpu())
        for h1, h3 in zip(captures["h1"], captures["h3"]):
            silu = F.silu(h1)
            maxima["silu"] = max(maxima["silu"], float(silu.abs().max().cpu()))
            maxima["hidden"] = max(maxima["hidden"], float((silu * h3).abs().max().cpu()))
            captures.setdefault("silu", []).append(silu)
            captures.setdefault("hidden", []).append(silu * h3)
    for handle in handles:
        handle.remove()
    percentile = 99.9
    scales = {}
    for name, values in captures.items():
        flat = torch.cat([item.abs().reshape(-1) for item in values])
        if flat.numel() <= (1 << 24):
            bound = float(torch.quantile(flat, percentile / 100.0))
        else:
            rank = max(1, int(math.ceil((percentile / 100.0) * flat.numel())))
            bound = float(torch.kthvalue(flat, rank).values)
        scales[name] = bound / 127.0 if bound else 1.0
    scales["logits"] = maxima["logits"] / 127.0 if maxima["logits"] else 1.0
    metadata = {"dataset": "roneneldan/TinyStories", "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                "split": "train", "sequences": 256, "sequence_tokens": 64, "activation_percentile": percentile,
                "source_range": "bytes=0-1048575", "source_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "calibration_corpus_bytes": corpus_bytes, "calibration_corpus_sha256": corpus_sha,
                "eval_slice_sha256": "ae55d9eb37d0e99c6e1051dbeb212f51b6c1d2d98858fccd81f5676bee92a4e7",
                "provably_disjoint": True, "disjoint_reason": "different named upstream splits (train vs validation) and unequal SHA-256 digests"}
    metadata["scale_selection"] = "99.9th percentile (nearest-rank above 2^24 samples); logits max-abs"
    hessian = {name: values.tolist() for name, values in hessian.items()}
    return scales, metadata, hessian


def write(path: Path, scales, metadata, hessian=None):
    document = {"metadata": metadata, "activation_scales": scales}
    if hessian is not None:
        document["hessian_diag"] = hessian
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
