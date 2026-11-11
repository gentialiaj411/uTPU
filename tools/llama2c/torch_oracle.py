"""Bridge v0 binary weights into pinned upstream llama2.c model.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from .reference import Llama2CModel


def load_torch_model(reference: Llama2CModel, device: str = "cpu"):
    vendor = Path(__file__).with_name("vendor") / "model.py"
    spec = importlib.util.spec_from_file_location("llama2c_pinned_model", vendor)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    c = reference.config
    args = module.ModelArgs(dim=c.dim, n_layers=c.n_layers, n_heads=c.n_heads,
                            n_kv_heads=c.n_kv_heads, vocab_size=c.vocab_size,
                            hidden_dim=c.hidden_dim, max_seq_len=c.seq_len,
                            dropout=0.0)
    model = module.Transformer(args)
    with torch.no_grad():
        def copy(dst, src):
            dst.copy_(torch.from_numpy(src.copy()))
        copy(model.tok_embeddings.weight, reference.token_embedding)
        for i, layer in enumerate(model.layers):
            copy(layer.attention_norm.weight, reference.rms_att[i])
            copy(layer.attention.wq.weight, reference.wq[i])
            copy(layer.attention.wk.weight, reference.wk[i])
            copy(layer.attention.wv.weight, reference.wv[i])
            copy(layer.attention.wo.weight, reference.wo[i])
            copy(layer.ffn_norm.weight, reference.rms_ffn[i])
            copy(layer.feed_forward.w1.weight, reference.w1[i])
            copy(layer.feed_forward.w2.weight, reference.w2[i])
            copy(layer.feed_forward.w3.weight, reference.w3[i])
        copy(model.norm.weight, reference.rms_final)
        if not c.shared_classifier:
            model.tok_embeddings.weight = torch.nn.Parameter(torch.from_numpy(reference.token_embedding.copy()))
            copy(model.output.weight, reference.wcls)
    model.eval().to(device)
    return model
