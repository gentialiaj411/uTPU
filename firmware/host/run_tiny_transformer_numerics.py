"""Train, train-calibrate, and evaluate the Phase-1 tiny transformer."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from requantization import choose_multiplier_and_shift
from tiny_transformer_export import (
    choose_activation_exponent,
    export_schema_document,
    linear_record,
    quantize_activation,
    quantize_weight_per_output_channel,
    write_export_schema,
)
from tiny_transformer_model import TinyDecoder
from tiny_transformer_reference import TinyTransformerConfig, integer_decoder_logits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _fixture(name: str) -> dict:
    return json.loads((HERE / "fixtures" / name).read_text(encoding="utf-8"))


def _source_tree() -> dict[str, str]:
    return {
        path: subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True).strip()
        for path in ("firmware/host", "tools")
    }


def _pow2_quant(values: np.ndarray, exponent: int | None = None) -> tuple[np.ndarray, int]:
    e = choose_activation_exponent(values) if exponent is None else int(exponent)
    return quantize_activation(values, e), e


def _record_ratio(ratio: float, *, input_exponent: int, output_exponent: int) -> dict:
    m, shift = choose_multiplier_and_shift(ratio)
    return {
        "input_exponent": int(input_exponent),
        "output_exponent": int(output_exponent),
        "multipliers": [int(m)],
        "right_shifts": [int(shift)],
    }


def _trace_maxima(model: TinyDecoder, data: torch.Tensor, split: dict, cfg: TinyTransformerConfig) -> dict[str, float]:
    """Max calibration from deterministic train windows only (percentile=100)."""
    maxima: dict[str, float] = {}
    starts = np.linspace(split["train_start"], split["train_end_exclusive"] - cfg.T - 1, num=256, dtype=np.int64)
    with torch.no_grad():
        for batch_starts in np.array_split(starts, 16):
            x = torch.stack([data[int(s):int(s) + cfg.T] for s in batch_starts])
            _, trace = model.forward_with_trace(x)
            for name, value in trace.items():
                array = value.detach().cpu().numpy()
                array = array[np.isfinite(array)]
                if array.size:
                    maxima[name] = max(maxima.get(name, 0.0), float(np.max(np.abs(array))))
    if not maxima:
        raise RuntimeError("empty train calibration trace")
    return maxima


def _build_export(model: TinyDecoder, maxima: dict[str, float]) -> tuple[dict[str, np.ndarray], dict]:
    state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    qstate: dict[str, np.ndarray] = {}
    # Lookup/table values use one power-of-two tensor scale so VPOS_ADD has a
    # meaningful exponent. Linear matrices use the required per-output scales.
    qstate["token.weight"], token_e = _pow2_quant(state["token.weight"])
    qstate["pos.weight"], pos_e = _pow2_quant(state["pos.weight"])

    def exp(name: str) -> int:
        return choose_activation_exponent(np.asarray([maxima[name]], dtype=np.float64))

    acts: dict[str, int] = {
        "token": token_e,
        "position": pos_e,
        "pos_add": max(exp("pos_add"), token_e, pos_e),
        "rmsnorm1": exp("rmsnorm1"),
        "scores": exp("scores"),
        "softmax_probs": -8,  # UINT8 p represents p/255; metadata remains explicit.
        "av": exp("av"),
        "out_proj": exp("out_proj"),
        "rmsnorm2": exp("rmsnorm2"),
        "ffn1": exp("ffn1"),
        "ffn2": exp("ffn2"),
        "final_norm": exp("final_norm"),
        "logits": exp("logits"),
    }
    acts["qkv"] = choose_activation_exponent(np.asarray([max(maxima["q"], maxima["k"], maxima["v"])], dtype=np.float64))
    acts["q"] = acts["k"] = acts["v"] = acts["qkv"]
    acts["leaky_relu"] = acts["ffn1"]  # positive branch is identity; negative branch is exact /4.
    acts["residual1"] = max(exp("residual1"), acts["pos_add"], acts["out_proj"])
    acts["residual2"] = max(exp("residual2"), acts["residual1"], acts["ffn2"])

    def qlinear(name: str, input_e: int, output_e: int) -> dict:
        key = {
            "qkv": "blocks.0.qkv.weight", "out": "blocks.0.out.weight",
            "ff1": "blocks.0.ff1.weight", "ff2": "blocks.0.ff2.weight",
        }[name]
        codes, scales = quantize_weight_per_output_channel(state[key])
        qstate[key] = codes
        return linear_record(input_exponent=input_e, weight_scales=scales, output_exponent=output_e)

    linears = {
        "qkv": qlinear("qkv", acts["rmsnorm1"], acts["qkv"]),
        "out": qlinear("out", acts["av"], acts["out_proj"]),
        "ff1": qlinear("ff1", acts["rmsnorm2"], acts["ffn1"]),
        "ff2": qlinear("ff2", acts["leaky_relu"], acts["ffn2"]),
        "scores": _record_ratio(2.0 ** (acts["q"] + acts["k"] - 2 - acts["scores"]), input_exponent=acts["q"], output_exponent=acts["scores"]),
        "av": _record_ratio(2.0 ** (acts["v"] - acts["av"]) / 255.0, input_exponent=acts["v"], output_exponent=acts["av"]),
    }
    # Tied output uses the shared lookup codes, hence its uniform token scale.
    linears["logits"] = linear_record(
        input_exponent=acts["final_norm"],
        weight_scales=np.full(state["token.weight"].shape[0], 2.0 ** token_e, dtype=np.float64),
        output_exponent=acts["logits"],
    )

    norms = {}
    for label, key, out_name in (
        ("n1", "blocks.0.n1.weight", "rmsnorm1"),
        ("n2", "blocks.0.n2.weight", "rmsnorm2"),
        ("final", "final.weight", "final_norm"),
    ):
        gamma_e = choose_activation_exponent(state[key])
        # gamma_shift=e_out-e_gamma must be non-negative in the RTL formula.
        acts[out_name] = max(acts[out_name], gamma_e)
        qstate[key], gamma_e = _pow2_quant(state[key], acts[out_name])
        norms[label] = {
            "input_exponent": acts["pos_add"] if label == "n1" else (acts["residual1"] if label == "n2" else acts["residual2"]),
            "output_exponent": acts[out_name],
            "gamma_exponent": gamma_e,
            "gamma_shift": acts[out_name] - gamma_e,
        }
    residuals = {
        "pos_add": {"ea": token_e, "eb": pos_e, "eo": acts["pos_add"]},
        "residual1": {"ea": acts["pos_add"], "eb": acts["out_proj"], "eo": acts["residual1"]},
        "residual2": {"ea": acts["residual1"], "eb": acts["ffn2"], "eo": acts["residual2"]},
    }
    schema = export_schema_document(
        calibration={"split": "train", "percentile": 100.0, "method": "deterministic_256_window_max"},
        activations=acts, linears=linears, norms=norms, residuals=residuals,
    )["payload"]
    return qstate, schema


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("build/tiny_transformer/input.txt"))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--output", type=Path, default=Path("bench/results/tiny_transformer_numerics.json"))
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.use_deterministic_algorithms(True)
    vocab, split = _fixture("tiny_transformer_vocab.json"), _fixture("tiny_transformer_split.json")
    if hashlib.sha256(a.corpus.read_bytes()).hexdigest() != vocab["corpus_sha256"]:
        raise SystemExit("corpus SHA-256 mismatch")
    stoi = {c: i for i, c in enumerate(vocab["characters"])}
    data = torch.tensor([stoi[c] for c in a.corpus.read_text(encoding="utf-8")], dtype=torch.long)
    cfg = TinyTransformerConfig(vocab=len(stoi))
    model = TinyDecoder(cfg.T, cfg.D, cfg.H, cfg.Dff, cfg.L, cfg.vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    curve = []
    for step in range(a.steps):
        starts = torch.randint(split["train_start"], split["train_end_exclusive"] - cfg.T - 1, (a.batch_size,))
        x = torch.stack([data[s:s + cfg.T] for s in starts]); y = torch.stack([data[s + 1:s + cfg.T + 1] for s in starts])
        opt.zero_grad(); loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab), y.reshape(-1)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 100 == 0:
            curve.append({"step": step + 1, "loss": float(loss.detach())})
    model.eval()
    maxima = _trace_maxima(model, data, split, cfg)
    qstate, schema = _build_export(model, maxima)
    total_loss = correct = count = 0
    with torch.no_grad():
        for start in range(split["validation_start"], split["validation_end_exclusive"] - cfg.T - 1, cfg.T):
            x = data[start:start + cfg.T][None]; y = data[start + 1:start + cfg.T + 1][None]; logits = model(x)
            total_loss += float(F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1), reduction="sum"))
            correct += int((logits.argmax(-1) == y).sum()); count += cfg.T
    iloss = icorrect = icount = 0
    saturation_counts = {"position_add": 0, "residual_attention": 0, "residual_ffn": 0}
    starts = list(range(split["validation_start"], split["validation_end_exclusive"] - cfg.T - 1, cfg.T))
    for offset in range(0, len(starts), 8):
        group = starts[offset:offset + 8]
        x = np.stack([data[s:s + cfg.T].numpy() for s in group]); y = np.stack([data[s + 1:s + cfg.T + 1].numpy() for s in group])
        logits, counts = integer_decoder_logits(x, qstate, schema, cfg)
        logits_t, y_t = torch.from_numpy(logits.astype(np.float32)), torch.from_numpy(y)
        iloss += float(F.cross_entropy(logits_t.reshape(-1, cfg.vocab), y_t.reshape(-1), reduction="sum"))
        icorrect += int((logits_t.argmax(-1) == y_t).sum()); icount += int(y.size)
        for key, value in counts.items():
            saturation_counts[key] += int(value)
    source_tree = _source_tree()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    export_path = a.output.with_name("tiny_transformer_export.json")
    write_export_schema(export_path, export_schema_document(
        calibration=schema["calibration"], activations=schema["activations"], linears=schema["linears"], norms=schema["norms"], residuals=schema["residuals"],
    ))
    payload = {
        "status": "phase1_independent_scale_aware_integer_reference", "seed": a.seed, "config": cfg.__dict__,
        "parameter_count": cfg.parameter_count, "corpus_sha256": vocab["corpus_sha256"], "split": split,
        "training_steps": a.steps, "plateau": False, "loss_curve": curve, "source_tree": source_tree,
        "export_schema_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
        "float_validation": {"loss": total_loss / count, "bits_per_char": total_loss / count / np.log(2), "top1_accuracy": correct / count, "tokens": count},
        "integer_validation": {"loss": iloss / icount, "bits_per_char": iloss / icount / np.log(2), "top1_accuracy": icorrect / icount, "tokens": icount},
        "degradation": {"loss": iloss / icount - total_loss / count, "bits_per_char": (iloss / icount - total_loss / count) / np.log(2), "top1_accuracy": icorrect / icount - correct / count},
        "saturation_counts": saturation_counts,
        "limits": ["independent NumPy integer reference; ISA-simulator and RTL equality are Phase 2/3 work"],
    }
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save({"state_dict": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, "qstate": qstate, "export_schema": schema, "config": cfg.__dict__}, a.output.with_suffix(".pt"))
    print(json.dumps(payload["degradation"], indent=2))


if __name__ == "__main__":
    main()
