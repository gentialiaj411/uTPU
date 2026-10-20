"""Generate the committed stories260K numerics-v2 metric artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.llama2c.metrics import tokenize_eval
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer

from .export import accumulator_bounds, export_model
from .runtime import IntegerLlama

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def perplexity(model, ids, logit_scale):
    total, count = 0.0, 0
    seq_len = int(model.cfg["seq_len"])
    for start in range(0, len(ids) - 1, seq_len):
        chunk = ids[start:min(start + seq_len + 1, len(ids))]
        if len(chunk) < 2:
            continue
        logits = model.forward_sequence(chunk[:-1]).astype(np.float64) * logit_scale
        target = np.asarray(chunk[1:], dtype=np.int64)
        maximum = np.max(logits, axis=1)
        total += float(np.sum(maximum + np.log(np.sum(np.exp(logits - maximum[:, None]), axis=1)) - logits[np.arange(target.size), target]))
        count += target.size
    mean = total / count
    return {"eval_tokens": count, "mean_nll": mean, "perplexity": math.exp(mean)}


def prompts_from(ids, count=64, length=8):
    span = len(ids) - length - 128
    return [ids[(i * span) // count:(i * span) // count + length] for i in range(count)]


def golden_continuations(checkpoint, prompts, steps=128):
    golden = Llama2CModel(checkpoint)
    all_tokens = []
    for prompt in prompts:
        golden.reset()
        logits = None
        for pos, token in enumerate(prompt):
            logits = golden.forward(token, pos)
        generated = []
        for step in range(steps):
            token = int(logits.argmax())
            generated.append(token)
            logits = golden.forward(token, len(prompt) + step)
        all_tokens.append(generated)
    return all_tokens


def agreement(model, prompts, golden, steps=128):
    same, first = 0, []
    for prompt, expected in zip(prompts, golden):
        model.reset()
        logits = None
        for pos, token in enumerate(prompt):
            logits = model.forward(token, pos)
        divergence = None
        for step in range(steps):
            token = int(logits.argmax())
            same += token == expected[step]
            if divergence is None and token != expected[step]:
                divergence = step
            logits = model.forward(token, len(prompt) + step)
        first.append(divergence)
    observed = [item for item in first if item is not None]
    return {"prompts": len(prompts), "tokens_per_prompt": steps,
            "identical_tokens_pct_vs_float": 100.0 * same / (len(prompts) * steps),
            "first_divergence_position": first,
            "first_divergence_min": min(observed) if observed else None}


def evaluate_config(reference, scales, ids, prompts, golden, bits, group, improved):
    exported = export_model(reference, scales, bits, group, improved)
    model = IntegerLlama(exported, HERE / "luts")
    row = {"weight_bits": bits, "group_size": None if bits == 8 else group,
           "weight_quantizer": "mse_optimal_clipping" if improved else "symmetric_maxabs",
           "accumulator_bounds": accumulator_bounds(reference, bits, group)}
    row.update(perplexity(model, ids, scales["logits"]))
    row["greedy_agreement"] = agreement(model, prompts, golden)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "bench/results/numerics_v2_260k.json")
    parser.add_argument("--refresh-bounds-only", action="store_true")
    args = parser.parse_args()
    base = ROOT / "models/llama2c"
    checkpoint = base / "stories260K/stories260K.bin"
    reference = Llama2CModel(checkpoint)
    if args.refresh_bounds_only:
        result = json.loads(args.output.read_text(encoding="utf-8"))
        rows = result["configs"] + result.get("improvement", {}).get("configs", [])
        for row in rows:
            row["accumulator_bounds"] = accumulator_bounds(reference, row["weight_bits"], row["group_size"] or 32)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return
    tokenizer = Tokenizer(base / "stories260K/tok512.bin", reference.config.vocab_size)
    ids = tokenize_eval(tokenizer, base / "eval_slice.txt")
    prompts = prompts_from(ids)
    golden = golden_continuations(checkpoint, prompts)
    calibration = json.loads((HERE / "calibration_260k.json").read_text(encoding="utf-8"))
    scales = calibration["activation_scales"]
    baseline = [evaluate_config(reference, scales, ids, prompts, golden, 8, 32, False)]
    baseline += [evaluate_config(reference, scales, ids, prompts, golden, 4, group, False) for group in (32, 64, 128)]
    int8_ppl = baseline[0]["perplexity"]
    large = any(row["perplexity"] - int8_ppl > 0.25 or row["greedy_agreement"]["identical_tokens_pct_vs_float"] < 90.0 for row in baseline[1:])
    improved = [evaluate_config(reference, scales, ids, prompts, golden, 4, group, True) for group in (32, 64, 128)] if large else []
    result = {"schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "model": "stories260K", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "eval": {"tokens": 50000, "slice_sha256": "ae55d9eb37d0e99c6e1051dbeb212f51b6c1d2d98858fccd81f5676bee92a4e7"},
              "calibration": calibration["metadata"], "configs": baseline,
              "int4_loss_large_rule": "ppl delta vs INT8 > 0.25 or greedy agreement < 90%",
              "int4_loss_large": large, "improvement": {"name": "MSE-optimal clipping per group", "configs": improved}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"configs": [{"bits": r["weight_bits"], "group": r["group_size"], "quantizer": r["weight_quantizer"],
                                   "ppl": r["perplexity"], "agreement": r["greedy_agreement"]["identical_tokens_pct_vs_float"]}
                                  for r in baseline + improved]}, indent=2))


if __name__ == "__main__":
    main()
