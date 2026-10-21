"""Generate the committed stories260K numerics-v2 metric artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from tools.llama2c.metrics import tokenize_eval
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from tools.llama2c.torch_oracle import load_torch_model

from .ablation_260k import run_ablation
from .export import accumulator_bounds, export_runq_model
from .runq_runtime import RunQIntegerLlama

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def float_eval_windows(reference, ids, device="cuda"):
    model = load_torch_model(reference, device)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(ids) - 1, reference.config.seq_len):
            chunk = ids[start:min(start + reference.config.seq_len + 1, len(ids))]
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device)[None]
            y = torch.tensor(chunk[1:], dtype=torch.long, device=device)[None]
            rows.append(model(x, y).cpu().numpy()[0])
    return rows


def teacher_forced_metrics(model, ids, base_logit_scale, float_windows):
    total_nll, total_kl, matches, count = 0.0, 0.0, 0, 0
    seq_len = int(model.cfg["seq_len"])
    for window_index, start in enumerate(range(0, len(ids) - 1, seq_len)):
        chunk = ids[start:min(start + seq_len + 1, len(ids))]
        if len(chunk) < 2:
            continue
        codes = model.forward_sequence(chunk[:-1]).astype(np.float64)
        if getattr(model, "logits_are_base_units", False):
            logits = codes * base_logit_scale
        else:
            logits = codes * (base_logit_scale * model.last_logit_scale_multipliers.astype(np.float64) / 128.0)[:, None]
        float_logits = float_windows[window_index].astype(np.float64)
        target = np.asarray(chunk[1:], dtype=np.int64)
        maximum = np.max(logits, axis=1)
        candidate_logsum = maximum + np.log(np.sum(np.exp(logits - maximum[:, None]), axis=1))
        total_nll += float(np.sum(candidate_logsum - logits[np.arange(target.size), target]))
        float_max = np.max(float_logits, axis=1)
        float_logsum = float_max + np.log(np.sum(np.exp(float_logits - float_max[:, None]), axis=1))
        float_logp = float_logits - float_logsum[:, None]
        candidate_logp = logits - candidate_logsum[:, None]
        probability = np.exp(float_logp)
        total_kl += float(np.sum(probability * (float_logp - candidate_logp)))
        matches += int(np.sum(np.argmax(float_logits, axis=1) == np.argmax(logits, axis=1)))
        count += target.size
    mean = total_nll / count
    return {"eval_tokens": count, "mean_nll": mean, "perplexity": math.exp(mean),
            "teacher_forced_top1_agreement_pct": 100.0 * matches / count,
            "logit_kl_float_to_candidate": total_kl / count}


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
    first = []
    for prompt, expected in zip(prompts, golden):
        model.reset()
        logits = None
        for pos, token in enumerate(prompt):
            logits = model.forward(token, pos)
        divergence = None
        for step in range(steps):
            token = int(logits.argmax())
            if token != expected[step]:
                divergence = step
                break
            logits = model.forward(token, len(prompt) + step)
        first.append(divergence)
    censored = [item if item is not None else steps for item in first]
    return {"prompts": len(prompts), "tokens_per_prompt": steps,
            "first_divergence_position": first,
            "median_first_divergence_step": float(np.median(censored)),
            "no_divergence_prompts": sum(item is None for item in first)}


def evaluate_config(reference, scales, ids, prompts, golden, float_windows, bits, group, improved):
    exported = export_runq_model(reference, scales, bits, group, improved)
    base_scales = exported.pop("dynamic_base_scales")
    model = RunQIntegerLlama(exported, HERE / "luts")
    row = {"weight_bits": bits, "group_size": group,
           "weight_quantizer": "mse_optimal_clipping" if improved else "symmetric_maxabs",
           "accumulator_bounds": accumulator_bounds(reference, bits, group)}
    row.update(teacher_forced_metrics(model, ids, base_scales["logits"], float_windows))
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
    float_windows = float_eval_windows(reference, ids)
    calibration = json.loads((HERE / "calibration_260k.json").read_text(encoding="utf-8"))
    scales = calibration["activation_scales"]
    ablation_rows, ablation_ranking = run_ablation(reference, scales, ids, HERE / "luts")
    float_row = next(row for row in ablation_rows if row["name"] == "float")
    int8 = evaluate_config(reference, scales, ids, prompts, golden, float_windows, 8, 64, False)
    target = {"max_perplexity_ratio": 1.02, "min_teacher_forced_top1_agreement_pct": 97.0}
    if int8["perplexity"] > float_row["perplexity"] * target["max_perplexity_ratio"] or int8["teacher_forced_top1_agreement_pct"] < target["min_teacher_forced_top1_agreement_pct"]:
        raise RuntimeError(f"INT8 quality gate failed: {int8}")
    baseline = [int8]
    baseline += [evaluate_config(reference, scales, ids, prompts, golden, float_windows, 4, group, False) for group in (32, 64, 128)]
    int8_ppl = baseline[0]["perplexity"]
    large = any(row["perplexity"] - int8_ppl > 0.25 or row["teacher_forced_top1_agreement_pct"] < 90.0 for row in baseline[1:])
    improved = [evaluate_config(reference, scales, ids, prompts, golden, float_windows, 4, group, True) for group in (32, 64, 128)] if large else []
    result = {"schema_version": 2, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "model": "stories260K", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "eval": {"tokens": 50000, "slice_sha256": "ae55d9eb37d0e99c6e1051dbeb212f51b6c1d2d98858fccd81f5676bee92a4e7"},
              "calibration": calibration["metadata"], "configs": baseline,
              "prior_static_results": {
                  "int8": {"perplexity": 4.096857063361471},
                  "int4_g32_maxabs": {"perplexity": 5.079092740660963},
                  "note": "Chunk-2 committed baseline; it predates teacher-forced top-1 and KL fields"},
              "ablation_pre_fix": {"rows": ablation_rows, "ranking_by_kl_reduction": ablation_ranking},
              "quality_target": {**target, "float_perplexity": float_row["perplexity"], "passed": True},
              "activation_path": {"scheme": "runtime max-abs INT8, code=RNE(value*128/peak), scale=(peak_multiplier, right_shift=7)",
                                  "matmul_group_size": 64, "kv_group_size": "one 8-element KV head",
                                  "attention_value_group_size": 16,
                                  "calibration_base_guard_bits": 8, "weight_group_size_int8": 64},
              "int4_loss_large_rule": "ppl delta vs INT8 > 0.25 or teacher-forced top-1 agreement < 90%",
              "int4_loss_large": large, "improvement": {"name": "MSE-optimal clipping per group", "configs": improved}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"configs": [{"bits": r["weight_bits"], "group": r["group_size"], "quantizer": r["weight_quantizer"],
                                   "ppl": r["perplexity"], "top1": r["teacher_forced_top1_agreement_pct"],
                                   "kl": r["logit_kl_float_to_candidate"], "median_divergence": r["greedy_agreement"]["median_first_divergence_step"]}
                                  for r in baseline + improved]}, indent=2))


if __name__ == "__main__":
    main()
