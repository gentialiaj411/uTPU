"""Generate stories15M numerics-v2 metrics and bandwidth estimates."""

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

from .export import accumulator_bounds, export_runq_model
from .run_260k import golden_continuations, prompts_from
from .runq_runtime import RunQIntegerLlama

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LOGIT_CHUNK = 4096


def teacher_forced_metrics(reference, model, ids, base_logit_scale, device="cuda"):
    float_model = load_torch_model(reference, device)
    total_nll = total_kl = 0.0
    matches = count = 0
    seq_len = int(reference.config.seq_len)
    with torch.inference_mode():
        for start in range(0, len(ids) - 1, seq_len):
            chunk = ids[start:min(start + seq_len + 1, len(ids))]
            inputs = torch.tensor(chunk[:-1], dtype=torch.long, device=device)[None]
            targets = np.asarray(chunk[1:], dtype=np.int64)
            float_logits = float_model(inputs, inputs).cpu().numpy()[0].astype(np.float64)
            float_max = np.max(float_logits, axis=1)
            float_logsum = float_max + np.log(np.sum(np.exp(float_logits - float_max[:, None]), axis=1))

            candidate_chunks = []
            candidate_max = np.full(targets.size, -np.inf, dtype=np.float64)
            candidate_sum = np.zeros(targets.size, dtype=np.float64)
            candidate_argmax = np.zeros(targets.size, dtype=np.int64)
            target_logits = np.empty(targets.size, dtype=np.float64)
            for begin, codes in model.forward_sequence_logits_chunks(chunk[:-1], LOGIT_CHUNK):
                logits = codes.astype(np.float64) * base_logit_scale
                candidate_chunks.append((begin, codes))
                local_index = np.argmax(codes, axis=1)
                local_max = logits[np.arange(targets.size), local_index]
                new_max = np.maximum(candidate_max, local_max)
                candidate_sum = (candidate_sum * np.exp(candidate_max - new_max) +
                                 np.sum(np.exp(logits - new_max[:, None]), axis=1))
                update = local_max > candidate_max
                candidate_argmax[update] = begin + local_index[update]
                candidate_max = new_max
                selected = (targets >= begin) & (targets < begin + codes.shape[1])
                target_logits[selected] = logits[selected, targets[selected] - begin]
            candidate_logsum = candidate_max + np.log(candidate_sum)
            total_nll += float(np.sum(candidate_logsum - target_logits))
            matches += int(np.sum(np.argmax(float_logits, axis=1) == candidate_argmax))

            for begin, codes in candidate_chunks:
                end = begin + codes.shape[1]
                float_logp = float_logits[:, begin:end] - float_logsum[:, None]
                candidate_logp = codes.astype(np.float64) * base_logit_scale - candidate_logsum[:, None]
                total_kl += float(np.sum(np.exp(float_logp) * (float_logp - candidate_logp)))
            count += targets.size
    mean = total_nll / count
    return {"eval_tokens": count, "mean_nll": mean, "perplexity": math.exp(mean),
            "teacher_forced_top1_agreement_pct": 100.0 * matches / count,
            "logit_kl_float_to_candidate": total_kl / count,
            "logit_chunk_size": LOGIT_CHUNK}


def greedy_first_divergence(model, prompts, golden, steps=128):
    first = []
    for prompt, expected in zip(prompts, golden):
        model.reset()
        candidate = None
        for pos, token in enumerate(prompt):
            candidate = model.forward_argmax(token, pos, LOGIT_CHUNK)
        divergence = None
        for step in range(steps):
            if candidate != expected[step]:
                divergence = step
                break
            candidate = model.forward_argmax(candidate, len(prompt) + step, LOGIT_CHUNK)
        first.append(divergence)
    censored = [item if item is not None else steps for item in first]
    return {"prompts": len(prompts), "tokens_per_prompt": steps,
            "first_divergence_position": first,
            "median_first_divergence_step": float(np.median(censored)),
            "no_divergence_prompts": sum(item is None for item in first)}


def _linear_storage(record, bits):
    codes = (record["weight"].size * bits + 7) // 8
    scale_pairs = record["multiplier"].size * 5  # signed INT32 multiplier + UINT8 shift
    return int(codes + scale_pairs)


def hardware_budget(exported, bits, seq=256):
    embedding = int(exported["embedding"].size + exported["embedding_scale_multiplier"].size * 4)
    gamma = int(exported["gamma_att"].nbytes + exported["gamma_ffn"].nbytes + exported["gamma_final"].nbytes)
    linears = sum(_linear_storage(item, bits) for layer in exported["layers"] for item in layer.values())
    classifier = _linear_storage(exported["classifier"], bits)
    weight_bytes = embedding + gamma + linears + classifier
    embedding_row = int(exported["embedding"].shape[1] + 4)
    bytes_streamed = linears + classifier + gamma + embedding_row
    c = exported["config"]
    kv_codes = int(c["n_layers"] * seq * c["kv_dim"] * 2)
    kv_scales = int(c["n_layers"] * seq * c["n_kv_heads"] * 4 * 2)
    rates = {f"{rate}_MBps": rate * 1_000_000.0 / bytes_streamed for rate in (25, 50, 100)}
    return {"weight_bytes_including_scales": weight_bytes,
            "kv_cache_bytes_seq256_including_scales": kv_codes + kv_scales,
            "bytes_streamed_per_token": bytes_streamed,
            "estimated_tokens_per_second": rates,
            "estimate_assumption": "bandwidth-only QSPI limit; decimal MB/s; no compute/command overhead"}


def evaluate(reference, scales, ids, prompts, golden, bits, group, improved, device):
    exported = export_runq_model(reference, scales, bits, group, improved)
    base_scales = exported.pop("dynamic_base_scales")
    budget = hardware_budget(exported, bits)
    model = RunQIntegerLlama(exported, HERE / "luts")
    row = {"weight_bits": bits, "group_size": group,
           "weight_quantizer": "mse_optimal_clipping" if improved else "symmetric_maxabs",
           "accumulator_bounds": accumulator_bounds(reference, bits, group),
           "hardware_budget": budget}
    row.update(teacher_forced_metrics(reference, model, ids, base_scales["logits"], device))
    if golden is not None:
        row["greedy_agreement"] = greedy_first_divergence(model, prompts, golden)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "bench/results/numerics_v2_15m.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, default=50000)
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge", nargs="+", type=Path,
                        help="merge completed JSON shards into --output without rerunning")
    parser.add_argument("--config", action="append", choices=("int8", "int4-g32", "int4-g64", "int4-g128",
                                                                  "mse-g32", "mse-g64", "mse-g128"),
                        help="evaluate only the named configuration; repeat as needed")
    args = parser.parse_args()
    if args.merge:
        shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.merge]
        if not shards or any(shard.get("eval", {}).get("tokens") != 50000 for shard in shards):
            raise ValueError("all merge shards must use the 50,000-token evaluation")
        rows = []
        seen = set()
        for shard in shards:
            for row in shard.get("configs", []):
                key = (row["weight_bits"], row["group_size"], row["weight_quantizer"] == "mse_optimal_clipping")
                if key in seen:
                    raise ValueError(f"duplicate configuration in merge: {key}")
                seen.add(key)
                rows.append(row)
        order = {(8, 64, False): 0, (4, 32, False): 1, (4, 64, False): 2,
                 (4, 128, False): 3, (4, 32, True): 4, (4, 64, True): 5, (4, 128, True): 6}
        rows.sort(key=lambda row: order[(row["weight_bits"], row["group_size"], row["weight_quantizer"] == "mse_optimal_clipping")])
        result = dict(shards[0])
        result["complete"] = len(rows) == 7
        result["configs"] = rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps([(r["weight_bits"], r["group_size"], r["perplexity"]) for r in rows]))
        return
    base = ROOT / "models/llama2c"
    checkpoint = base / "stories15M.bin"
    reference = Llama2CModel(checkpoint)
    tokenizer = Tokenizer(base / "tokenizer.bin", reference.config.vocab_size)
    ids = tokenize_eval(tokenizer, base / "eval_slice.txt", minimum=args.tokens)[:args.tokens + 1]
    prompts = prompts_from(ids)
    golden = None if args.skip_greedy else golden_continuations(checkpoint, prompts)
    calibration = json.loads((HERE / "calibration_15m.json").read_text(encoding="utf-8"))
    scales = calibration["activation_scales"]
    spec_by_name = {"int8": (8, 64, False),
                    **{f"int4-g{group}": (4, group, False) for group in (32, 64, 128)},
                    **{f"mse-g{group}": (4, group, True) for group in (32, 64, 128)}}
    specs = list(spec_by_name.values()) if not args.config else [spec_by_name[name] for name in args.config]
    rows = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("eval", {}).get("tokens") == args.tokens:
            rows = previous.get("configs", [])
    completed = {(row["weight_bits"], row["group_size"], row["weight_quantizer"] == "mse_optimal_clipping")
                 for row in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = {"schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "complete": len(rows) == len(specs), "model": "stories15M",
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "eval": {"tokens": args.tokens, "slice_sha256": "ae55d9eb37d0e99c6e1051dbeb212f51b6c1d2d98858fccd81f5676bee92a4e7"},
               "calibration": calibration["metadata"],
               "logits": {"vocab_size": 32000, "chunk_size": LOGIT_CHUNK,
               "argmax_topk": "integer chunk-local selection and integer global merge"}, "configs": rows}
    for bits, group, improved in specs:
        if (bits, group, improved) in completed:
            continue
        print(f"starting bits={bits} group={group} improved={improved}", flush=True)
        rows.append(evaluate(reference, scales, ids, prompts, golden, bits, group, improved, args.device))
        partial = {"schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "complete": len(rows) == len(specs), "model": "stories15M",
                   "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "eval": {"tokens": args.tokens, "slice_sha256": "ae55d9eb37d0e99c6e1051dbeb212f51b6c1d2d98858fccd81f5676bee92a4e7"},
                   "calibration": calibration["metadata"],
                   "logits": {"vocab_size": 32000, "chunk_size": LOGIT_CHUNK,
                   "argmax_topk": "integer chunk-local selection and integer global merge"}, "configs": rows}
        args.output.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
    result = partial
    print(json.dumps([{"bits": row["weight_bits"], "group": row["group_size"],
                       "quantizer": row["weight_quantizer"], "ppl": row["perplexity"],
                       "top1": row["teacher_forced_top1_agreement_pct"],
                       "median_divergence": row.get("greedy_agreement", {}).get("median_first_divergence_step"),
                       "budget": row["hardware_budget"]} for row in rows], indent=2))


if __name__ == "__main__":
    main()
