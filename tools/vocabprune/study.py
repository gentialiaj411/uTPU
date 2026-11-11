"""Measure training-free llama2.c vocabulary pruning for stories15M."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from tools.vocabprune.core import (KEEP_SIZES, build_idmap, count_tokens, decode_compact,
                                    encode_with_fallback, prune_export, write_idmap)

TOKENIZER = ROOT / "models/llama2c/tokenizer.bin"
CHECKPOINT = ROOT / "models/llama2c/stories15M.bin"
EXPORT = ROOT / "models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz"
LUTS = ROOT / "firmware/host/numerics_v2/luts"
EVAL_TEXT = Path(r"C:\Users\bhask\Documents\PROJECTS\uTPU-codesign\build\codesign\eval_text.txt")
TRAIN_SAMPLE = Path(r"C:\Users\bhask\Documents\PROJECTS\uTPU-codesign\build\codesign\data\train_sample_256MiB.txt")
OUT = ROOT / "bench/results/vocab_prune_study.json"
IDMAP_DIR = ROOT / "tools/vocabprune/idmaps"
BUILD = ROOT / "build/vocabprune"


def load_maps() -> dict[int, dict]:
    return {k: json.loads((IDMAP_DIR / f"idmap_{k}.json").read_text(encoding="utf-8")) for k in KEEP_SIZES}


def build_maps() -> dict:
    tok = Tokenizer(TOKENIZER, 32000)
    counts, total = count_tokens(tok, TRAIN_SAMPLE)
    maps = {}
    for keep in KEEP_SIZES:
        item = build_idmap(tok, counts, total, keep, TRAIN_SAMPLE)
        write_idmap(IDMAP_DIR / f"idmap_{keep}.json", item)
        maps[keep] = item
    out = {
        "status": "MEASURED",
        "tokenizer": str(TOKENIZER),
        "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "train_sample": str(TRAIN_SAMPLE),
        "train_sample_bytes": TRAIN_SAMPLE.stat().st_size,
        "train_sample_sha256": hashlib.sha256(TRAIN_SAMPLE.read_bytes()).hexdigest(),
        "token_occurrences": total,
        "distinct_tokens": len(counts),
        "maps": {str(k): {"path": str(IDMAP_DIR / f"idmap_{k}.json"),
                           "top_frequency_rows": item["ranked_rows"][:10],
                           "kept_rows": len(item["kept_original_ids"])} for k, item in maps.items()},
    }
    (BUILD / "maps.json").parent.mkdir(parents=True, exist_ok=True)
    (BUILD / "maps.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def fixed_eval_sequences(tok: Tokenizer, maps: dict[int, dict]) -> tuple[list[dict], int]:
    raw = EVAL_TEXT.read_bytes()
    docs = [part for part in raw.decode("utf-8").split("<|endoftext|>") if part]
    rows = []
    for doc in docs:
        row = {"text": doc, "raw_bytes": len(doc.encode("utf-8")), "maps": {}}
        original = tok.encode(doc, bos=False, eos=False)
        row["original"] = original
        for keep, idmap in maps.items():
            compact, expanded, fallback, original_count = encode_with_fallback(tok, doc, idmap)
            row["maps"][keep] = {"compact": compact, "expanded": expanded,
                                 "fallback": fallback, "original_count": original_count}
        rows.append(row)
    return rows, len(raw)


def _float_scores(model, rows: list[dict], maps: dict[int, dict], raw_bytes: int, device: str) -> dict:
    import torch
    result = {}
    kept_by_k = {k: torch.tensor(v["kept_original_ids"], dtype=torch.long, device=device) for k, v in maps.items()}
    all_ids = torch.arange(32000, dtype=torch.long, device=device)
    sums = {k: [0.0, 0] for k in (32000, *KEEP_SIZES)}
    t0 = time.time()
    model.eval()
    with torch.inference_mode():
        for keep in (32000, *KEEP_SIZES):
            for row in rows:
                if keep == 32000:
                    sequence = row["original"]
                    target_ids = lambda begin, end: row["original"][begin:end]
                else:
                    info = row["maps"][keep]
                    sequence = info["expanded"]
                    target_ids = lambda begin, end, info=info: info["compact"][begin:end]
                for start in range(0, len(sequence) - 1, 128):
                    window = sequence[start:min(start + 129, len(sequence))]
                    if len(window) < 2:
                        continue
                    x = torch.tensor(window[:-1], dtype=torch.long, device=device)[None]
                    # The pinned torch oracle returns only the last position when targets are
                    # omitted; supplying a dummy target requests the full teacher-forced logit
                    # matrix while leaving the logits themselves unchanged.
                    selected = model(x, torch.zeros_like(x))[0]
                    if keep != 32000:
                        selected = selected[:, kept_by_k[keep]]
                    target = torch.tensor(target_ids(start + 1, start + len(window)), dtype=torch.long, device=device)
                    nll = torch.logsumexp(selected, dim=-1) - selected.gather(1, target[:, None]).squeeze(1)
                    sums[keep][0] += float(nll.sum().cpu())
                    sums[keep][1] += int(target.numel())
    for keep, (nll, count) in sums.items():
        result[str(keep)] = {"status": "MEASURED", "eval_tokens": count, "nll_nats": nll,
                             "validation_bytes_scored": raw_bytes,
                             "bits_per_byte": nll / math.log(2) / raw_bytes,
                             "token_loss": nll / count if count else None,
                             "device": device, "elapsed_seconds": time.time() - t0,
                             "softmax": "restricted to kept rows"}
    return result


def _integer_nll(logits: np.ndarray, targets: np.ndarray, logit_scale: float) -> float:
    values = np.asarray(logits, dtype=np.float64) * logit_scale
    m = np.max(values, axis=1)
    return float(np.sum(m + np.log(np.sum(np.exp(values - m[:, None]), axis=1)) -
                       values[np.arange(targets.size), targets]))


def _integer_scores(rows: list[dict], maps: dict[int, dict], raw_bytes: int) -> dict:
    from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama, _linear_high
    payload = np.load(EXPORT, allow_pickle=True)["payload"][0]
    logit_scale = float(payload["dynamic_base_scales"]["logits"])
    full = RunQIntegerLlama(payload, LUTS, kv_bits=4)
    pruned = {k: prune_export(payload, v["kept_original_ids"]) for k, v in maps.items()}
    classifiers = {k: pruned[k]["classifier"] for k in KEEP_SIZES}
    sums = {k: [0.0, 0] for k in (32000, *KEEP_SIZES)}
    t0 = time.time()
    for keep in (32000, *KEEP_SIZES):
        for row in rows:
            if keep == 32000:
                sequence = row["original"]
                classifier = full.classifier
                target_ids = lambda begin, end: row["original"][begin:end]
            else:
                info = row["maps"][keep]
                sequence = info["expanded"]
                classifier = classifiers[keep]
                target_ids = lambda begin, end, info=info: info["compact"][begin:end]
            for start in range(0, len(sequence) - 1, 128):
                window = sequence[start:min(start + 129, len(sequence))]
                if len(window) < 2:
                    continue
                # Candidate hidden state is computed from the exact byte-expanded input.  Use
                # its sliced classifier; this is equivalent to RunQIntegerLlama on prune_export.
                hidden = full._forward_sequence_hidden(np.asarray(window[:-1], dtype=np.int64))
                logits = _linear_high(hidden, classifier)
                targets = np.asarray(target_ids(start + 1, start + len(window)), dtype=np.int64)
                sums[keep][0] += _integer_nll(logits, targets, logit_scale)
                sums[keep][1] += int(targets.size)
    result = {}
    for keep, (nll, count) in sums.items():
        result[str(keep)] = {"status": "MEASURED", "eval_tokens": count, "nll_nats": nll,
                             "validation_bytes_scored": raw_bytes,
                             "bits_per_byte": nll / math.log(2) / raw_bytes,
                             "token_loss": nll / count if count else None, "kv_bits": 4,
                             "logit_scale": logit_scale, "elapsed_seconds": time.time() - t0,
                             "softmax": "restricted to kept rows",
                             "implementation": "RunQIntegerLlama hidden path plus pruned classifier row slice"}
    return result


def _generation(rows: list[dict], maps: dict[int, dict]) -> dict:
    from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
    payload = np.load(EXPORT, allow_pickle=True)["payload"][0]
    full = RunQIntegerLlama(payload, LUTS, kv_bits=4)
    candidates = {k: RunQIntegerLlama(prune_export(payload, v["kept_original_ids"]), LUTS, kv_bits=4)
                  for k, v in maps.items()}
    inverse = {k: np.asarray(v["kept_original_ids"], dtype=np.int64) for k, v in maps.items()}
    prompts = []
    for row in rows[:20]:
        info = row["maps"][KEEP_SIZES[-1]]
        if len(info["compact"]) >= 1:
            prompts.append((info["expanded"][:min(8, len(info["expanded"]))],
                            info["compact"][:min(8, len(info["compact"]))]))
    out = {str(k): {"prompts": len(prompts), "generated_steps": 0, "full_argmax_outside": 0,
                    "sequence_mismatches": 0} for k in KEEP_SIZES}
    tokenizer = Tokenizer(TOKENIZER, 32000)
    for row in rows[:20]:
        for keep in KEEP_SIZES:
            # Build each prompt from the same source text so a smaller K can byte-fallback
            # rows that the 8192 map retained.
            compact, expanded, _, _ = encode_with_fallback(tokenizer, row["text"], maps[keep])
            compact = compact[:8]
            expanded = expanded[:8]
            if not compact:
                continue
            inv = np.asarray(maps[keep]["kept_original_ids"], dtype=np.int64)
            full.reset(); candidates[keep].reset()
            full_generated, candidate_generated = [], []
            full_tokens = list(expanded)
            candidate_tokens = list(compact)
            for pos in range(len(compact) + 64 - 1):
                full_out = int(full.forward(full_tokens[pos], pos).argmax())
                cand_logits = candidates[keep].forward(candidate_tokens[pos], pos)
                cand_compact = int(cand_logits.argmax())
                cand_out = int(inverse[keep][cand_compact])
                if pos + 1 >= len(compact):
                    full_generated.append(full_out)
                    candidate_generated.append(cand_out)
                    out[str(keep)]["generated_steps"] += 1
                    if int(np.asarray(inverse[keep] == full_out).any()) == 0:
                        out[str(keep)]["full_argmax_outside"] += 1
                full_tokens.append(full_out if pos + 1 >= len(compact) else expanded[pos + 1])
                candidate_tokens.append(cand_compact if pos + 1 >= len(compact) else compact[pos + 1])
            if full_generated != candidate_generated:
                out[str(keep)]["sequence_mismatches"] += 1
    for item in out.values():
        item["full_argmax_outside_fraction"] = (item["full_argmax_outside"] / item["generated_steps"]
                                                 if item["generated_steps"] else None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if not (args.maps or args.quality or args.generation or args.all):
        parser.error("choose --maps, --quality, --generation, or --all")
    result = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    if args.maps or args.all:
        result["maps"] = build_maps()
    maps = load_maps()
    tok = Tokenizer(TOKENIZER, 32000)
    rows, raw_bytes = fixed_eval_sequences(tok, maps)
    result["eval"] = {"path": str(EVAL_TEXT), "sha256": hashlib.sha256(EVAL_TEXT.read_bytes()).hexdigest(),
                       "bytes": raw_bytes, "documents": len(rows),
                       "fallback": {str(k): {"fallback_tokens": sum(r["maps"][k]["fallback"] for r in rows),
                                             "original_tokens": sum(r["maps"][k]["original_count"] for r in rows),
                                             "fraction": sum(r["maps"][k]["fallback"] for r in rows) /
                                             sum(r["maps"][k]["original_count"] for r in rows)} for k in KEEP_SIZES}}
    if args.quality or args.all:
        import torch
        ref = Llama2CModel(CHECKPOINT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result["float"] = _float_scores(__import__("tools.llama2c.torch_oracle", fromlist=["load_torch_model"]).load_torch_model(ref, device), rows, maps, raw_bytes, device)
        result["integer"] = _integer_scores(rows, maps, raw_bytes)
    if args.generation or args.all:
        result["generation"] = _generation(rows, maps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
