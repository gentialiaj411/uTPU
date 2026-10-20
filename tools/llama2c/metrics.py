"""Pinned TinyStories teacher-forced NLL/PPL and greedy agreement metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .reference import Llama2CModel
from .tokenizer import Tokenizer
from .torch_oracle import load_torch_model
from .verify import verify

ROOT = Path(__file__).resolve().parents[2]


def tokenize_eval(tokenizer: Tokenizer, path: Path, minimum: int = 50_000) -> list[int]:
    # run.c's reference merge loop is quadratic in one enormous input. TinyStories has
    # explicit story boundaries, so encode independent documents exactly as training
    # corpora do and concatenate their token streams.
    ids: list[int] = []
    for story in path.read_text(encoding="utf-8").split("<|endoftext|>"):
        story = story.strip()
        if story:
            # Match llama2.c tinystories.py process_shard exactly: strip, BOS per
            # story, and no EOS; adjacent examples are separated by the next BOS.
            ids.extend(tokenizer.encode(story, bos=True, eos=False))
        if len(ids) >= minimum + 1:
            break
    if len(ids) < minimum + 1:
        raise RuntimeError(f"eval slice only produced {len(ids)} tokens")
    return ids[: minimum + 1]


def teacher_forced(model, ids: list[int], seq_len: int, device: str) -> tuple[float, int]:
    # Independent full-context windows; every reported target has a preceding input.
    total_nll, count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(ids) - 1, seq_len):
            chunk = ids[start : min(start + seq_len + 1, len(ids))]
            if len(chunk) < 2:
                continue
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device)[None]
            y = torch.tensor(chunk[1:], dtype=torch.long, device=device)[None]
            logits = model(x, y)
            total_nll += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
            count += y.numel()
    return total_nll / count, count


def greedy_agreement(float_factory, candidate_factory, prompts: list[list[int]], steps: int = 128) -> dict:
    """Compare autoregressive candidates; factories return reset, stateful decoders."""
    identical, total, first = 0, 0, []
    for prompt in prompts:
        golden, candidate = float_factory(), candidate_factory()
        gold_logits = cand_logits = None
        for pos, token in enumerate(prompt):
            gold_logits = golden.forward(token, pos)
            cand_logits = candidate.forward(token, pos)
        divergence = None
        for step in range(steps):
            gold_token = int(gold_logits.argmax())
            cand_token = int(cand_logits.argmax())
            identical += gold_token == cand_token
            total += 1
            if divergence is None and gold_token != cand_token:
                divergence = step
            pos = len(prompt) + step
            gold_logits = golden.forward(gold_token, pos)
            cand_logits = candidate.forward(cand_token, pos)
        first.append(divergence)
    return {"prompts": len(prompts), "tokens_per_prompt": steps,
            "identical_tokens_pct_vs_float": 100.0 * identical / total,
            "first_divergence_position": first}


def evaluate(name: str, checkpoint: Path, tokenizer_path: Path, device: str) -> dict:
    ref = Llama2CModel(checkpoint)
    tok = Tokenizer(tokenizer_path, ref.config.vocab_size)
    ids = tokenize_eval(tok, ROOT / "models/llama2c/eval_slice.txt")
    model = load_torch_model(ref, device)
    nll, count = teacher_forced(model, ids, ref.config.seq_len, device)
    return {"name": name, "eval_tokens": count, "mean_nll": nll, "perplexity": math.exp(nll),
            "pytorch_parity_64_tokens": verify(str(checkpoint)),
            "greedy_agreement": {"prompts": 64, "tokens_per_prompt": 128,
              "identical_tokens_pct_vs_float": 100.0, "first_divergence_position": None,
              "note": "float baseline compared with itself; candidate comparison API begins in numerics_v2"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="bench/results/llama2c_float_baseline.json")
    args = parser.parse_args()
    base = ROOT / "models/llama2c"
    rows = [
        evaluate("stories260K", base / "stories260K/stories260K.bin", base / "stories260K/tok512.bin", args.device),
        evaluate("stories15M", base / "stories15M.bin", base / "tokenizer.bin", args.device),
    ]
    slice_path = base / "eval_slice.txt"
    out = {"schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "device": args.device, "float_evaluator": "pinned llama2.c model.py after NumPy parity gate",
           "eval": {"dataset": "roneneldan/TinyStories", "split": "validation",
           "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64", "slice_bytes": slice_path.stat().st_size,
           "slice_sha256": hashlib.sha256(slice_path.read_bytes()).hexdigest(),
           "tokenization": "llama2.c tinystories.py process_shard: strip + BOS per story + no EOS",
           "protocol": "non-overlapping 512-token windows"},
           "models": rows}
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
