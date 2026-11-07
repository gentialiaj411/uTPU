"""Measure lossless speculative acceptance on TinyStories validation openings.

The expensive part of this study is deliberately only the 15M greedy run. A
saved greedy sequence is the oracle for every drafter: because decoding is
lossless greedy, a proposed token is accepted exactly when it equals the next
token in that saved sequence. This keeps the acceptance sweep offline and
reproducible.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.llama2c.tokenizer import Tokenizer
from tools.decode.harness import load_runtime_260k
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama


PYTHON_REQUESTED = r"C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe"
SEED = 0
PROMPT_COUNT = 50
MAX_SEQ = 128
KS = (1, 2, 3, 4, 6)
SEQUENCE_PATH = ROOT / "bench/results/specdec_sequences.json"
HOST_DRAFT_PATH = ROOT / "bench/results/specdec_host_drafts.json"
STUDY_PATH = ROOT / "bench/results/specdec_study.json"


def load15() -> RunQIntegerLlama:
    payload = np.load(
        ROOT / "models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz",
        allow_pickle=True,
    )["payload"][0]
    payload.pop("dynamic_base_scales", None)
    return RunQIntegerLlama(
        payload, ROOT / "firmware/host/numerics_v2/luts", kv_bits=4
    )


def prompts50():
    """Return a fixed, listed sample of 50 validation story openings."""
    tok = Tokenizer(ROOT / "models/llama2c/tokenizer.bin", 32000)
    stories = [s.strip() for s in
               (ROOT / "models/llama2c/TinyStories-valid.txt")
               .read_text(encoding="utf8").split("<|endoftext|>") if s.strip()]
    candidates = []
    seen = set()
    for story_index, story in enumerate(stories):
        ids = tok.encode(story[:160])[:8]
        key = tuple(ids)
        if len(ids) >= 2 and key not in seen:
            seen.add(key)
            candidates.append({"story_index": story_index, "text": story[:160],
                               "tokens": ids})
    random.Random(SEED).shuffle(candidates)
    selected = candidates[:PROMPT_COUNT]
    if len(selected) < PROMPT_COUNT:
        raise RuntimeError(f"only found {len(selected)} usable validation prompts")
    return selected, tok


def greedy_sequence(rt: RunQIntegerLlama, prompt: list[int]):
    """Generate one bit-exact sequence through position 127 or EOS."""
    rt.reset()
    seq = list(map(int, prompt))
    for pos, token in enumerate(seq[:-1]):
        rt.forward_argmax(token, pos, 4096)
    eos = False
    while len(seq) < MAX_SEQ:
        pos = len(seq) - 1
        nxt = int(rt.forward_argmax(seq[-1], pos, 4096))
        seq.append(nxt)
        if nxt == 2:
            eos = True
            break
    return seq, eos


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)


def collect_sequences(prompt_rows):
    """Generate/resume the one expensive sequence per prompt with checkpoints."""
    existing = None
    if SEQUENCE_PATH.exists():
        try:
            existing = json.loads(SEQUENCE_PATH.read_text(encoding="utf8"))
        except (OSError, ValueError):
            existing = None
    prompt_tokens = [row["tokens"] for row in prompt_rows]
    rows = []
    if (existing and existing.get("seed") == SEED and
            existing.get("max_seq") == MAX_SEQ and
            existing.get("prompt_tokens") == prompt_tokens):
        rows = existing.get("sequences", [])

    rt = load15() if len(rows) < len(prompt_rows) else None
    total_start = time.perf_counter()
    for index in range(len(rows), len(prompt_rows)):
        row = prompt_rows[index]
        start = time.perf_counter()
        seq, eos = greedy_sequence(rt, row["tokens"])
        elapsed = time.perf_counter() - start
        rows.append({"index": index, "story_index": row["story_index"],
                     "prompt_tokens": row["tokens"], "sequence_tokens": seq,
                     "generated_tokens": len(seq) - len(row["tokens"]),
                     "eos": eos, "wall_seconds": elapsed})
        _write_json(SEQUENCE_PATH, {
            "status": "partial", "seed": SEED, "max_seq": MAX_SEQ,
            "prompt_count_target": PROMPT_COUNT,
            "prompt_tokens": prompt_tokens, "sequences": rows,
            "model": "stories15M", "board_config": {"kv_bits": 4,
            "max_seq": MAX_SEQ}, "python": sys.executable,
            "numpy": np.__version__, "wall_seconds_so_far":
            time.perf_counter() - total_start,
        })
        print(f"greedy {index + 1}/{len(prompt_rows)}: "
              f"{elapsed:.3f}s, {len(seq)} tokens", flush=True)
    # A resumed complete artifact has no new model work in this invocation;
    # retain the measured wall time from the original per-prompt runs.
    total = (sum(r["wall_seconds"] for r in rows)
             if len(rows) == len(prompt_rows) and existing else
             time.perf_counter() - total_start)
    out = {
        "status": "pass", "seed": SEED, "max_seq": MAX_SEQ,
        "prompt_count": len(rows), "prompt_count_target": PROMPT_COUNT,
        "prompt_tokens": prompt_tokens, "sequences": rows,
        "model": "stories15M", "board_config": {"kv_bits": 4,
        "max_seq": MAX_SEQ}, "python": sys.executable,
        "numpy": np.__version__, "wall_seconds": total,
        "per_prompt_wall_seconds": [r["wall_seconds"] for r in rows],
    }
    _write_json(SEQUENCE_PATH, out)
    return rows, total


def build_static_tables(tok):
    text_path = ROOT / "models/llama2c/TinyStories-train.slice.txt"
    text = text_path.read_text(encoding="utf8")
    # Tokenizer.encode is the small reference port and scans for merges over
    # the whole input. Tokenize story records independently so the 1 MiB
    # training slice remains tractable while preserving the dataset's record
    # boundaries; no cross-story n-gram is useful to a text drafter.
    ids = []
    tables = {"static_bigram": {}, "static_trigram": {}}
    for story in text.split("<|endoftext|>"):
        if story:
            story_ids = tok.encode(story, bos=False)
            ids.extend(story_ids)
            for n, label in ((1, "static_bigram"), (2, "static_trigram")):
                table = tables[label]
                for i in range(len(story_ids) - n):
                    table[tuple(map(int, story_ids[i:i + n]))] = int(story_ids[i + n])
    # The value for a repeated key is the last successor in the train slice.
    table_sizes = {}
    for n, label in ((1, "static_bigram"), (2, "static_trigram")):
        entries = len(tables[label])
        table_sizes[label] = {
            "entries": entries,
            "packed_bytes_32bit_ids": entries * (n + 1) * 4,
            "storage": "host RAM or flash; not BRAM",
        }
    return tables, {"path": str(text_path.relative_to(ROOT)),
                    "text_bytes": len(text.encode("utf8")),
                    "token_count": len(ids), "tables": table_sizes}


def context_next(history: list[int], n: int):
    """Most recent prior match of the current n-token suffix."""
    if len(history) < n:
        return None
    key = tuple(history[-n:])
    current_start = len(history) - n
    for i in range(current_start - 1, -1, -1):
        if tuple(history[i:i + n]) == key:
            return int(history[i + n])
    return None


def _static_next(history, table, n):
    if len(history) < n:
        return None
    return table.get(tuple(history[-n:]))


def draft_tokens(name, history, target_offset, k, host_tokens, static_tables):
    """Make exactly the proposals a drafter can make before verification."""
    temp = list(history)
    out = []
    for j in range(k):
        if name.startswith("context"):
            candidate = context_next(temp, int(name[-1]))
        elif name == "static_bigram":
            candidate = _static_next(temp, static_tables[name], 1)
        elif name == "static_trigram":
            candidate = _static_next(temp, static_tables[name], 2)
        elif name == "host_tiny":
            candidate = (host_tokens[target_offset + j]
                         if target_offset + j < len(host_tokens) else None)
        elif name == "combined":
            candidate = context_next(temp, 4)
            if candidate is None:
                candidate = (host_tokens[target_offset + j]
                             if target_offset + j < len(host_tokens) else None)
        else:
            raise ValueError(name)
        if candidate is None:
            break
        out.append(int(candidate))
        temp.append(int(candidate))
    return out


def offline_passes(prompt, seq, name, k, host_tokens, static_tables):
    """Evaluate a drafter against a saved greedy sequence, without model calls."""
    history = list(map(int, prompt))
    target_index = len(prompt)
    counts, emitted, uart_bytes = [], [], 0
    while target_index < len(seq):
        target_offset = target_index - len(prompt)
        drafts = draft_tokens(name, history, target_offset, k, host_tokens,
                              static_tables)
        accepted = 0
        for j, draft in enumerate(drafts):
            if target_index + j >= len(seq) or int(draft) != int(seq[target_index + j]):
                break
            accepted += 1
        emit = min(accepted + 1, len(seq) - target_index)
        counts.append(accepted)
        emitted.append(emit)
        # A real host-draft round carries the requested k 16-bit candidates,
        # plus one command byte and a two-byte response header.
        if name == "host_tiny":
            uart_bytes += 1 + 2 * k + 2
        history.extend(seq[target_index:target_index + emit])
        target_index += emit
    return {"accepted": counts, "emitted": emitted,
            "passes": len(emitted), "uart_bytes": uart_bytes}


def mean_ci(values):
    values = [float(x) for x in values]
    mean = statistics.fmean(values) if values else float("nan")
    if len(values) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean,
                "n": len(values), "stdev": 0.0}
    stdev = statistics.stdev(values)
    half = 1.96 * stdev / math.sqrt(len(values))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half,
            "n": len(values), "stdev": stdev}


def drafter_measurement(name, rows, host_drafts, static_tables):
    result = {}
    for k in KS:
        per_prompt = []
        all_accepted, all_emitted = [], []
        total_uart, total_rounds = 0, 0
        for row, host_tokens in zip(rows, host_drafts):
            pass_data = offline_passes(row["prompt_tokens"], row["sequence_tokens"],
                                       name, k, host_tokens, static_tables)
            per_prompt.append(sum(pass_data["emitted"]) / pass_data["passes"])
            all_accepted.extend(pass_data["accepted"])
            all_emitted.extend(pass_data["emitted"])
            total_uart += pass_data["uart_bytes"]
            total_rounds += pass_data["passes"]
        row_out = {
            "label": "MEASURED_OFFLINE",
            "mean_tokens_per_pass": mean_ci(per_prompt),
            "mean_accepted_tokens": statistics.fmean(all_accepted),
            "distribution_accepted_tokens": dict(sorted(Counter(all_accepted).items())),
            "distribution_emitted_tokens": dict(sorted(Counter(all_emitted).items())),
            "passes": len(all_emitted), "prompt_pass_means": per_prompt,
        }
        if name == "host_tiny":
            row_out["uart"] = {
                "bytes_per_round": 1 + 2 * k + 2,
                "total_bytes": total_uart,
                "mean_bytes_per_round": total_uart / total_rounds,
                "baud_assumption": "115200 8N1; 10 bit-times/byte",
            }
        result[str(k)] = row_out
    return result


def host_drafts(rows, target_tok):
    """Generate the small 260K continuation once per prompt on the host."""
    prompt_tokens = [row["prompt_tokens"] for row in rows]
    if HOST_DRAFT_PATH.exists():
        try:
            cached = json.loads(HOST_DRAFT_PATH.read_text(encoding="utf8"))
            if (cached.get("status") == "pass" and
                    cached.get("generation_steps") == 256 and
                    cached.get("prompt_tokens") == prompt_tokens):
                return cached["draft_tokens"]
        except (OSError, ValueError, KeyError):
            pass
    _, tiny = load_runtime_260k(4)
    tiny_tok = Tokenizer(ROOT / "models/llama2c/stories260K/tok512.bin", 512)
    outputs = []
    # 256 draft steps covers the full 120-token 32K-tokenizer continuation even
    # after retokenization expansion differences across the two tokenizers.
    for index, row in enumerate(rows):
        tiny_prompt = tiny_tok.encode(target_tok.decode(row["prompt_tokens"]))
        tiny.reset()
        for pos, token in enumerate(tiny_prompt[:-1]):
            tiny.forward_argmax(token, pos, 4096)
        cur = tiny_prompt[-1]
        tiny_out = []
        for pos in range(256):
            nxt = int(tiny.forward_argmax(cur, len(tiny_prompt) - 1 + pos, 4096))
            tiny_out.append(nxt)
            cur = nxt
        retokenized = target_tok.encode(tiny_tok.decode(tiny_out), bos=False)
        outputs.append(retokenized)
        print(f"host_tiny {index + 1}/{len(rows)}: {len(retokenized)} target tokens",
              flush=True)
    _write_json(HOST_DRAFT_PATH, {"status": "pass", "generation_steps": 256,
                                  "prompt_tokens": prompt_tokens,
                                  "draft_tokens": outputs,
                                  "model": "stories260K",
                                  "retokenized_with": "tokenizer.bin"})
    return outputs


def speed_model(acceptance):
    stream = 10_684_413
    compute_one = 12_338_619
    rates = (0.25, 0.5, 1.0)
    pruned_stream = (5_900_000, 6_600_000)
    out = {"label": "DERIVED", "clock_hz": 50_000_000,
           "baseline": {"stream_bytes_per_token": stream,
                         "compute_cycles_per_token": compute_one},
           "full_vocab": {}, "pruned_vocab_stream_bytes_per_token":
           list(pruned_stream), "pruned_vocab": {}}
    for k in KS:
        krow = {}
        mean_tokens = acceptance["combined"][str(k)]["mean_tokens_per_pass"]
        for rate in rates:
            stream_cycles = stream / rate
            compute_cycles = compute_one * (k + 1)
            cycles = max(stream_cycles, compute_cycles)
            krow[str(rate)] = {
                "tokens_per_pass": mean_tokens,
                "stream_cycles": stream_cycles,
                "compute_cycles": compute_cycles,
                "cycles_per_pass": cycles,
                "tok_s": {"mean": mean_tokens["mean"] * 50_000_000 / cycles,
                          "ci95_low": mean_tokens["ci95_low"] * 50_000_000 / cycles,
                          "ci95_high": mean_tokens["ci95_high"] * 50_000_000 / cycles},
            }
        out["full_vocab"][str(k)] = krow
        pkrow = {}
        for rate in rates:
            compute_cycles = compute_one * (k + 1)
            low_stream = pruned_stream[0] / rate
            high_stream = pruned_stream[1] / rate
            low_cycles, high_cycles = max(low_stream, compute_cycles), max(high_stream, compute_cycles)
            pkrow[str(rate)] = {
                "tokens_per_pass": mean_tokens,
                "stream_cycles_range": [low_stream, high_stream],
                "compute_cycles": compute_cycles,
                "cycles_per_pass_range": [low_cycles, high_cycles],
                "tok_s_range_mean": [mean_tokens["mean"] * 50_000_000 / high_cycles,
                                     mean_tokens["mean"] * 50_000_000 / low_cycles],
            }
        out["pruned_vocab"][str(k)] = pkrow
    return out


def main():
    prompt_rows, tok = prompts50()
    rows, greedy_wall = collect_sequences(prompt_rows)
    static_tables, static_meta = build_static_tables(tok)
    host = host_drafts(rows, tok)

    names = [f"context{n}" for n in (1, 2, 3, 4)] + [
        "static_bigram", "static_trigram", "host_tiny", "combined"]
    acceptance = {name: drafter_measurement(name, rows, host, static_tables)
                  for name in names}
    acceptance_meta = {
        "label": "MEASURED_OFFLINE",
        "prompt_count": len(rows), "generation_target": MAX_SEQ,
        "drafters": names, "k_values": list(KS),
        "static_tables": static_meta,
        "host_tiny": {"model": "stories260K", "generation_steps": 256,
                       "retokenized_with": "models/llama2c/tokenizer.bin"},
    }
    speed = speed_model(acceptance)
    study = {
        "status": "pass", "labels": {"acceptance": "MEASURED_OFFLINE",
        "greedy_wall_time": "MEASURED", "speed": "DERIVED",
        "hardware": "MEASURED", "estimates": "ESTIMATED"},
        "python_requested": PYTHON_REQUESTED,
        "python_used": sys.executable, "python_version": sys.version,
        "numpy": np.__version__, "model": "stories15M",
        "board_config": {"kv_bits": 4, "max_seq": MAX_SEQ,
                         "clock_hz": 50_000_000},
        "prompt_selection": {"dataset": "TinyStories validation story openings",
                              "seed": SEED, "prompt_count": len(rows),
                              "listed_in": str(SEQUENCE_PATH.relative_to(ROOT))},
        "greedy_sequences": {"artifact": str(SEQUENCE_PATH.relative_to(ROOT)),
                             "wall_seconds": greedy_wall,
                             "per_prompt_wall_seconds":
                             [r["wall_seconds"] for r in rows]},
        "acceptance_measurement": acceptance_meta,
        "acceptance": acceptance,
        "speed_model": speed,
        "hardware": {"source": "bench/results/boolean_top_impl.json",
                     "dsp_used": 73, "dsp_limit": 120,
                     "bram36_used": 67, "bram36_limit": 75,
                     "stream_bytes_per_token": 10_684_413,
                     "compute_cycles_per_token": 12_338_619},
        "design_points": [
            {"name": "k1_time_multiplexed", "k": 1,
             "dsp48": "73 measured base; 73 estimated with time multiplexing",
             "bram36": "71-75 ESTIMATED; base 67 measured, limit 75",
             "verify_compute_cycles": 24_677_238,
             "fit_assessment": "only plausible speculative point; BRAM margin near zero"},
            {"name": "k3_time_multiplexed", "k": 3,
             "dsp48": "73 measured base; 73 estimated with time multiplexing",
             "bram36": "75-83 ESTIMATED; likely over limit",
             "verify_compute_cycles": 49_354_476,
             "fit_assessment": "not recommended"},
        ],
        "recommendation": {
            "decision": "do not build speculative RTL",
            "best_candidate": "k=1 time-multiplexed, if a follow-up build is forced",
            "pruned_vocab_label": "ESTIMATED",
        },
        "rtl_simulation": {"status": "NOT RUN", "reason":
                           "No speculative RTL was requested or changed"},
        "not_run": {"speculative_rtl_build": "NOT RUN",
                    "board_uart_measurement": "NOT RUN",
                    "pruned_vocabulary_hardware_measurement": "NOT RUN"},
        "provenance": ["bench/results/decode_e2e_rtl_15m_kv4_p1.json",
                       "bench/results/boolean_top_impl.json",
                       "bench/results/specdec_sequences.json"],
    }
    _write_json(STUDY_PATH, study)
    print(json.dumps({"status": "pass", "prompt_count": len(rows),
                      "greedy_wall_seconds": greedy_wall,
                      "drafters": names}, indent=2))


if __name__ == "__main__":
    main()
