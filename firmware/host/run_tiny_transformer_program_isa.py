"""Run the complete lowered tiny-decoder image against all 512 validation windows."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
HOST = ROOT / "firmware" / "host"
sys.path.insert(0, str(HOST))

from isa_encoder import encodeHalt
from isa_simulator import UTPUISASimulator
from lower_tiny_transformer import CFG, M, _pack, _words, build_program, build_program_stages, build_weight_image, image_hash
from tiny_transformer_reference import TinyTransformerConfig, integer_decoder_trace

OUT = ROOT / "bench" / "results" / "tiny_transformer_program_isa.json"


def _lfin(words: list[int], base: int, rows: int, cols: int) -> np.ndarray:
    stored_cols = ((cols + 7) // 8) * 8
    raw = np.frombuffer(np.asarray(words[base:base + rows * stored_cols // 2], dtype="<u2").tobytes(), dtype=np.int8)
    return np.asarray([raw[(col // 8 * rows + row) * 8 + col % 8]
                       for row in range(rows) for col in range(cols)], dtype=np.int8).reshape(rows, cols)


def _rm(words: list[int], base: int, rows: int, cols: int, *, unsigned: bool = False) -> np.ndarray:
    raw = np.frombuffer(np.asarray(words[base:base + rows * cols // 2], dtype="<u2").tobytes(),
                        dtype=np.uint8 if unsigned else np.int8)
    return raw.reshape(rows, cols)


def _qkv(trace: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([trace[name][0].transpose(1, 0, 2).reshape(32, 48)
                           for name in ("q", "k", "v")], axis=1)


def _target(words: list[int], name: str, head: int | None = None) -> np.ndarray | None:
    if name == "pos_add": return _rm(words, M.x, 32, 48)
    if name == "rmsnorm1": return _rm(words, M.r, 32, 48)
    if name == "qkv_to_rm": return _rm(words, M.qkv_r, 32, 144)
    if name.startswith("scores_head"): return _rm(words, M.score, 32, 32)
    if name.startswith("softmax_head"): return _rm(words, M.score, 32, 32, unsigned=True)
    if name.startswith("av_head"): return _rm(words, M.head + int(head) * 256, 32, 16)
    if name == "out_proj": return _lfin(words, M.out_l, 32, 48)
    if name == "residual1": return _rm(words, M.out_r, 32, 48)
    if name == "rmsnorm2": return _rm(words, M.r, 32, 48)
    if name == "ffn1_leaky_relu": return _lfin(words, M.hidden_l, 32, 192)
    if name == "ffn2": return _lfin(words, M.ffn2_l, 32, 48)
    if name == "residual2": return _rm(words, M.r, 32, 48)
    if name == "final_norm": return _rm(words, M.out_l, 32, 48)
    if name == "logits": return _lfin(words, M.logits_l, 32, 65)
    return None


def _expected(trace: dict[str, np.ndarray], name: str, head: int | None = None) -> np.ndarray | None:
    if name == "pos_add": return trace["pos_add"][0]
    if name == "rmsnorm1": return trace["rmsnorm1"][0]
    if name == "qkv_to_rm": return _qkv(trace)
    if name.startswith("scores_head"): return trace["scores"][0, int(head)]
    if name.startswith("softmax_head"): return trace["softmax_probs"][0, int(head)]
    if name.startswith("av_head"): return trace["av"][0, :, int(head) * 16:(int(head) + 1) * 16]
    if name == "out_proj": return trace["out_proj"][0]
    if name == "residual1": return trace["residual1"][0]
    if name == "rmsnorm2": return trace["rmsnorm2"][0]
    if name == "ffn1_leaky_relu": return trace["leaky_relu"][0]
    if name == "ffn2": return trace["ffn2"][0]
    if name == "residual2": return trace["residual2"][0]
    if name == "final_norm": return trace["final_norm"][0]
    if name == "logits": return trace["logits"][0]
    return None


def main() -> int:
    checkpoint = torch.load(ROOT / "bench" / "results" / "tiny_transformer_numerics.pt", map_location="cpu", weights_only=False)
    state, schema = checkpoint["qstate"], checkpoint["export_schema"]
    cfg = TinyTransformerConfig(**checkpoint["config"])
    vocab = json.loads((HOST / "fixtures" / "tiny_transformer_vocab.json").read_text(encoding="utf-8"))["characters"]
    data = np.asarray([vocab.index(char) for char in (ROOT / "build" / "tiny_transformer" / "input.txt").read_text(encoding="utf-8")], dtype=np.int64)
    split = json.loads((HOST / "fixtures" / "tiny_transformer_split.json").read_text(encoding="utf-8"))
    starts = range(split["validation_start"], split["validation_start"] + 512 * 32, 32)
    image, stages, program = build_weight_image(state, schema), build_program_stages(state, schema), build_program(state, schema)
    checked = {"pos_add", "rmsnorm1", "qkv_to_rm", "out_proj", "residual1", "rmsnorm2",
               "ffn1_leaky_relu", "ffn2", "residual2", "final_norm", "logits"}
    observed = {name: {"compared_elements": 0, "mismatches": 0} for name, _ in stages
                if name in checked or name.startswith(("scores_head", "softmax_head", "av_head"))}
    all_logits, labels = [], []
    halt = _words(encodeHalt(cfg=CFG))
    for start in starts:
        tokens, label = data[start:start + 32], data[start + 1:start + 33]
        sim = UTPUISASimulator(array_size=8, buffer_size=32768, cfg=CFG)
        sim.buffer[:len(image)] = image
        sim.buffer[M.x:M.x + 768] = _pack(state["token.weight"][tokens])
        trace = integer_decoder_trace(tokens, state, schema, cfg)
        for name, stage in stages:
            sim.run_words(stage + halt, preserve_buffer=True)
            head = int(name[-1]) if name.startswith(("scores_head", "softmax_head", "av_head")) else None
            actual, expected = _target(sim.buffer, name, head), _expected(trace, name, head)
            if actual is not None:
                observed[name]["compared_elements"] += int(actual.size)
                observed[name]["mismatches"] += int(np.count_nonzero(actual != expected))
        all_logits.append(_target(sim.buffer, "logits")); labels.append(label)
    logits, targets = np.stack(all_logits), np.stack(labels)
    logits_t = torch.from_numpy(logits.astype(np.float32) * (2.0 ** schema["activations"]["logits"]))
    targets_t = torch.from_numpy(targets)
    loss = float(F.cross_entropy(logits_t.reshape(-1, cfg.vocab), targets_t.reshape(-1), reduction="mean"))
    payload = {
        "status": "passed" if not any(item["mismatches"] for item in observed.values()) else "failed",
        "command": "python firmware/host/run_tiny_transformer_program_isa.py",
        "sequences": 512, "program_words": len(program), "weight_words": len(image),
        "buffer_peak_words": 31760, "program_sha256": image_hash(program), "weight_sha256": image_hash(image),
        "stages": observed,
        "metrics": {"top1_accuracy": float((logits_t.argmax(-1) == targets_t).float().mean()),
                    "bits_per_char": float(loss / np.log(2)), "tokens": int(targets.size)},
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
