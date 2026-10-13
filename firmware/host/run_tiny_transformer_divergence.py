"""Report float versus schema-scaled independent-integer stage divergence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from tiny_transformer_model import TinyDecoder
from tiny_transformer_reference import TinyTransformerConfig, integer_decoder_trace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _fixture(name: str) -> dict:
    return json.loads((HERE / "fixtures" / name).read_text(encoding="utf-8"))


def _source_tree() -> dict[str, str]:
    return {path: subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True).strip() for path in ("firmware/host", "tools")}


def _dequant(name: str, code: np.ndarray, acts: dict[str, int]) -> np.ndarray:
    if name == "softmax_probs":
        return np.asarray(code, dtype=np.float64) / 255.0
    return np.asarray(code, dtype=np.float64) * (2.0 ** int(acts[name]))


def _metric(float_value: np.ndarray, int_code: np.ndarray, exponent: int | str) -> dict[str, float | int | str]:
    f = np.asarray(float_value, dtype=np.float64).reshape(-1)
    q = np.asarray(int_code, dtype=np.float64).reshape(-1)
    finite = np.isfinite(f)
    f, q = f[finite], q[finite]
    norm = float(np.linalg.norm(f) * np.linalg.norm(q))
    cosine = 1.0 if norm == 0.0 and np.array_equal(f, q) else (0.0 if norm == 0.0 else float((f @ q) / norm))
    return {
        "cosine": cosine,
        "max_abs_error": float(np.max(np.abs(f - q))) if len(f) else 0.0,
        "integer_exponent": exponent,
        "max_abs_code": int(np.max(np.abs(int_code.astype(np.int64)))) if int_code.size else 0,
        "pct_codes_in_neg1_zero_pos1": float(100.0 * np.mean(np.abs(int_code.astype(np.int64)) <= 1)) if int_code.size else 100.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("build/tiny_transformer/input.txt"))
    p.add_argument("--checkpoint", type=Path, default=Path("bench/results/tiny_transformer_numerics.pt"))
    p.add_argument("--output", type=Path, default=Path("bench/results/tiny_transformer_divergence.json"))
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    vocab, split = _fixture("tiny_transformer_vocab.json"), _fixture("tiny_transformer_split.json")
    if hashlib.sha256(a.corpus.read_bytes()).hexdigest() != vocab["corpus_sha256"]:
        raise SystemExit("corpus SHA-256 mismatch")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = TinyTransformerConfig(**checkpoint["config"])
    model = TinyDecoder(cfg.T, cfg.D, cfg.H, cfg.Dff, cfg.L, cfg.vocab)
    model.load_state_dict({name: torch.from_numpy(value) for name, value in checkpoint["state_dict"].items()})
    model.eval()
    stoi = {c: i for i, c in enumerate(vocab["characters"])}
    data = np.asarray([stoi[c] for c in a.corpus.read_text(encoding="utf-8")], dtype=np.int64)
    starts = np.arange(split["validation_start"], split["validation_start"] + a.batch_size * cfg.T, cfg.T)
    tokens = np.stack([data[s:s + cfg.T] for s in starts])
    _, float_trace = model.forward_with_trace(torch.from_numpy(tokens))
    schema = checkpoint["export_schema"]
    integer_trace = integer_decoder_trace(tokens, checkpoint["qstate"], schema, cfg)
    stages = {
        name: _metric(
            float_trace[name].cpu().numpy(),
            _dequant(name, integer_trace[name], schema["activations"]),
            "p/255" if name == "softmax_probs" else int(schema["activations"][name]),
        )
        for name in integer_trace
    }
    payload = {
        "status": "phase1_export_scale_aware", "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "batch_starts": starts.tolist(), "stages": stages, "source_tree": _source_tree(),
        "first_cosine_below_0_99": next((name for name, item in stages.items() if float(item["cosine"]) < 0.99), None),
        "non_sparse_exemptions": ["softmax_probs masked entries", "leaky_relu negative branch"],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"first_cosine_below_0_99": payload["first_cosine_below_0_99"], "stages": stages}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
