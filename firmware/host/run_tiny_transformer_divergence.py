"""Report float versus independent-integer divergence for one validation batch.

This intentionally does not invent missing integer scale metadata.  Until the
export records a power-of-two exponent at every stage, comparison uses a least
squares diagnostic scale solely for error visibility; the report labels it as
such and retains raw-code utilization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tiny_transformer_model import TinyDecoder
from tiny_transformer_reference import TinyTransformerConfig, integer_decoder_trace


HERE = Path(__file__).resolve().parent


def _fixture(name: str) -> dict:
    return json.loads((HERE / "fixtures" / name).read_text(encoding="utf-8"))


def _metric(float_value: np.ndarray, int_code: np.ndarray) -> dict[str, float | str]:
    f = np.asarray(float_value, dtype=np.float64).reshape(-1)
    q = np.asarray(int_code, dtype=np.float64).reshape(-1)
    finite = np.isfinite(f)
    f, q = f[finite], q[finite]
    denom = float(q @ q)
    fitted_scale = 0.0 if denom == 0.0 else float((f @ q) / denom)
    reconstructed = q * fitted_scale
    norm = float(np.linalg.norm(f) * np.linalg.norm(q))
    cosine = 1.0 if norm == 0.0 and np.array_equal(f, q) else (0.0 if norm == 0.0 else float((f @ q) / norm))
    return {
        "cosine": cosine,
        "max_abs_error_after_diagnostic_fit": float(np.max(np.abs(f - reconstructed))) if len(f) else 0.0,
        "diagnostic_fitted_scale_not_export_scale": fitted_scale,
        "integer_exponent": "MISSING_FROM_CURRENT_EXPORT",
        "max_abs_code": int(np.max(np.abs(q))) if len(q) else 0,
        "pct_codes_in_neg1_zero_pos1": float(100.0 * np.mean(np.abs(q) <= 1)) if len(q) else 100.0,
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
    model.load_state_dict({name: torch.from_numpy(value) for name, value in checkpoint["state_dict"].items()}); model.eval()
    stoi = {c: i for i, c in enumerate(vocab["characters"])}
    data = np.asarray([stoi[c] for c in a.corpus.read_text(encoding="utf-8")], dtype=np.int64)
    starts = np.arange(split["validation_start"], split["validation_start"] + a.batch_size * cfg.T, cfg.T)
    tokens = np.stack([data[s:s + cfg.T] for s in starts])
    _, float_trace = model.forward_with_trace(torch.from_numpy(tokens))
    integer_trace = integer_decoder_trace(tokens, checkpoint["qstate"], cfg)
    stages = {name: _metric(float_trace[name].cpu().numpy(), integer_trace[name]) for name in integer_trace}
    payload = {
        "status": "diagnostic_only_missing_export_exponents",
        "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "batch_starts": starts.tolist(),
        "stages": stages,
        "first_cosine_below_0_99": next((name for name, item in stages.items() if float(item["cosine"]) < 0.99), None),
        "warning": "Current qstate has per-tensor float scales, but integer_decoder_trace discards them and has no exported stage exponent. This is an export/semantics defect, not evidence for QAT.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"first_cosine_below_0_99": payload["first_cosine_below_0_99"], "stages": stages}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
