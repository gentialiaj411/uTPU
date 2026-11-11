"""Verify 64 cached NumPy decode steps against pinned upstream PyTorch model.py."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .reference import Llama2CModel
from .torch_oracle import load_torch_model


def verify(path: str, tokens: int = 64) -> dict:
    ref = Llama2CModel(path)
    model = load_torch_model(ref)
    ids = np.asarray([(i * 17 + 1) % ref.config.vocab_size for i in range(tokens)], dtype=np.int64)
    # Supplying targets makes pinned model.py return logits for every position.
    with torch.inference_mode():
        logits = model(torch.from_numpy(ids)[None], torch.from_numpy(ids)[None]).cpu().numpy()[0]
    diffs = []
    for pos, token in enumerate(ids):
        diffs.append(float(np.max(np.abs(ref.forward(int(token), pos) - logits[pos]))))
    result = {"tokens": tokens, "max_abs_logit_diff": max(diffs), "threshold": 1e-4,
              "passed": max(diffs) < 1e-4}
    if not result["passed"]:
        raise AssertionError(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    print(json.dumps(verify(args.checkpoint), indent=2))
