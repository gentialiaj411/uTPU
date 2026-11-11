"""Canonical prompt encoding shared by both sample generators."""
from __future__ import annotations
import numpy as np

PROMPTS = ('\n<s>\nSpot. Spot saw the', 'climb. She climbed trees,', '?"\nRoxy told Billy about')

def encode_prompts(tokenizer):
    return [np.asarray(tokenizer.encode(text, bos=False, eos=False), dtype=np.int64)
            for text in PROMPTS]
