"""Adversarial input distributions (Task 2 hardening pass, 2026-05-25).

Provides a small library of input-tensor distributions the fuzzer can
sample. Each distribution targets a numeric regime where compilers
sometimes diverge from the reference:

* ``random_normal``        — N(0, 1) noise; the workhorse default.
* ``zeros``                — all-zero inputs (exposes degenerate matmul
                              shortcuts, dead-code paths).
* ``all_positive``         — uniform in ``[0, 1]`` (RELU is identity, so
                              the fused vs unfused output should still
                              match — useful for cross-checking the
                              fusion relation).
* ``alternating_sign``     — ``+a, -a, +a, ...`` interleaved (stresses
                              accumulation order in matmul).
* ``minmax_float_boundary``— ``+1.0``/``-1.0`` extremes (saturation
                              candidate for tanh-like / scaled paths).
* ``sparse``               — mostly zeros with a few non-zero entries
                              (stresses prefetch / launch overhead).

Every sampler is **deterministic** given a `(seed, distribution)` pair.
Callers can ask for the same distribution / seed and always get the same
tensor — important for reproducing a divergence by name.

`apply_to_program` replaces a `GeneratedProgram`'s input list with
distribution-sampled tensors of the same shape / dtype, and records
which distribution was used in the returned program's metadata copy.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fuzz.graph_generator import GeneratedProgram


DISTRIBUTIONS: Tuple[str, ...] = (
    "random_normal",
    "zeros",
    "all_positive",
    "alternating_sign",
    "minmax_float_boundary",
    "sparse",
)


def sample_tensor(
    distribution: str,
    shape: Sequence[int],
    seed: int,
) -> np.ndarray:
    """Sample one tensor from the named distribution.

    Deterministic: same ``(distribution, shape, seed)`` always returns the
    same ndarray (bit-identical).
    """
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {distribution!r}; valid: {DISTRIBUTIONS!r}")
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    shape_tup = tuple(int(d) for d in shape)
    if distribution == "random_normal":
        return rng.standard_normal(size=shape_tup).astype(np.float32)
    if distribution == "zeros":
        return np.zeros(shape_tup, dtype=np.float32)
    if distribution == "all_positive":
        return rng.uniform(low=0.0, high=1.0, size=shape_tup).astype(np.float32)
    if distribution == "alternating_sign":
        # Stable alternating pattern + tiny noise so it isn't exactly periodic
        size = int(np.prod(shape_tup)) if shape_tup else 1
        base = np.where(np.arange(size) % 2 == 0, 1.0, -1.0).astype(np.float32)
        noise = rng.uniform(low=-0.05, high=0.05, size=size).astype(np.float32)
        return (base + noise).reshape(shape_tup)
    if distribution == "minmax_float_boundary":
        size = int(np.prod(shape_tup)) if shape_tup else 1
        choices = rng.choice([-1.0, 1.0], size=size).astype(np.float32)
        return choices.reshape(shape_tup)
    if distribution == "sparse":
        # ~10% non-zero entries, drawn from N(0, 1); rest are exact zeros.
        size = int(np.prod(shape_tup)) if shape_tup else 1
        mask = rng.uniform(size=size) < 0.1
        values = rng.standard_normal(size=size).astype(np.float32)
        out = np.where(mask, values, 0.0).astype(np.float32)
        return out.reshape(shape_tup)
    raise AssertionError(f"unhandled distribution {distribution!r}")


def apply_to_program(
    program: GeneratedProgram,
    distribution: str,
    seed: int,
) -> GeneratedProgram:
    """Return a copy of ``program`` with ``inputs`` resampled from ``distribution``.

    Shapes / dtypes are preserved. The metadata dict is shallow-copied and
    annotated with ``input_distribution`` so downstream coverage tracking
    knows which distribution was actually used.
    """
    new_inputs: List[np.ndarray] = []
    for i, arr in enumerate(program.inputs):
        shape = tuple(arr.shape)
        sub_seed = (int(seed) ^ (hash((distribution, i)) & 0x7FFFFFFF)) & 0x7FFFFFFF
        new_inputs.append(sample_tensor(distribution, shape, sub_seed))
    new_metadata = dict(program.metadata)
    new_metadata["input_distribution"] = distribution
    return GeneratedProgram(
        seed=program.seed,
        graph=program.graph,
        inputs=new_inputs,
        metadata=new_metadata,
    )


def pick_distribution(rng: random.Random) -> str:
    """Weighted choice over the distribution list.

    `random_normal` is weighted heavily (it's the default for most graphs);
    the adversarial distributions are weighted low enough that a 50k-graph
    corpus still hits each of them with several hundred samples.
    """
    weights = {
        "random_normal": 60,
        "zeros": 8,
        "all_positive": 8,
        "alternating_sign": 8,
        "minmax_float_boundary": 8,
        "sparse": 8,
    }
    names = list(DISTRIBUTIONS)
    ws = [weights[n] for n in names]
    return rng.choices(names, weights=ws, k=1)[0]


def coverage_summary(distributions_used: Sequence[str]) -> Dict[str, Any]:
    """Aggregate distribution-usage coverage across a corpus."""
    counts: Dict[str, int] = {d: 0 for d in DISTRIBUTIONS}
    for d in distributions_used:
        if d in counts:
            counts[d] += 1
    return {
        "input_distributions_covered": sorted({d for d in distributions_used if d in counts}),
        "input_distribution_counts": counts,
    }
