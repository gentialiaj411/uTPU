"""Independent TILE_GEMM golden model — written from the frozen Part A spec.

Source of truth: ``docs/descriptor_isa_part_a_design.md`` §2 (encoding) + §3
(sequencer semantics). This module must NOT import or call
``lowering_blocked_fc_utpu`` / ``program_loader`` emission helpers; those are
comparison oracles, not parents of this model.

Semantics (hoist_payloads=1, Part A v1):
  for ob in range(out_blocks):
      for ib in range(in_blocks):
          W = weight_tile[ob, ib]          # (N, N)
          X = input_tile[ib]               # (N, B) rows=feature, cols=batch
          acc = W @ X if (acc_clear_first and ib==0) else acc + W @ X
      if finalize_last:
          Y[ob] = quantize_clip(acc)       # no ReLU in v1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from isa_encoder import (
    TILE_GEMM_WORD_COUNT,
    IsaConfig,
    decodeTileGemmWords,
    encodeTileGemm,
    pack_values_to_word,
)


def quantize_clip(x: np.ndarray, compute_data_width: int) -> np.ndarray:
    """Spec finalize v1: saturate to signed compute_data_width (no ReLU, no requant mul)."""
    hi = (1 << (compute_data_width - 1)) - 1
    lo = -(1 << (compute_data_width - 1))
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi).astype(np.int8)


def clip_to_width(x: int, width: int) -> int:
    hi = (1 << (width - 1)) - 1
    lo = -(1 << (width - 1))
    if x > hi:
        return hi
    if x < lo:
        return lo
    return int(x)


@dataclass(frozen=True)
class TileGemmDesc:
    """Decoded TILE_GEMM fields (frozen layout)."""

    weight_base: int
    input_base: int
    result_base: int
    out_blocks: int
    in_blocks: int
    batch_size: int
    array_size: int
    dtype_code: int
    acc_clear_first: bool = True
    finalize_last: bool = True
    hoist_payloads: bool = True
    weight_overlap: bool = False

    @classmethod
    def from_words(cls, words: Sequence[int], cfg: IsaConfig) -> "TileGemmDesc":
        d = decodeTileGemmWords(words, cfg=cfg)
        return cls(
            weight_base=int(d["weight_base"]),
            input_base=int(d["input_base"]),
            result_base=int(d["result_base"]),
            out_blocks=int(d["out_blocks"]),
            in_blocks=int(d["in_blocks"]),
            batch_size=int(d["batch_size"]),
            array_size=int(d["array_size"]),
            dtype_code=int(d["dtype_code"]),
            acc_clear_first=bool(d["acc_clear_first"]),
            finalize_last=bool(d["finalize_last"]),
            hoist_payloads=bool(d["hoist_payloads"]),
            weight_overlap=bool(d["weight_overlap"]),
        )


def encode_desc(desc: TileGemmDesc, cfg: IsaConfig) -> bytes:
    return encodeTileGemm(
        weight_base=desc.weight_base,
        input_base=desc.input_base,
        result_base=desc.result_base,
        out_blocks=desc.out_blocks,
        in_blocks=desc.in_blocks,
        batch_size=desc.batch_size,
        array_size=desc.array_size,
        cfg=cfg,
        acc_clear_first=desc.acc_clear_first,
        finalize_last=desc.finalize_last,
        hoist_payloads=desc.hoist_payloads,
        weight_overlap=desc.weight_overlap,
    )


def hoist_layout_words(
    *,
    array_size: int,
    batch_size: int,
    items_per_word: int,
) -> Tuple[int, int, int]:
    """(weight_words_per_tile, input_words_per_tile, result_words_per_out_block)."""
    n = int(array_size)
    b = int(batch_size)
    items = int(items_per_word)
    w_words = (n * n) // items
    x_words = (n * b) // items
    out_chunks = max(1, (b + n - 1) // n)
    y_words = ((n * n) // items) * out_chunks
    return w_words, x_words, y_words


def pack_result_bytes(
    y_blocks: Sequence[np.ndarray],
    *,
    array_size: int,
    batch_size: int,
    compute_data_width: int,
) -> List[int]:
    """Pack finalized out-blocks to FETCH lo/hi byte stream for live batch columns."""
    n = int(array_size)
    b = int(batch_size)
    width = int(compute_data_width)
    items = 16 // width
    cfg = IsaConfig(address_width=12, compute_data_width=width, descriptor_isa=True)
    fetch_words = (b * n) // items
    out: List[int] = []
    for block in y_blocks:
        flat = np.zeros(n * n * max(1, (b + n - 1) // n), dtype=np.int8)
        for col in range(b):
            for row in range(n):
                flat[col * n + row] = int(block[row, col])
        for widx in range(fetch_words):
            chunk = [int(x) for x in flat[widx * items : (widx + 1) * items]]
            while len(chunk) < items:
                chunk.append(0)
            word = pack_values_to_word(chunk, cfg)
            out.append(word & 0xFF)
            out.append((word >> 8) & 0xFF)
    return out


def golden_tile_gemm(
    weights: np.ndarray,
    activations: np.ndarray,
    *,
    array_size: int,
    compute_data_width: int = 4,
    acc_clear_first: bool = True,
    finalize_last: bool = True,
) -> Tuple[List[np.ndarray], List[int]]:
    """Spec-only TILE_GEMM: tensor math → finalized blocks + fetch bytes."""
    w = np.asarray(weights, dtype=np.int64)
    x = np.asarray(activations, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("activations must be (batch, in_features)")
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    batch = int(x.shape[0])
    if int(x.shape[1]) != in_f:
        raise ValueError("activation in_features mismatch")
    n = int(array_size)
    out_blocks = (out_f + n - 1) // n
    in_blocks = (in_f + n - 1) // n
    w_pad = np.zeros((out_blocks * n, in_blocks * n), dtype=np.int64)
    x_pad = np.zeros((batch, in_blocks * n), dtype=np.int64)
    w_pad[:out_f, :in_f] = w
    x_pad[:, :in_f] = x

    y_blocks: List[np.ndarray] = []
    for ob in range(out_blocks):
        acc = np.zeros((n, batch), dtype=np.int64)
        for ib in range(in_blocks):
            wt = w_pad[ob * n : (ob + 1) * n, ib * n : (ib + 1) * n]
            xt = x_pad[:, ib * n : (ib + 1) * n].T
            prod = wt @ xt
            if acc_clear_first and ib == 0:
                acc = prod
            else:
                acc = acc + prod
        if finalize_last:
            y_blocks.append(quantize_clip(acc, compute_data_width))
        else:
            y_blocks.append(acc.astype(np.int8))

    fetch_bytes = pack_result_bytes(
        y_blocks,
        array_size=n,
        batch_size=batch,
        compute_data_width=compute_data_width,
    )
    return y_blocks, fetch_bytes


__all__ = [
    "TILE_GEMM_WORD_COUNT",
    "TileGemmDesc",
    "clip_to_width",
    "encode_desc",
    "golden_tile_gemm",
    "hoist_layout_words",
    "pack_result_bytes",
    "quantize_clip",
]
