from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


ROUNDING_MODE = "arithmetic_right_shift_truncation"
MAX_REQUANT_MULTIPLIER = 0xFFFF
MAX_REQUANT_RIGHT_SHIFT = 31


@dataclass(frozen=True)
class RequantParams:
    multiplier: int
    right_shift: int
    enable: bool = True

    def as_dict(self) -> dict:
        return {
            "enable": bool(self.enable),
            "multiplier": int(self.multiplier),
            "right_shift": int(self.right_shift),
            "rounding_mode": ROUNDING_MODE,
        }


def clip_signed(x: int, width: int) -> int:
    hi = (1 << (width - 1)) - 1
    lo = -(1 << (width - 1))
    if x > hi:
        return hi
    if x < lo:
        return lo
    return int(x)


def requantize_value(
    acc: int,
    *,
    multiplier: int,
    right_shift: int,
    out_width: int,
) -> int:
    if int(multiplier) < 0 or int(multiplier) > MAX_REQUANT_MULTIPLIER:
        raise ValueError(f"multiplier out of range: {multiplier}")
    if int(right_shift) < 0 or int(right_shift) > MAX_REQUANT_RIGHT_SHIFT:
        raise ValueError(f"right_shift out of range: {right_shift}")
    product = int(acc) * int(multiplier)
    scaled = product >> int(right_shift) if int(right_shift) > 0 else product
    return clip_signed(scaled, out_width)


def requantize_array(
    values: np.ndarray,
    *,
    multiplier: int,
    right_shift: int,
    out_width: int,
    dtype: np.dtype,
) -> np.ndarray:
    vec = np.vectorize(
        lambda x: requantize_value(
            int(x),
            multiplier=multiplier,
            right_shift=right_shift,
            out_width=out_width,
        ),
        otypes=[np.int64],
    )
    return vec(np.asarray(values, dtype=np.int64)).astype(dtype)


def choose_multiplier_and_shift(
    scale: float,
    *,
    max_multiplier: int = MAX_REQUANT_MULTIPLIER,
    max_right_shift: int = MAX_REQUANT_RIGHT_SHIFT,
) -> Tuple[int, int]:
    if scale <= 0.0:
        return 1, 0
    best_multiplier = 1
    best_shift = 0
    best_error = abs(scale - 1.0)
    for shift in range(max_right_shift + 1):
        candidate = int(round(scale * (1 << shift)))
        if candidate <= 0 or candidate > max_multiplier:
            continue
        approx = float(candidate) / float(1 << shift)
        error = abs(scale - approx)
        if error < best_error:
            best_multiplier = int(candidate)
            best_shift = int(shift)
            best_error = error
    return best_multiplier, best_shift


def symmetric_scale(values: np.ndarray, *, qmax: int) -> float:
    max_abs = float(np.max(np.abs(np.asarray(values, dtype=np.float32))))
    if max_abs == 0.0:
        return 1.0
    return max_abs / float(qmax)


def quantize_symmetric(values: np.ndarray, *, bits: int, scale: float) -> np.ndarray:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    q = np.round(np.asarray(values, dtype=np.float32) / float(scale))
    return np.clip(q, qmin, qmax).astype(np.int16 if bits > 4 else np.int8)
