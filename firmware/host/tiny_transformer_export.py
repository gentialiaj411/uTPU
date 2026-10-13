"""Scale/exponent export records for the Phase-1 integer reference.

All activation scales are powers of two.  Weight scales are symmetric INT8
per output channel; their effect is represented by existing multiplier/shift
records at each linear boundary.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from requantization import choose_multiplier_and_shift


CALIBRATION_PERCENTILE = 100.0  # max: no train-calibration clipping is permitted in v1.


def choose_activation_exponent(values: np.ndarray, *, percentile: float = CALIBRATION_PERCENTILE) -> int:
    """Smallest exponent e with percentile(|values|) / 2**e <= 127."""
    bound = float(np.percentile(np.abs(np.asarray(values, dtype=np.float64)), percentile))
    if bound == 0.0:
        return -31
    return int(math.ceil(math.log2(bound / 127.0)))


def quantize_activation(values: np.ndarray, exponent: int) -> np.ndarray:
    """Quantize a power-of-two-scaled activation with ties-to-even rounding."""
    code = np.rint(np.asarray(values, dtype=np.float64) / (2.0 ** int(exponent)))
    return np.clip(code, -128, 127).astype(np.int8)


def quantize_weight_per_output_channel(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize [out, in] weights symmetrically, one real scale per out row."""
    value = np.asarray(weight, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("weight must have shape [out, in]")
    scales = np.max(np.abs(value), axis=1) / 127.0
    scales = np.where(scales == 0.0, 1.0, scales)
    codes = np.clip(np.rint(value / scales[:, None]), -128, 127).astype(np.int8)
    return codes, scales.astype(np.float64)


def linear_record(*, input_exponent: int, weight_scales: np.ndarray, output_exponent: int) -> dict[str, Any]:
    """Per-output multiplier/shift records realizing s_in*s_w/s_out."""
    ratio = (2.0 ** int(input_exponent)) * np.asarray(weight_scales, dtype=np.float64) / (2.0 ** int(output_exponent))
    params = [choose_multiplier_and_shift(float(item)) for item in ratio]
    return {
        "input_exponent": int(input_exponent),
        "output_exponent": int(output_exponent),
        "multipliers": [int(item[0]) for item in params],
        "right_shifts": [int(item[1]) for item in params],
    }


def _rshift_rne(values: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Signed right shift with the frozen nearest-even rule, broadcast by channel."""
    value = np.asarray(values, dtype=np.int64)
    amount = np.asarray(shift, dtype=np.int64)
    magnitude = np.abs(value)
    quotient = np.right_shift(magnitude, amount)
    remainder = magnitude - np.left_shift(quotient, amount)
    half = np.left_shift(np.ones_like(amount, dtype=np.int64), amount - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    return np.where(value < 0, -(quotient + increment), quotient + increment)


def linear_int8_rne(input_code: np.ndarray, weight_code: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    """INT8 GEMM plus exported per-output RNE multiplier/shift requantization."""
    accum = np.asarray(input_code, dtype=np.int64) @ np.asarray(weight_code, dtype=np.int64).T
    multiplier = np.asarray(record["multipliers"], dtype=np.int64)
    shift = np.asarray(record["right_shifts"], dtype=np.int64)
    scaled = _rshift_rne(accum * multiplier, shift)
    return np.clip(scaled, -128, 127).astype(np.int8)
