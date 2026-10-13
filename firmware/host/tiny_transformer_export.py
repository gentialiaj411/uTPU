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
