"""Scale/exponent export records for the Phase-1 integer reference.

All activation scales are powers of two.  Weight scales are symmetric INT8
per output channel; their effect is represented by existing multiplier/shift
records at each linear boundary.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
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
    safe_amount = np.maximum(amount, 1)
    magnitude = np.abs(value)
    quotient = np.right_shift(magnitude, safe_amount)
    remainder = magnitude - np.left_shift(quotient, safe_amount)
    half = np.left_shift(np.ones_like(safe_amount, dtype=np.int64), safe_amount - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    rounded = np.where(value < 0, -(quotient + increment), quotient + increment)
    return np.where(amount == 0, value, rounded)


def linear_int8_rne(input_code: np.ndarray, weight_code: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    """INT8 GEMM plus exported per-output RNE multiplier/shift requantization."""
    accum = np.asarray(input_code, dtype=np.int64) @ np.asarray(weight_code, dtype=np.int64).T
    multiplier = np.asarray(record["multipliers"], dtype=np.int64)
    shift = np.asarray(record["right_shifts"], dtype=np.int64)
    scaled = _rshift_rne(accum * multiplier, shift)
    return np.clip(scaled, -128, 127).astype(np.int8)


def export_schema_document(*, calibration: dict[str, Any], activations: dict[str, int], linears: dict[str, dict[str, Any]], norms: dict[str, dict[str, Any]], residuals: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Canonical JSON payload and hash consumed by the independent reference."""
    if calibration.get("split") != "train":
        raise ValueError("activation calibration must be train-only")
    for name, exponent in activations.items():
        if not isinstance(exponent, int):
            raise ValueError(f"activation {name} lacks an explicit integer exponent")
    for name, record in residuals.items():
        if int(record["eo"]) < max(int(record["ea"]), int(record["eb"])):
            raise ValueError(f"residual {name} violates eo >= max(ea, eb)")
    payload = {"version": 1, "calibration": calibration, "activations": activations, "linears": linears, "norms": norms, "residuals": residuals}
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"payload": payload, "payload_sha256": hashlib.sha256(rendered).hexdigest()}


def write_export_schema(path: Path, document: dict[str, Any]) -> None:
    """Write the one-file JSON schema consumed by the reference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_export_schema(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document["payload"]
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if actual != document.get("payload_sha256"):
        raise ValueError("export schema payload hash mismatch")
    if payload.get("calibration", {}).get("split") != "train":
        raise ValueError("export schema is not train-calibrated")
    return payload


def calibrate_activation_exponents(train_traces: dict[str, np.ndarray], *, percentile: float = CALIBRATION_PERCENTILE) -> dict[str, int]:
    """Derive every activation exponent from train-only float trace samples."""
    if not train_traces:
        raise ValueError("train trace set is empty")
    return {name: choose_activation_exponent(values, percentile=percentile) for name, values in train_traces.items()}
