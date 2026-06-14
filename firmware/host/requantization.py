from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


ROUNDING_MODE = "arithmetic_right_shift_truncation"
MAX_REQUANT_MULTIPLIER = 0xFFFF
MAX_REQUANT_RIGHT_SHIFT = 31


@dataclass(frozen=True)
class RequantParams:
    multiplier: int
    right_shift: int
    enable: bool = True
    per_channel_multipliers: Tuple[int, ...] | None = None
    per_channel_right_shifts: Tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.per_channel_multipliers is None and self.per_channel_right_shifts is None:
            _validate_requant_value(self.multiplier, "multiplier", max_value=MAX_REQUANT_MULTIPLIER)
            _validate_requant_value(self.right_shift, "right_shift", max_value=MAX_REQUANT_RIGHT_SHIFT)
            return
        if self.per_channel_multipliers is None or self.per_channel_right_shifts is None:
            raise ValueError("per_channel_multipliers and per_channel_right_shifts must be provided together")
        if len(self.per_channel_multipliers) != len(self.per_channel_right_shifts):
            raise ValueError("per-channel multiplier/shift vectors must have the same length")
        if len(self.per_channel_multipliers) == 0:
            raise ValueError("per-channel requant vectors must not be empty")
        for value in self.per_channel_multipliers:
            _validate_requant_value(value, "per_channel_multiplier", max_value=MAX_REQUANT_MULTIPLIER)
        for value in self.per_channel_right_shifts:
            _validate_requant_value(value, "per_channel_right_shift", max_value=MAX_REQUANT_RIGHT_SHIFT)

    @property
    def is_per_channel(self) -> bool:
        return self.per_channel_multipliers is not None

    @property
    def vector_length(self) -> int:
        return len(self.per_channel_multipliers) if self.per_channel_multipliers is not None else 1

    def block(self, start: int, count: int, *, pad_to: int | None = None) -> "RequantParams":
        if not self.is_per_channel:
            return self
        assert self.per_channel_multipliers is not None
        assert self.per_channel_right_shifts is not None
        if start < 0 or count < 0 or (start + count) > len(self.per_channel_multipliers):
            raise ValueError(
                f"invalid per-channel block slice start={start} count={count} length={len(self.per_channel_multipliers)}"
            )
        multipliers = list(self.per_channel_multipliers[start : start + count])
        shifts = list(self.per_channel_right_shifts[start : start + count])
        if pad_to is not None:
            while len(multipliers) < int(pad_to):
                multipliers.append(1)
                shifts.append(0)
        return RequantParams(
            multiplier=self.multiplier,
            right_shift=self.right_shift,
            enable=self.enable,
            per_channel_multipliers=tuple(int(v) for v in multipliers),
            per_channel_right_shifts=tuple(int(v) for v in shifts),
        )

    @classmethod
    def from_dict(cls, raw: dict) -> "RequantParams":
        if "per_channel_multipliers" in raw or "per_channel_right_shifts" in raw:
            return cls(
                multiplier=int(raw.get("multiplier", 1)),
                right_shift=int(raw.get("right_shift", 0)),
                enable=bool(raw.get("enable", True)),
                per_channel_multipliers=tuple(int(v) for v in raw.get("per_channel_multipliers", [])),
                per_channel_right_shifts=tuple(int(v) for v in raw.get("per_channel_right_shifts", [])),
            )
        return cls(
            multiplier=int(raw["multiplier"]),
            right_shift=int(raw["right_shift"]),
            enable=bool(raw.get("enable", True)),
        )

    def as_dict(self) -> dict:
        payload = {
            "enable": bool(self.enable),
            "multiplier": int(self.multiplier),
            "right_shift": int(self.right_shift),
            "rounding_mode": ROUNDING_MODE,
            "mode": "per_channel_symmetric" if self.is_per_channel else "per_layer_symmetric",
        }
        if self.is_per_channel:
            assert self.per_channel_multipliers is not None
            assert self.per_channel_right_shifts is not None
            payload["per_channel_multipliers"] = [int(v) for v in self.per_channel_multipliers]
            payload["per_channel_right_shifts"] = [int(v) for v in self.per_channel_right_shifts]
            payload["vector_length"] = len(self.per_channel_multipliers)
        return payload


def _validate_requant_value(value: int, name: str, *, max_value: int) -> None:
    if int(value) < 0 or int(value) > int(max_value):
        raise ValueError(f"{name} out of range: {value}")


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
    _validate_requant_value(multiplier, "multiplier", max_value=MAX_REQUANT_MULTIPLIER)
    _validate_requant_value(right_shift, "right_shift", max_value=MAX_REQUANT_RIGHT_SHIFT)
    product = int(acc) * int(multiplier)
    scaled = product >> int(right_shift) if int(right_shift) > 0 else product
    return clip_signed(scaled, out_width)


def requantize_array(
    values: np.ndarray,
    *,
    multiplier: int | Sequence[int],
    right_shift: int | Sequence[int],
    out_width: int,
    dtype: np.dtype,
    axis: int = -1,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if np.isscalar(multiplier) and np.isscalar(right_shift):
        return np.vectorize(
            lambda x: requantize_value(
                int(x),
                multiplier=int(multiplier),
                right_shift=int(right_shift),
                out_width=out_width,
            ),
            otypes=[np.int64],
        )(array).astype(dtype)

    mult = np.asarray(multiplier, dtype=np.int64)
    shift = np.asarray(right_shift, dtype=np.int64)
    if mult.ndim != 1 or shift.ndim != 1:
        raise ValueError("per-channel requant vectors must be 1D")
    if mult.shape != shift.shape:
        raise ValueError("per-channel requant multiplier/shift vectors must have the same length")
    for value in mult.tolist():
        _validate_requant_value(int(value), "multiplier", max_value=MAX_REQUANT_MULTIPLIER)
    for value in shift.tolist():
        _validate_requant_value(int(value), "right_shift", max_value=MAX_REQUANT_RIGHT_SHIFT)
    axis = int(axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise ValueError(f"axis out of range for requantize_array: axis={axis}, ndim={array.ndim}")
    if array.shape[axis] != mult.shape[0]:
        raise ValueError(
            f"requant vector length mismatch: axis {axis} has size {array.shape[axis]}, "
            f"vector length is {mult.shape[0]}"
        )
    reshape = [1] * array.ndim
    reshape[axis] = mult.shape[0]
    mult = mult.reshape(reshape)
    shift = shift.reshape(reshape)
    scaled = np.right_shift(array.astype(np.int64) * mult, shift)
    hi = (1 << (out_width - 1)) - 1
    lo = -(1 << (out_width - 1))
    return np.clip(scaled, lo, hi).astype(dtype)


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


def symmetric_scale(
    values: np.ndarray,
    *,
    qmax: int,
    axis: int | None = None,
    percentile: float | None = None,
) -> float | np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    abs_values = np.abs(array)
    if axis is None:
        max_abs = float(
            np.percentile(abs_values, percentile) if percentile is not None else np.max(abs_values)
        )
        if max_abs == 0.0:
            return 1.0
        return max_abs / float(qmax)
    reduce_axes = tuple(dim for dim in range(array.ndim) if dim != int(axis))
    max_abs = (
        np.percentile(abs_values, percentile, axis=reduce_axes)
        if percentile is not None
        else np.max(abs_values, axis=reduce_axes)
    )
    max_abs = np.asarray(max_abs, dtype=np.float32)
    max_abs = np.where(max_abs == 0.0, 1.0, max_abs)
    return max_abs / float(qmax)


def quantize_symmetric(values: np.ndarray, *, bits: int, scale: float | np.ndarray) -> np.ndarray:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    q = np.round(np.asarray(values, dtype=np.float32) / np.asarray(scale, dtype=np.float32))
    return np.clip(q, qmin, qmax).astype(np.int16 if bits > 4 else np.int8)
