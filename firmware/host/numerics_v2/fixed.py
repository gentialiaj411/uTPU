"""Shared integer arithmetic contract. This module intentionally has no floats."""

from __future__ import annotations

import numpy as np

I32_MAX = (1 << 31) - 1
RECIP_SHIFT = 24
INV127_Q24 = 132104
INV7_Q24 = 2396745
INV127_Q32 = 33818641
INV7_Q32 = 613566757


def rshift_rne(value, shift):
    """Signed right shift, round-to-nearest with ties to even."""
    x = np.asarray(value, dtype=np.int64)
    s = np.asarray(shift, dtype=np.int64)
    if np.any(s < 0):
        raise ValueError("right shift must be nonnegative")
    safe = np.maximum(s, 1)
    mag = np.abs(x)
    q = np.right_shift(mag, safe)
    rem = mag - np.left_shift(q, safe)
    half = np.left_shift(np.ones_like(safe), safe - 1)
    inc = (rem > half) | ((rem == half) & ((q & 1) != 0))
    rounded = q + inc
    signed = np.where(x < 0, -rounded, rounded)
    return np.where(s == 0, x, signed)


def div_rne(numerator, denominator):
    """Signed integer division, round-to-nearest with ties to even."""
    n = np.asarray(numerator, dtype=np.int64)
    d = np.asarray(denominator, dtype=np.int64)
    if np.any(d <= 0):
        raise ValueError("denominator must be positive")
    mag = np.abs(n)
    q, rem = np.divmod(mag, d)
    inc = (rem * 2 > d) | ((rem * 2 == d) & ((q & 1) != 0))
    return np.where(n < 0, -(q + inc), q + inc)


def reciprocal_rne(target, denominator):
    """One Q24 reciprocal divide: RNE((target << 24) / denominator)."""
    return div_rne(np.asarray(target, dtype=np.int64) << RECIP_SHIFT, denominator)


def mul_recip(value, reciprocal):
    """Per-element reciprocal application: RNE(value * reciprocal / 2^24)."""
    return rshift_rne(np.asarray(value, dtype=np.int64) * np.asarray(reciprocal, dtype=np.int64), RECIP_SHIFT)


def div127(value):
    """Constant division by 127 using a Q24 reciprocal multiply and shift."""
    return div_const_rne(value, 127)


def div_const_rne(value, denominator):
    """Exact RNE division by a supported constant using reciprocal multiply.

    The reciprocal estimate is followed by a bounded residual correction (one
    multiply/compare/subtract), so this contains no variable divide.  The
    correction makes the Q32 approximation exact for every reachable numerator
    (the runtime asserts |numerator| <= 2^31 before this boundary).
    """
    d = int(denominator)
    if d <= 0:
        raise ValueError("denominator must be positive")
    shift = 32
    reciprocal = {7: INV7_Q32, 127: INV127_Q32}.get(d, int(round((1 << shift) / d)))
    n = np.asarray(value, dtype=np.int64)
    if np.any(np.abs(n) > (1 << 31)):
        # Wide-product fallback models the RTL's 64x32->96-bit multiplier;
        # it still performs only constant multiply, shift, and correction.
        flat = n.reshape(-1)
        out = []
        for item in flat:
            sign = -1 if int(item) < 0 else 1
            mag_item = abs(int(item)); qi = (mag_item * reciprocal + (1 << (shift - 1))) >> shift
            err = abs(mag_item - qi * d) * 2
            if err > d: qi += 1 if mag_item > qi * d else -1
            elif err == d and (qi & 1) and mag_item < qi * d: qi -= 1
            out.append(sign * qi)
        return np.asarray(out, dtype=np.int64).reshape(n.shape)
    mag = np.abs(n)
    q = rshift_rne(mag * reciprocal, shift)
    twice_err = np.abs((mag - q * d) * 2)
    down = twice_err > d
    tie_odd = (twice_err == d) & ((q & 1) != 0)
    q = q + ((twice_err > d) & (mag > q * d))
    q = q - ((twice_err > d) & (mag < q * d))
    q = q - (tie_odd & (mag < q * d))
    return np.where(n < 0, -q, q)


def requant(value, multiplier, shift, lo=-128, hi=127, dtype=np.int8):
    product = np.asarray(value, dtype=np.int64) * np.asarray(multiplier, dtype=np.int64)
    return np.clip(rshift_rne(product, shift), lo, hi).astype(dtype)


def checked_matmul(a, w, k_bound, weight_bound):
    """INT32 dot product after a contractual worst-case overflow proof."""
    bound = int(k_bound) * 127 * int(weight_bound)
    assert bound <= I32_MAX, (k_bound, bound)
    lhs = np.asarray(a, dtype=np.int32)
    rhs = np.asarray(w, dtype=np.int32)
    out = lhs @ rhs.T
    assert out.dtype == np.int32
    return out
