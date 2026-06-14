from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from isa_encoder import IsaConfig
from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from requantization import RequantParams, requantize_array


DEFAULT_BMM_CFG = IsaConfig(address_width=13, compute_data_width=8)
DEFAULT_WEIGHT_ADDR = 0
DEFAULT_INPUT_ADDR = 256
DEFAULT_RESULT_ADDR = 1024


def _requant_identity(cfg: IsaConfig) -> RequantParams:
    return RequantParams(multiplier=1, right_shift=0, enable=True)


def _decode_fetch_bytes(
    fetch_bytes: Sequence[int],
    *,
    out_features: int,
    batch_size: int,
    array_size: int,
    cfg: IsaConfig,
) -> np.ndarray:
    words: List[int] = []
    for i in range(0, len(fetch_bytes), 2):
        lo = int(fetch_bytes[i]) & 0xFF
        hi = int(fetch_bytes[i + 1]) & 0xFF if i + 1 < len(fetch_bytes) else 0
        words.append(lo | (hi << 8))
    values: List[int] = []
    mask = (1 << cfg.compute_data_width) - 1
    sign_bit = 1 << (cfg.compute_data_width - 1)
    for word in words:
        for shift in range(0, 16, cfg.compute_data_width):
            raw = (word >> shift) & mask
            values.append(raw - (1 << cfg.compute_data_width) if raw & sign_bit else raw)
    out_blocks = (int(out_features) + int(array_size) - 1) // int(array_size)
    out_padded = out_blocks * int(array_size)
    result = np.zeros((int(batch_size), out_padded), dtype=np.int8 if cfg.compute_data_width <= 8 else np.int16)
    cursor = 0
    for ob in range(out_blocks):
        tile_vals = values[cursor : cursor + (int(array_size) * int(batch_size))]
        cursor += int(array_size) * int(batch_size)
        tile = np.asarray(tile_vals, dtype=result.dtype).reshape((int(array_size), int(batch_size)), order="F")
        result[:, ob * int(array_size) : (ob + 1) * int(array_size)] = tile.T
    return result[:, : int(out_features)]


def batched_matmul_int_oracle(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    requant_params: RequantParams | None = None,
    cfg: IsaConfig = DEFAULT_BMM_CFG,
    apply_relu: bool = False,
) -> np.ndarray:
    a = np.asarray(lhs, dtype=np.int32)
    b = np.asarray(rhs, dtype=np.int32)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(f"batched_matmul_int_oracle expects rank >= 2, got {a.shape} and {b.shape}")
    if a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2]:
        raise ValueError(f"incompatible batched matmul shapes: {a.shape} and {b.shape}")
    params = requant_params if requant_params is not None else _requant_identity(cfg)
    accum = np.matmul(a, b).astype(np.int32, copy=False)
    out_dtype = np.int8 if cfg.compute_data_width <= 8 else np.int16
    quant = requantize_array(
        accum,
        multiplier=params.per_channel_multipliers if params.is_per_channel else params.multiplier,
        right_shift=params.per_channel_right_shifts if params.is_per_channel else params.right_shift,
        out_width=cfg.compute_data_width,
        dtype=out_dtype,
        axis=-1,
    )
    if apply_relu:
        quant = np.where(quant < 0, quant.astype(np.int32) >> 2, quant.astype(np.int32)).astype(out_dtype)
    return quant


@dataclass(frozen=True)
class LoweredBatchedMatmulProgram:
    batch_index: Tuple[int, ...]
    program: bytes
    program_instruction_words: int
    expected_fetch_bytes: List[int]
    output_shape: Tuple[int, int]
    decoded_output: np.ndarray


@dataclass(frozen=True)
class LoweredBatchedMatmulUTPU:
    programs: List[LoweredBatchedMatmulProgram]
    output_shape: Tuple[int, ...]
    cfg: Dict[str, Any]
    requant_params: Dict[str, Any]
    all_programs_bit_exact_vs_oracle: bool
    output: np.ndarray


def lower_batched_matmul_utpu(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    array_size: int = 16,
    cfg: IsaConfig = DEFAULT_BMM_CFG,
    requant_params: RequantParams | None = None,
    apply_relu: bool = False,
    weight_addr: int = DEFAULT_WEIGHT_ADDR,
    input_addr: int = DEFAULT_INPUT_ADDR,
    result_addr: int = DEFAULT_RESULT_ADDR,
) -> LoweredBatchedMatmulUTPU:
    a = np.asarray(lhs, dtype=np.int8)
    b = np.asarray(rhs, dtype=np.int8)
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(f"uTPU batched matmul lowering expects rank >= 2, got {a.shape} and {b.shape}")
    if a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2]:
        raise ValueError(f"incompatible batched matmul shapes: {a.shape} and {b.shape}")
    params = requant_params if requant_params is not None else _requant_identity(cfg)
    prefix = tuple(int(v) for v in a.shape[:-2])
    batch_outer = int(np.prod(prefix)) if prefix else 1
    m = int(a.shape[-2])
    k = int(a.shape[-1])
    n = int(b.shape[-1])
    a_flat = a.reshape(batch_outer, m, k)
    b_flat = b.reshape(batch_outer, k, n)
    oracle = batched_matmul_int_oracle(a, b, requant_params=params, cfg=cfg, apply_relu=apply_relu)
    oracle_flat = oracle.reshape(batch_outer, m, n)
    programs: List[LoweredBatchedMatmulProgram] = []
    outputs: List[np.ndarray] = []
    all_exact = True
    for batch_idx in range(batch_outer):
        lowered = lower_blocked_fc_program_utpu(
            weights_int4=b_flat[batch_idx].T,
            activations_int4=a_flat[batch_idx],
            out_features=n,
            in_features=k,
            array_size=array_size,
            apply_relu=apply_relu,
            apply_quant=True,
            weight_addr=weight_addr,
            input_addr=input_addr,
            result_addr=result_addr,
            cfg=cfg,
            requant_params=params,
        )
        sim = simulate_program_bytes(
            lowered["program"],
            array_size=array_size,
            buffer_size=(1 << cfg.address_width),
            cfg=cfg,
            accumulator_data_width=32,
        )
        decoded = _decode_fetch_bytes(
            sim.fetch_bytes,
            out_features=n,
            batch_size=m,
            array_size=array_size,
            cfg=cfg,
        )
        outputs.append(decoded.astype(np.int8, copy=False))
        exact = np.array_equal(decoded, oracle_flat[batch_idx])
        all_exact = all_exact and bool(exact)
        unraveled = np.unravel_index(batch_idx, prefix) if prefix else ()
        programs.append(
            LoweredBatchedMatmulProgram(
                batch_index=tuple(int(v) for v in unraveled),
                program=lowered["program"],
                program_instruction_words=int(lowered["program_instruction_words"]),
                expected_fetch_bytes=list(sim.fetch_bytes),
                output_shape=(m, n),
                decoded_output=decoded.astype(np.int8, copy=False),
            )
        )
    stacked = np.stack(outputs, axis=0).reshape(prefix + (m, n))
    return LoweredBatchedMatmulUTPU(
        programs=programs,
        output_shape=tuple(int(v) for v in stacked.shape),
        cfg={
            "address_width": int(cfg.address_width),
            "compute_data_width": int(cfg.compute_data_width),
            "buffer_size": int(1 << cfg.address_width),
            "extended_address": bool(cfg.extended_address),
        },
        requant_params=params.as_dict(),
        all_programs_bit_exact_vs_oracle=bool(all_exact),
        output=stacked.astype(np.int8, copy=False),
    )


def simulate_lowered_batched_matmul_utpu(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    array_size: int = 16,
    cfg: IsaConfig = DEFAULT_BMM_CFG,
    requant_params: RequantParams | None = None,
    apply_relu: bool = False,
) -> Dict[str, Any]:
    lowered = lower_batched_matmul_utpu(
        lhs,
        rhs,
        array_size=array_size,
        cfg=cfg,
        requant_params=requant_params,
        apply_relu=apply_relu,
    )
    return {
        "output": lowered.output,
        "all_programs_bit_exact_vs_oracle": lowered.all_programs_bit_exact_vs_oracle,
        "program_count": len(lowered.programs),
        "cfg": dict(lowered.cfg),
        "requant_params": dict(lowered.requant_params),
    }
