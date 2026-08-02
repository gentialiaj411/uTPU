import json
from pathlib import Path

import pytest

import numpy as np

from isa_encoder import IsaConfig, encodeRequantParams, encodeRequantParamsVector
from requantization import ROUNDING_MODE, RequantParams, requantize_array, requantize_value
from run_rtl_batched_gemm_sim import run_rtl_batched_gemm_sim


REPO_ROOT = Path(__file__).resolve().parents[2]
INT8_CFG = IsaConfig(address_width=13, compute_data_width=8)
INT8_REQUANT = RequantParams(multiplier=11, right_shift=6, enable=True)
INT8_PER_CHANNEL_REQUANT = RequantParams(
    multiplier=1,
    right_shift=0,
    enable=True,
    per_channel_multipliers=(11, 7, 3, 5, 9, 13, 15, 17, 6, 10, 12, 14, 4, 8, 16, 18),
    per_channel_right_shifts=(6, 5, 3, 4, 5, 6, 7, 7, 4, 5, 6, 6, 3, 4, 7, 7),
)


def test_encode_requant_params_extended_mode():
    encoded = encodeRequantParams(1234, 9, enable=True, cfg=INT8_CFG)
    assert len(encoded) == 6
    assert encoded[:2] == bytes([0x1D, 0x00])
    assert int.from_bytes(encoded[2:4], byteorder="little") == 1234
    assert int.from_bytes(encoded[4:6], byteorder="little") == 9


def test_encode_requant_params_vector_extended_mode():
    encoded = encodeRequantParamsVector([1234, 77, 5], [9, 3, 1], enable=True, cfg=INT8_CFG)
    assert len(encoded) == 14
    assert int.from_bytes(encoded[:2], byteorder="little") == 0x023D
    assert int.from_bytes(encoded[2:4], byteorder="little") == 1234
    assert int.from_bytes(encoded[4:6], byteorder="little") == 9
    assert int.from_bytes(encoded[6:8], byteorder="little") == 77
    assert int.from_bytes(encoded[8:10], byteorder="little") == 3


def test_requant_math_uses_shared_truncation_and_saturation():
    assert ROUNDING_MODE == "arithmetic_right_shift_truncation"
    assert requantize_value(1000, multiplier=11, right_shift=6, out_width=8) == 127
    assert requantize_value(-1000, multiplier=11, right_shift=6, out_width=8) == -128
    assert requantize_value(320, multiplier=11, right_shift=6, out_width=8) == 55
    assert requantize_value(-320, multiplier=11, right_shift=6, out_width=8) == -55


def test_requant_pre_mul_saturates_at_dsp48_operand_width():
    """|acc| >= 2^24 saturates before multiply; in-range values unchanged."""
    from requantization import (
        REQUANT_MUL_OPERAND_MAX,
        REQUANT_MUL_OPERAND_MIN,
        clip_signed,
        saturate_requant_mul_operand,
    )

    assert saturate_requant_mul_operand(REQUANT_MUL_OPERAND_MAX) == REQUANT_MUL_OPERAND_MAX
    assert saturate_requant_mul_operand(REQUANT_MUL_OPERAND_MIN) == REQUANT_MUL_OPERAND_MIN
    assert saturate_requant_mul_operand(REQUANT_MUL_OPERAND_MAX + 100) == REQUANT_MUL_OPERAND_MAX
    assert saturate_requant_mul_operand(REQUANT_MUL_OPERAND_MIN - 100) == REQUANT_MUL_OPERAND_MIN
    # In-range: identical to legacy full-width product then shift/saturate.
    for acc in (0, 1, -1, 320, -320, 1000, -1000, (1 << 20), -(1 << 20)):
        expected = clip_signed((acc * 11) >> 6, 8)
        assert requantize_value(acc, multiplier=11, right_shift=6, out_width=8) == expected
    # Beyond DSP A-port: saturates to ±2^24-1 / -2^24 before multiply.
    assert requantize_value(
        REQUANT_MUL_OPERAND_MAX + 50, multiplier=1, right_shift=17, out_width=8
    ) == requantize_value(REQUANT_MUL_OPERAND_MAX, multiplier=1, right_shift=17, out_width=8)
    assert requantize_value(
        REQUANT_MUL_OPERAND_MIN - 50, multiplier=1, right_shift=17, out_width=8
    ) == requantize_value(REQUANT_MUL_OPERAND_MIN, multiplier=1, right_shift=17, out_width=8)


def test_per_channel_requant_math_matches_manual_reference():
    values = np.asarray([[320, -320, 64], [1000, -1000, 7]], dtype=np.int32)
    actual = requantize_array(
        values,
        multiplier=[11, 9, 3],
        right_shift=[6, 5, 1],
        out_width=8,
        dtype=np.int8,
        axis=1,
    )
    expected = np.asarray(
        [
            [requantize_value(320, multiplier=11, right_shift=6, out_width=8),
             requantize_value(-320, multiplier=9, right_shift=5, out_width=8),
             requantize_value(64, multiplier=3, right_shift=1, out_width=8)],
            [requantize_value(1000, multiplier=11, right_shift=6, out_width=8),
             requantize_value(-1000, multiplier=9, right_shift=5, out_width=8),
             requantize_value(7, multiplier=3, right_shift=1, out_width=8)],
        ],
        dtype=np.int8,
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("shape_batch", [(16, 16, 4), (16, 16, 8), (32, 16, 4), (32, 16, 8)])
def test_requantized_int8_rtl_bitmatch_shape_batch_sweep(shape_batch):
    out_features, in_features, batch_size = shape_batch
    output_json = REPO_ROOT / "build" / "reports" / f"requant_rtl_o{out_features}_i{in_features}_b{batch_size}.json"
    metrics = run_rtl_batched_gemm_sim(
        str(output_json),
        out_features=out_features,
        in_features=in_features,
        batch_size=batch_size,
        stem=f"requant_int8_o{out_features}_i{in_features}_b{batch_size}",
        cfg=INT8_CFG,
        accumulator_data_width=32,
        requant_params=INT8_REQUANT,
    )
    if not metrics["rtl_sim_executed"]:
        pytest.skip("iverilog unavailable on this host")
    assert metrics["rtl_sim_passed"] is True
    assert metrics["requant_params"]["rounding_mode"] == ROUNDING_MODE
    assert metrics["cfg"]["compute_data_width"] == 8
    assert metrics["perf_cycle_counter"] is not None
    assert json.loads(output_json.read_text(encoding="utf-8"))["rtl_sim_passed"] is True


@pytest.mark.parametrize("shape_batch", [(16, 16, 4), (32, 16, 4)])
def test_per_channel_requantized_int8_rtl_bitmatch_shape_batch_sweep(shape_batch):
    out_features, in_features, batch_size = shape_batch
    output_json = REPO_ROOT / "build" / "reports" / f"requant_pc_rtl_o{out_features}_i{in_features}_b{batch_size}.json"
    per_channel = INT8_PER_CHANNEL_REQUANT
    if out_features != 16:
        multipliers = per_channel.per_channel_multipliers * (out_features // 16)
        shifts = per_channel.per_channel_right_shifts * (out_features // 16)
        per_channel = RequantParams(
            multiplier=1,
            right_shift=0,
            enable=True,
            per_channel_multipliers=tuple(int(v) for v in multipliers),
            per_channel_right_shifts=tuple(int(v) for v in shifts),
        )
    metrics = run_rtl_batched_gemm_sim(
        str(output_json),
        out_features=out_features,
        in_features=in_features,
        batch_size=batch_size,
        stem=f"requant_pc_int8_o{out_features}_i{in_features}_b{batch_size}",
        cfg=INT8_CFG,
        accumulator_data_width=32,
        requant_params=per_channel,
    )
    if not metrics["rtl_sim_executed"]:
        pytest.skip("iverilog unavailable on this host")
    assert metrics["rtl_sim_passed"] is True
    assert metrics["requant_params"]["mode"] == "per_channel_symmetric"
    assert metrics["requant_params"]["vector_length"] == out_features
