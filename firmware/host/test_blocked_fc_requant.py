import json
from pathlib import Path

import pytest

from isa_encoder import IsaConfig, encodeRequantParams
from requantization import ROUNDING_MODE, RequantParams, requantize_value
from run_rtl_batched_gemm_sim import run_rtl_batched_gemm_sim


REPO_ROOT = Path(__file__).resolve().parents[2]
INT8_CFG = IsaConfig(address_width=13, compute_data_width=8)
INT8_REQUANT = RequantParams(multiplier=11, right_shift=6, enable=True)


def test_encode_requant_params_extended_mode():
    encoded = encodeRequantParams(1234, 9, enable=True, cfg=INT8_CFG)
    assert len(encoded) == 6
    assert encoded[:2] == bytes([0x1D, 0x00])
    assert int.from_bytes(encoded[2:4], byteorder="little") == 1234
    assert int.from_bytes(encoded[4:6], byteorder="little") == 9


def test_requant_math_uses_shared_truncation_and_saturation():
    assert ROUNDING_MODE == "arithmetic_right_shift_truncation"
    assert requantize_value(1000, multiplier=11, right_shift=6, out_width=8) == 127
    assert requantize_value(-1000, multiplier=11, right_shift=6, out_width=8) == -128
    assert requantize_value(320, multiplier=11, right_shift=6, out_width=8) == 55
    assert requantize_value(-320, multiplier=11, right_shift=6, out_width=8) == -55


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
