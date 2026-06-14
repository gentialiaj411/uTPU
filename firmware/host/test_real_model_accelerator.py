import json
from pathlib import Path

import pytest

from run_rtl_batched_gemm_sim import _resolve_iverilog_tools


ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "bench" / "results" / "real_model_accelerator.json"


def test_real_model_accelerator_artifact_schema_and_floors():
    if not ARTIFACT_PATH.exists():
        pytest.skip("real_model_accelerator artifact missing; regenerate locally")

    data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["model_name"] == "mnist_14x14_196x256x10_mlp_ptq"

    ds = data["dataset"]
    assert ds["name"] == "mnist_14x14_local"
    assert ds["input_dim"] == 196
    assert ds["num_classes"] == 10

    acc = data["accuracy_sweep"]
    assert float(acc["float_accuracy"]) >= 0.95
    assert float(acc["int8_accuracy"]) >= 0.90
    assert 0.0 <= float(acc["int4_accuracy"]) <= 1.0
    assert "accuracy_comparison" in data
    assert set(data["accuracy_comparison"]["int8"]) == {"per_layer_accuracy", "per_channel_accuracy"}
    assert set(data["accuracy_comparison"]["int4"]) == {"per_layer_accuracy", "per_channel_accuracy"}

    contract = data["quantization_contract"]
    assert contract["rounding_mode"] == "arithmetic_right_shift_truncation"
    assert contract["symmetric_zero_point"] == 0
    assert "quantizer.sv" in contract["requant_multiply_location"]
    assert "simulator" in contract["requant_multiply_location"]

    board_fit = data["board_fit"]
    assert board_fit["selected_board"] is not None
    assert board_fit["per_board"]["vu13p_uram"]["fits_instruction_bram"] is True

    accel = data["accelerator_validation"]
    assert accel["deployed_bitwidth"] == 8
    assert accel["reference_semantics"] == "independent_scaled_integer_reference_for_lowered_batched_blocked_fc_program"
    assert accel["batch_size"] == 4
    assert accel["evaluated_samples"] == 4
    assert accel["bit_exact_vs_reference"] is True
    assert accel["isa_bit_exact_vs_reference"] is True

    if not accel["rtl_sim_executed"]:
        pytest.skip("RTL simulator unavailable on this host")

    iv_bin, _ = _resolve_iverilog_tools()
    if iv_bin is None:
        pytest.skip("iverilog unavailable on this host")

    assert accel["isa_rtl_bitmatch"] is True
    assert accel["rtl_sim_passed"] is True

    util = data["batched_utilization"]["per_layer"]
    if util["fc1"]["compute_span_duty_cycle"] is not None:
        assert 0.0 < float(util["fc1"]["compute_span_duty_cycle"]) <= 1.0
    assert util["fc2"]["compute_span_duty_cycle"] is not None
    assert 0.0 < float(util["fc2"]["compute_span_duty_cycle"]) <= 1.0
