import json
import os

import pytest


def test_mnist_utpu_demo_artifact_schema_and_floor():
    path = os.path.join("build", "reports", "mnist_utpu_demo.json")
    if not os.path.exists(path):
        pytest.skip(f"Missing artifact: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = {
        "generated_at_utc",
        "model_name",
        "seed",
        "dataset",
        "float_acc",
        "quant_acc",
        "proxy_quant_acc",
        "deployed_raw_integer_acc",
        "reference_semantics",
        "scope_note",
        "accuracy_split",
        "board_config",
        "instruction_bram_words",
        "fits_instruction_bram",
        "bit_exact_vs_reference",
        "isa_rtl_bitmatch",
        "shapes",
        "cases",
        "quantization",
        "rtl_trace_log_path",
    }
    assert required.issubset(data.keys())

    assert isinstance(data["generated_at_utc"], str) and data["generated_at_utc"].endswith("Z")
    assert isinstance(data["model_name"], str) and data["model_name"]
    assert isinstance(data["seed"], int)

    assert float(data["float_acc"]) >= 0.90
    assert float(data["proxy_quant_acc"]) >= 0.90
    assert float(data["quant_acc"]) == pytest.approx(float(data["proxy_quant_acc"]))
    assert float(data["deployed_raw_integer_acc"]) < 0.50
    assert data["bit_exact_vs_reference"] is True
    assert data["fits_instruction_bram"] is True
    assert data["rtl_sim_executed"] in {True, False}
    assert data["reference_semantics"] == "raw_integer_reference_for_lowered_fused_mlp_program"
    assert isinstance(data["scope_note"], str) and data["scope_note"]

    accuracy_split = data["accuracy_split"]
    assert float(accuracy_split["float_accuracy"]) == pytest.approx(float(data["float_acc"]))
    assert float(accuracy_split["deployed_raw_integer_accuracy"]) == pytest.approx(float(data["deployed_raw_integer_acc"]))
    assert float(accuracy_split["proxy_quant_accuracy"]) == pytest.approx(float(data["proxy_quant_acc"]))

    board = data["board_config"]
    assert isinstance(board, dict)
    assert board["name"] in {"pynqz2_baseline", "pynqz2_bram_max", "vu13p_uram"}
    assert int(board["prog_depth"]) in {1024, 8192, 131072}

    ds = data["dataset"]
    assert isinstance(ds, dict)
    assert ds["name"] == "mnist_14x14_local_downsampled_to_8x8"
    assert ds["input_dim"] == 64
    assert ds["output_dim"] == 10

    if not data.get("rtl_sim_executed"):
        pytest.skip("RTL sim unavailable on this host")

    assert data["isa_rtl_bitmatch"] is True
    assert data["rtl_sim_passed"] is True

    shapes = data["shapes"]
    assert shapes["fc1"] == [64, 64]
    assert shapes["fc2"] == [64, 10]
    assert shapes["array_size"] == 16

    cases = data["cases"]
    assert isinstance(cases, list) and len(cases) == 3
    for case in cases:
        assert isinstance(case, dict)
        assert case["bit_exact_vs_reference"] is True
        assert case["isa_rtl_bitmatch"] is True
        assert isinstance(case["expected_fetch_bytes"], list)
        assert len(case["expected_fetch_bytes"]) == 6
        assert len(case["expected_fetch_bytes"]) == len(case["isa_fetch_bytes"])
        assert case["rtl_fetch_bytes"] == case["expected_fetch_bytes"]
        assert case["program_words"] <= data["instruction_bram_words"]
        assert isinstance(case["proxy_pred"], int)
        assert isinstance(case["deployed_pred"], int)

    q = data["quantization"]
    assert q["activation_mode"] == "per-sample symmetric int4"
    assert q["weight_mode"] == "per-row symmetric int4"
    assert q["relu_mode"] == "leaky_relu(alpha=0.25)"
    assert q["biases"] is False
