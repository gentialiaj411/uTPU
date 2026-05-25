import json
import os

import pytest


def test_real_model_end_to_end_artifact_schema():
    pytest.importorskip("torchvision")
    from run_real_model_end_to_end import run_real_model_benchmark

    out_path = "build/reports/real_model_end_to_end_test.json"
    report = run_real_model_benchmark(
        output_json_path=out_path,
        input_size=32,
        warmup=0,
        iters=1,
        seeds=(0, 1, 42),
    )

    assert report["model"] == "resnet18"
    assert report["backend"] == "cuda"
    assert report["tolerance"]["rtol"] == 1e-3
    assert report["tolerance"]["atol"] == 1e-3
    assert len(report["cases"]) == 3
    assert report["all_cases_within_tolerance_vs_eager"] is True

    for case in report["cases"]:
        assert case["eager_pytorch"]["within_tolerance"] is True
        inductor = case["torch_compile_inductor"]
        if inductor["status"] == "pass":
            assert inductor["within_tolerance"] is True
        elif inductor["status"] == "skipped":
            reason = (inductor.get("reason") or "").lower()
            assert any(
                token in reason
                for token in ("inductor", "triton", "compile", "cuda", "winerror", "not supported")
            )

    assert os.path.exists(out_path)
    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["model"] == "resnet18"
