import os

from differential_test_harness import run_differential_harness


def test_differential_harness_report_and_tolerances():
    report = run_differential_harness(output_json_path="build/reports/differential_test_report.json")

    assert report["tolerance"]["atol"] == 1e-5
    assert report["tolerance"]["rtol"] == 1e-5
    assert len(report["shapes"]) >= 3
    assert "jitter" in report["fixture"]["weights"]
    assert "torch.compile_inductor" in report["oracles"]
    assert os.path.exists("build/reports/differential_test_report.json")

    nonzero_abs_errors = []
    for case in report["shapes"]:
        assert {"shape", "backends"}.issubset(case.keys())
        assert {"in_features", "hidden_features", "out_features"}.issubset(case["shape"].keys())
        assert isinstance(case["backends"], list)
        backends = {entry["backend"]: entry for entry in case["backends"]}
        assert "cuda" in backends
        assert "utpu" in backends
        assert "torch_compile" in backends

        torch_compile_entry = backends["torch_compile"]
        assert {"backend", "status", "within_tolerance", "max_abs_error", "max_rel_error"}.issubset(
            torch_compile_entry.keys()
        )
        if torch_compile_entry["status"] == "skipped":
            reason = (torch_compile_entry.get("reason") or "").lower()
            assert (
                "compile" in reason
                or "inductor" in reason
                or "triton" in reason
                or "backend" in reason
                or "winerror" in reason
                or "not supported" in reason
            )
        else:
            assert torch_compile_entry["status"] == "pass"
            assert torch_compile_entry["within_tolerance"] is True
            assert torch_compile_entry["max_abs_error"] <= report["tolerance"]["atol"]
            nonzero_abs_errors.append(torch_compile_entry["max_abs_error"])

        cuda_entry = backends["cuda"]
        assert {"backend", "status", "within_tolerance", "max_abs_error", "max_rel_error"}.issubset(cuda_entry.keys())
        if cuda_entry["status"] == "skipped":
            reason = (cuda_entry.get("reason") or "").lower()
            assert "cuda" in reason or "nvrtc" in reason or "driver" in reason
        else:
            assert cuda_entry["status"] == "pass"
            assert cuda_entry["within_tolerance"] is True
            assert cuda_entry["max_abs_error"] <= report["tolerance"]["atol"]
            nonzero_abs_errors.append(cuda_entry["max_abs_error"])

        utpu_entry = backends["utpu"]
        assert {"backend", "status", "within_tolerance", "max_abs_error", "max_rel_error", "execution_mode"}.issubset(
            utpu_entry.keys()
        )
        assert utpu_entry["execution_mode"] == "quantized_reference_emulation"
        assert utpu_entry["status"] == "pass"
        assert utpu_entry["within_tolerance"] is True
        assert utpu_entry["max_abs_error"] <= report["tolerance"]["atol"]
        nonzero_abs_errors.append(utpu_entry["max_abs_error"])

    assert any(error and error > 0.0 for error in nonzero_abs_errors)
