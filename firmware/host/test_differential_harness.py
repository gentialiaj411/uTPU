import os

from differential_test_harness import run_differential_harness


def test_differential_harness_report_and_tolerances():
    report = run_differential_harness(output_json_path="build/reports/differential_test_report.json")

    assert report["tolerance"]["atol"] == 1e-5
    assert report["tolerance"]["rtol"] == 1e-5
    assert len(report["shapes"]) >= 3
    assert os.path.exists("build/reports/differential_test_report.json")

    for case in report["shapes"]:
        backends = {entry["backend"]: entry for entry in case["backends"]}
        assert "cuda" in backends
        assert "utpu" in backends

        cuda_entry = backends["cuda"]
        if cuda_entry["status"] == "skipped":
            reason = (cuda_entry.get("reason") or "").lower()
            assert "cuda" in reason or "nvrtc" in reason or "driver" in reason
        else:
            assert cuda_entry["status"] == "pass"
            assert cuda_entry["within_tolerance"] is True
            assert cuda_entry["max_abs_error"] <= report["tolerance"]["atol"]

        utpu_entry = backends["utpu"]
        assert utpu_entry["execution_mode"] == "quantized_reference_emulation"
        assert utpu_entry["status"] == "pass"
        assert utpu_entry["within_tolerance"] is True
        assert utpu_entry["max_abs_error"] <= report["tolerance"]["atol"]
