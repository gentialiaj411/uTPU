from pathlib import Path

import pytest

from run_isa_rtl_residual_bitmatch import run_bitmatch


def test_rtl_residual_sim_artifact_lock():
    metrics_path = Path("build/reports/isa_rtl_residual_bitmatch_report.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_bitmatch(
        "build/reports/isa_rtl_residual_bitmatch_report.json",
        "build/reports/isa_rtl_residual_bitmatch_report.md",
    )
    if not metrics_path.exists():
        pytest.skip("residual bitmatch artifact missing (likely simulator unavailable)")

    if not report.get("rtl_sim_executed", False):
        pytest.skip("RTL simulator unavailable on this host")
    assert report["case_count"] >= 1
    assert all(report["cases"][idx]["isa_expected_bitmatch"] for idx in range(report["case_count"]))
    assert all(report["cases"][idx]["isa_rtl_bitmatch"] for idx in range(report["case_count"]))
    assert report["all_isa_expected_bitmatch"] is True
    assert report["all_isa_rtl_bitmatch"] is True
