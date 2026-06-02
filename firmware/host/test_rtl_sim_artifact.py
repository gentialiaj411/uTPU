import json
from pathlib import Path

import pytest

from generate_fused_rtl_test_vectors import generate_vectors
from run_rtl_fused_sim import run_rtl_fused_sim


def test_rtl_sim_artifact_lock():
    metrics_path = Path("build/reports/rtl_fused_sim_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    generate_vectors()
    run_rtl_fused_sim("build/reports/rtl_fused_sim_metrics.json", "build/reports/rtl_fused_sim_report.md")
    if not metrics_path.exists():
        pytest.skip("rtl metrics artifact missing (likely simulator unavailable)")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not data.get("rtl_sim_executed", False):
        pytest.skip("RTL simulator unavailable on this host")
    assert data["case_count"] >= 3
    assert all(data["case_passed"][f"case{i}"] for i in range(1, data["case_count"] + 1))
    assert isinstance(data["total_cycles"], int)
    assert data["total_cycles"] > 0
