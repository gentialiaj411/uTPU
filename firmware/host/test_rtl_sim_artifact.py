import json
from pathlib import Path

import pytest


def test_rtl_sim_artifact_lock():
    metrics_path = Path("build/reports/rtl_fused_sim_metrics.json")
    if not metrics_path.exists():
        pytest.skip("rtl metrics artifact missing (likely simulator unavailable)")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["case1_passed"] is True
    assert data["case2_passed"] is True
    assert data["total_cycles"] == 44018
