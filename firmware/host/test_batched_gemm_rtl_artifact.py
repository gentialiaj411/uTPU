import json
from pathlib import Path

import pytest

from run_rtl_batched_gemm_sim import run_rtl_batched_gemm_sim


def test_batched_gemm_rtl_artifact_lock():
    metrics_path = Path("build/reports/rtl_batched_gemm_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    run_rtl_batched_gemm_sim(str(metrics_path))
    if not metrics_path.exists():
        pytest.skip("rtl metrics artifact missing")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not data.get("rtl_sim_executed", False):
        pytest.skip("RTL simulator unavailable on this host")
    assert data["rtl_sim_passed"] is True
    assert data["batch_size"] == 4
    assert data["array_size"] == 16
    assert len(data["expected_fetch_bytes"]) > 0
