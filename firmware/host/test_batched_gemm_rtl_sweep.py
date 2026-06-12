import json
from pathlib import Path

import pytest

from run_rtl_batched_gemm_sweep import run_sweep


def test_batched_gemm_rtl_shape_batch_sweep():
    metrics_path = Path("build/reports/rtl_batched_gemm_sweep.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    run_sweep(str(metrics_path))
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not data["aggregate"]["rtl_available"]:
        pytest.skip("RTL simulator unavailable on this host")
    assert data["status"] == "ok"
    assert data["aggregate"]["all_cases_passed"] is True
    assert any(
        case["shape"]["out_features"] == 16
        and case["shape"]["in_features"] == 16
        and case["batch_size"] == 64
        for case in data["cases"]
    )
    assert any(
        case["shape"]["out_features"] == 64
        and case["shape"]["in_features"] == 64
        and case["batch_size"] == 16
        for case in data["cases"]
    )
