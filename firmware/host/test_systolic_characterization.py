from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from run_systolic_characterization import _characterize_case, build_artifact  # noqa: E402
from run_rtl_batched_gemm_sim import _resolve_iverilog_tools  # noqa: E402


def _write_artifact(path: Path) -> dict:
    artifact = build_artifact()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def test_systolic_characterization_schema_and_reality_floors():
    artifact_path = Path("build/reports/systolic_characterization_test.json")
    data = _write_artifact(artifact_path)

    assert data["schema_version"] == 1
    assert data["status"] in {"ok", "iverilog_unavailable"}
    assert "methodology" in data
    assert "headline_metrics" in data["methodology"]
    assert len(data["cases"]) == 9
    assert len(data["shape_summaries"]) == 3

    if data["status"] == "iverilog_unavailable":
        assert data["aggregate"]["rtl_available"] is False
        assert "instructions" in data
        for case in data["cases"]:
            assert case["measured"]["rtl_cycle_counter"] is None
            assert case["measured"]["rtl_busy_counter"] is None
        return

    assert data["aggregate"]["rtl_available"] is True
    assert data["aggregate"]["all_cases_passed"] is True
    assert data["aggregate"]["all_large_b_pe_occupancy_exceeds_b1"] is True

    for case in data["cases"]:
        measured = case["measured"]
        assert case["rtl_sim_passed"] is True
        assert measured["rtl_cycle_counter"] > 0
        assert measured["rtl_busy_counter"] > 0
        assert 0.0 < measured["busy_fraction"] <= 1.0
        assert 0.0 < measured["pe_occupancy"] <= 1.0
        assert measured["rtl_busy_counter"] <= measured["rtl_cycle_counter"]

    summaries = {
        (row["shape"]["out_features"], row["shape"]["in_features"]): row
        for row in data["shape_summaries"]
    }
    # Single-tile streaming curve is the flagship. Busy growth must be
    # sublinear in B, and occupancy must approach the streaming ceiling.
    single_tile = summaries[(16, 16)]
    curve = {point["batch_size"]: point for point in single_tile["batch_curve"]}
    marginals = {
        (point["from_batch_size"], point["to_batch_size"]): point
        for point in single_tile["marginal_busy_curve"]
    }
    assert [point["batch_size"] for point in single_tile["batch_curve"]] == [1, 4, 16, 32, 64]
    assert curve[64]["pe_occupancy"] > curve[1]["pe_occupancy"]
    assert curve[64]["streaming_ceiling"] is not None
    assert abs(curve[64]["pe_occupancy"] - curve[64]["streaming_ceiling"]) <= 0.06
    assert marginals[(16, 32)]["marginal_busy_cycles_per_added_batch"] <= 1.5
    assert marginals[(32, 64)]["marginal_busy_cycles_per_added_batch"] <= 1.5

    # Single-tile shape scales much more aggressively than the control-bound
    # multi-tile families.
    assert summaries[(16, 16)]["busy_cycles_growth_vs_b1"] > 2.0
    assert summaries[(32, 32)]["busy_cycles_growth_vs_b1"] > 1.0
    assert summaries[(16, 16)]["pe_occupancy_growth_vs_b1"] > summaries[(32, 32)]["pe_occupancy_growth_vs_b1"]


def test_systolic_characterization_deterministic_representative_case():
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        pytest.skip("RTL simulator unavailable on this host")

    case1 = _characterize_case(32, 32, 16, run_rtl=True)
    case2 = _characterize_case(32, 32, 16, run_rtl=True)
    assert case1["rtl_sim_passed"] is True
    assert case2["rtl_sim_passed"] is True
    assert case1["measured"]["rtl_cycle_counter"] == case2["measured"]["rtl_cycle_counter"]
    assert case1["measured"]["rtl_busy_counter"] == case2["measured"]["rtl_busy_counter"]
