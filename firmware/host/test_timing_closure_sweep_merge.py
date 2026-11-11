"""Regression: timing_closure_sweep merge must not drop or erase failure evidence."""
from __future__ import annotations

import sys
from pathlib import Path

HOST = Path(__file__).resolve().parent
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from run_timing_closure_sweep import merge_points, prefer_point, summarize_max_batch_lut_bisect


def test_prefer_point_keeps_lut_failure_over_missing_reports() -> None:
    old = {
        "MAX_BATCH_COUNT": 64,
        "clock_period_ns": 20.0,
        "QUANTIZER_PIPE_DEPTH": 3,
        "status": "failed_lut_overutilization",
        "failure_reason": "lut_overutilization",
        "failure_detail": {"lut_required": 67217, "lut_available": 63400},
    }
    new = {
        "MAX_BATCH_COUNT": 64,
        "clock_period_ns": 20.0,
        "QUANTIZER_PIPE_DEPTH": 3,
        "status": "missing_reports",
        "failure_reason": "missing_reports",
    }
    kept = prefer_point(old, new)
    assert kept["status"] == "failed_lut_overutilization"
    assert kept["failure_detail"]["lut_required"] == 67217


def test_merge_propagates_lut_failure_across_periods() -> None:
    existing = [
        {
            "MAX_BATCH_COUNT": 64,
            "clock_period_ns": 15.0,
            "QUANTIZER_PIPE_DEPTH": 3,
            "status": "failed_lut_overutilization",
            "failure_reason": "lut_overutilization",
            "failure_detail": {"lut_required": 67217, "lut_available": 63400},
        },
        {
            "MAX_BATCH_COUNT": 64,
            "clock_period_ns": 20.0,
            "QUANTIZER_PIPE_DEPTH": 3,
            "status": "missing_reports",
            "failure_reason": "missing_reports",
        },
    ]
    merged = merge_points(existing, [])
    by_period = {p["clock_period_ns"]: p for p in merged}
    assert by_period[20.0]["status"] == "failed_lut_overutilization"
    assert by_period[20.0]["failure_detail"]["lut_required"] == 67217


def test_merge_never_drops_prior_keys() -> None:
    existing = [
        {
            "MAX_BATCH_COUNT": 4,
            "clock_period_ns": 10.0,
            "QUANTIZER_PIPE_DEPTH": 3,
            "status": "closed",
        }
    ]
    new = [
        {
            "MAX_BATCH_COUNT": 48,
            "clock_period_ns": 20.0,
            "QUANTIZER_PIPE_DEPTH": 3,
            "status": "closed",
        }
    ]
    merged = merge_points(existing, new)
    assert len(merged) == 2
    assert {p["MAX_BATCH_COUNT"] for p in merged} == {4, 48}


def test_max_batch_lut_bisect_summary() -> None:
    points = [
        {
            "MAX_BATCH_COUNT": 32,
            "clock_period_ns": 20.0,
            "QUANTIZER_PIPE_DEPTH": 3,
            "status": "closed",
            "utilization": {"lut_used": 37293},
        },
        {
            "MAX_BATCH_COUNT": 48,
            "clock_period_ns": 20.0,
            "QUANTIZER_PIPE_DEPTH": 3,
            "status": "closed",
            "utilization": {"lut_used": 51751},
        },
        {
            "MAX_BATCH_COUNT": 64,
            "clock_period_ns": 20.0,
            "QUANTIZER_PIPE_DEPTH": 3,
            "status": "failed_lut_overutilization",
            "failure_reason": "lut_overutilization",
            "failure_detail": {"lut_required": 67217, "lut_available": 63400},
        },
    ]
    summary = summarize_max_batch_lut_bisect(points, pipe_depth=3)
    assert summary["largest_closing_MAX_BATCH_COUNT"] == 48
    assert abs(summary["occupancy_ceiling_at_largest_closing"]["streaming_occupancy_ceiling"] - 0.75) < 1e-9
