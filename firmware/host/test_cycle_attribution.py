"""Schema gate for bench/results/cycle_attribution.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

ARTIFACT = REPO / "bench" / "results" / "cycle_attribution.json"
ATTR_KEYS = [
    "fetch_decode",
    "store",
    "load",
    "compute",
    "writeback",
    "bstore",
    "ext_addr",
    "requant",
    "result_fetch",
]


def test_cycle_attribution_artifact_schema() -> None:
    assert ARTIFACT.exists(), f"missing {ARTIFACT}; run firmware/host/run_cycle_attribution.py"
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert data.get("schema_version") == 1
    assert data.get("verdict") == "confirmed_fetch_stream_dominates_load_negligible"
    assert data["methodology"]["array_size"] == 16
    assert data["methodology"]["uart_baud_hz_testbench"] == 100_000_000
    assert data["methodology"]["uart_baud_hz_shipping_default"] == 115_200
    assert "on-chip core" in data["methodology"]["measurement_scope"].lower() or "On-chip core" in data["methodology"]["measurement_scope"]
    amdahl = data.get("amdahl_part_a") or {}
    assert amdahl.get("array_streaming_ceiling_B_over_2N_B", {}).get("N") == 16
    assert abs(float(amdahl["array_streaming_ceiling_B_over_2N_B"]["value"]) - 0.6) < 1e-9
    assert float(amdahl["speedup_ceiling"]) == pytest.approx(2.96, rel=0.01)
    cases = data.get("cases") or []
    assert len(cases) == 12
    for case in cases:
        assert case.get("rtl_sim_passed") is True
        assert case.get("total_program_cycles", 0) > 0
        assert case.get("attr_sum_vs_total_delta") == 0
        groups = case["groups"]
        for key in ATTR_KEYS:
            assert key in groups
            assert groups[key]["cycles"] is not None
        # Weight LOAD must stay a small fraction on every row (hypothesis fence).
        load_pct = groups["load"]["pct_of_total_program_cycles"]
        assert load_pct is not None and load_pct < 0.05
    flag = next(c for c in cases if c["shape"]["out_features"] == 64 and c["batch_size"] == 48)
    assert flag["hypothesis_metrics"]["load_over_total"] < 0.05
    assert flag["hypothesis_metrics"]["instruction_stream_pct"] >= 0.50
