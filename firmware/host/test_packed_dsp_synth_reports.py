from __future__ import annotations

import json
from pathlib import Path


ARTIFACT = Path("bench/results/packed_dsp_synth.json")


def load_artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_schema_and_pending_status() -> None:
    data = load_artifact()
    assert data["version"] == 1
    assert data["generated_at_utc"].endswith("Z")
    assert "git_sha" in data
    assert len(data["runs"]) == 4
    names = {run["name"] for run in data["runs"]}
    assert names == {
        "packed_baseline_8x8_int8",
        "packed_baseline_16x16_int8",
        "packed_array_8x8_int8",
        "packed_array_16x16_int8",
    }


def test_run_entries_have_p4_2_shape() -> None:
    data = load_artifact()
    for run in data["runs"]:
        assert run["part"] == "xc7a100tcsg324-1"
        assert "params" in run
        assert "report_files" in run
        assert "vivado_version" in run
        if run["timing"] is not None:
            assert run["timing"] is not None
            assert run["utilization"] is not None
            assert run["timing"]["wns_ns"] >= 0.0
            assert run["utilization"]["dsp_used"] <= run["utilization"]["dsp_available"]
        else:
            assert run["utilization"] is None
            assert isinstance(run["route_status"], str)
            assert "synth_failed:" in run["route_status"] or run["route_status"] in {"missing", "unknown", "error", "unplaced", "unrouted"}
