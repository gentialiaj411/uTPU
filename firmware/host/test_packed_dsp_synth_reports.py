from __future__ import annotations

import json
from pathlib import Path


ARTIFACT = Path("bench/results/packed_dsp_synth.json")


def load_artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


# Schema v2 (write_packed_mac_budget.py): measured out-of-context Vivado DSP counts, replacing the v1 pending stub.
RUNS = {
    "packed_mac_ooc_cell1": "pe_packed_skewed",
    "packed_mac_ooc_8x8": "pe_array_packed",
    "packed_mac_ooc_16x16": "pe_array_packed",
}


def test_schema_and_pending_status() -> None:
    data = load_artifact()
    assert data["version"] == 2
    stamp = data["generated_at_utc"]
    assert stamp.endswith("Z") or stamp.endswith("+00:00"), stamp
    assert data["git_sha"]
    assert data["vivado_version"]
    assert data["part"] == "xc7a100tcsg324-1"
    assert data["methodology"]
    assert {run["name"]: run["top"] for run in data["runs"]} == RUNS


def test_run_entries_have_p4_2_shape() -> None:
    data = load_artifact()
    for run in data["runs"]:
        assert run["status"] in ("ok", "ok_part_capped"), run["name"]
        assert run["report_prefix"] == run["name"]
        assert isinstance(run["measured_dsp"], int) and run["measured_dsp"] > 0
        assert isinstance(run["macs_logical"], int) and run["macs_logical"] > 0
        assert run["hypothesis_dsp"] > 0
        if "dsp_available" in run:
            assert run["measured_dsp"] <= run["dsp_available"]
        if run["status"] == "ok_part_capped":
            assert run["measured_dsp"] == run["dsp_available"], "part-capped means the part's DSPs are exhausted"
