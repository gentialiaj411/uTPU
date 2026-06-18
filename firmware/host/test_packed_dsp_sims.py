from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from run_pe_packed_pair_sim import run_sim as run_pair_sim  # noqa: E402
from run_pe_array_packed_sim import run_sim as run_array_sim  # noqa: E402
from run_pe_array_packed_hardened_sim import run_sim as run_hardened_sim  # noqa: E402
from run_packed_array_cycle_compare import run_sim as run_cycle_sim  # noqa: E402
from run_top_packed_smoke import run_sim as run_top_smoke_sim  # noqa: E402
from write_packed_dsp_synth_json import build_artifact  # noqa: E402
from run_rtl_batched_gemm_sim import _resolve_iverilog_tools  # noqa: E402


PACKED_ARTIFACTS = [
    ("pe_packed_pair_sim.json", run_pair_sim),
    ("pe_array_packed_sim.json", run_array_sim),
    ("pe_array_packed_hardened.json", run_hardened_sim),
    ("packed_array_cycle_compare.json", run_cycle_sim),
    ("top_packed_smoke.json", run_top_smoke_sim),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("filename,runner", PACKED_ARTIFACTS)
def test_packed_sim_artifact_schema(filename: str, runner) -> None:
    path = Path("bench/results") / filename
    if not path.is_file():
        ok, _, artifact = runner()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    data = _load(path)
    assert data["version"] == 1
    assert data.get("status") in {"ran", "skipped", "pending_vivado_run"}
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        assert data.get("status") == "skipped" or data.get("result") is None
        return
    if data.get("status") == "skipped":
        pytest.skip("iverilog unavailable")
    assert data.get("result") == "PASS"


def test_packed_dsp_synth_pending_schema() -> None:
    path = Path("bench/results/packed_dsp_synth.json")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8")
    data = _load(path)
    assert data["version"] == 1
    assert len(data["runs"]) == 4
