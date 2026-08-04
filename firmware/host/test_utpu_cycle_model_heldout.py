"""Schema lock for utpu_cycle_model_heldout.json."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
ART = REPO / "bench" / "results" / "utpu_cycle_model_heldout.json"


@pytest.fixture(scope="module")
def report() -> dict:
    if not ART.exists():
        subprocess.check_call(
            [sys.executable, str(HOST / "run_utpu_cycle_model_heldout.py")],
            cwd=REPO,
        )
    return json.loads(ART.read_text(encoding="utf-8"))


def test_mirrors_cuda_field_names(report: dict) -> None:
    assert "latency_prediction" in report
    assert "selection_quality" in report
    assert "split" in report
    tm = report["latency_prediction"]["test_metrics"]
    assert "log_r2" in tm and "mape_pct" in tm


def test_finding_records_determinism_claim(report: dict) -> None:
    f = report["finding"]
    assert "trivially_accurate_due_to_determinism" in f
    # Hardware is deterministic; the fitted model need not be trivially accurate.
    assert isinstance(f["trivially_accurate_due_to_determinism"], bool)


def test_selection_summary_present(report: dict) -> None:
    s = report["selection_quality"]["summary"]
    assert "mean_regret_pct" in s
    assert "max_regret_pct" in s
