"""Schema lock + sanity floors for the uTPU cycle cost-model held-out artifact.

Mirrors ``test_cost_model_heldout.py`` (CUDA) but gates RTL-cycle predictability.
Floors are intentional headroom so modest refits do not churn CI.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from run_utpu_cycle_model_heldout import (  # noqa: E402
    DEFAULT_OUTPUT,
    HOLDOUT_FRAC,
    SPLIT_SEED,
    _deterministic_holdout_shapes,
    run,
)

REQUIRED_TOPLEVEL = {
    "version",
    "generated_at_utc",
    "git_sha",
    "phase",
    "methodology",
    "split",
    "fit",
    "latency_prediction",
    "selection_quality",
    "finding",
}

REQUIRED_METHODOLOGY = {
    "harness",
    "fit_function",
    "split_unit",
    "holdout_fraction",
    "split_seed",
    "split_policy",
    "selection_metric_definition",
    "claims_scope",
}

REQUIRED_LATENCY_FIELDS = {"log_r2", "mape_pct", "p95_abs_rel_error_pct"}


def test_committed_artifact_exists():
    assert DEFAULT_OUTPUT.exists(), (
        f"{DEFAULT_OUTPUT} missing; run "
        "`python firmware/host/run_utpu_cycle_model_heldout.py` first"
    )


def test_artifact_top_level_schema_locked():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    missing = REQUIRED_TOPLEVEL - set(blob.keys())
    assert not missing, f"missing top-level fields: {missing}"
    assert blob["phase"] == "utpu_cycle_cost_model_heldout"
    assert blob["version"] == 1
    assert blob["units"] == "rtl_total_program_cycles"


def test_methodology_block_locked():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    meth = blob["methodology"]
    missing = REQUIRED_METHODOLOGY - set(meth.keys())
    assert not missing, f"missing methodology fields: {missing}"
    assert meth["holdout_fraction"] == HOLDOUT_FRAC
    assert meth["split_seed"] == SPLIT_SEED
    assert meth["split_unit"] == "shape_used = (in_features, out_features)"


def test_split_disjoint_and_nonempty():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    split = blob["split"]
    assert split["n_train_rows"] > 0
    assert split["n_test_rows"] > 0
    assert split["n_unique_test_shapes"] >= 1
    assert split["n_unique_train_shapes"] >= 3
    holdout_shapes = {
        (s["in_features"], s["out_features"]) for s in split["holdout_shapes"]
    }
    assert len(holdout_shapes) == split["n_unique_test_shapes"]


def test_test_metrics_within_floor():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    test = blob["latency_prediction"]["test_metrics"]
    missing = REQUIRED_LATENCY_FIELDS - set(test.keys())
    assert not missing, f"missing test latency fields: {missing}"
    assert test["log_r2"] >= 0.80, (
        f"held-out log_R^2 collapsed: {test['log_r2']:.4f} < 0.80"
    )
    assert test["mape_pct"] <= 25.0, (
        f"held-out MAPE inflated: {test['mape_pct']:.2f}% > 25%"
    )
    assert test["p95_abs_rel_error_pct"] <= 60.0, (
        f"held-out p95 abs-rel error inflated: "
        f"{test['p95_abs_rel_error_pct']:.2f}% > 60%"
    )


def test_selection_quality_floor():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    sel = blob["selection_quality"]["summary"]
    n = sel["n_held_out_shapes_with_multi_schedule"]
    assert n >= 1
    # Deterministic hardware: expect perfect or near-perfect selection.
    assert sel["max_regret_pct"] <= 5.0, (
        f"held-out max regret blew past sanity ceiling: "
        f"{sel['max_regret_pct']:.2f}% > 5%"
    )
    assert sel["mean_regret_pct"] <= 2.0, (
        f"held-out mean regret blew past sanity ceiling: "
        f"{sel['mean_regret_pct']:.2f}% > 2%"
    )
    assert sel["top1_accuracy"] >= 0.9, (
        f"held-out top-1 collapsed: {sel['top1_accuracy']:.3f} < 0.9"
    )


def test_ground_truth_sources_are_honest():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    allowed_prefixes = (
        "rtl_iverilog_",
        "systolic_characterization_seed",
        "cache",
        "isa_calibrated_from_train_rtl",
    )
    for row in blob.get("per_point", []):
        src = str(row.get("ground_truth_source") or "")
        assert any(src.startswith(p) or src == p for p in allowed_prefixes), (
            f"unexpected ground_truth_source: {src}"
        )


def test_holdout_shape_selection_is_deterministic():
    keys = [
        (in_, out_)
        for in_ in [16, 32, 48, 64]
        for out_ in [16, 32, 48, 64]
    ]
    a = _deterministic_holdout_shapes(keys, HOLDOUT_FRAC, SPLIT_SEED)
    b = _deterministic_holdout_shapes(keys, HOLDOUT_FRAC, SPLIT_SEED)
    assert a == b
    c = _deterministic_holdout_shapes(keys, HOLDOUT_FRAC, "utpu-cycle-heldout-v999")
    assert len(c) == len(a)


def test_replay_from_cache_is_deterministic(tmp_path: Path):
    """Re-fit from cached sims must be bit-stable on coeffs / metrics / split."""
    out = tmp_path / "utpu_cycle_model_heldout.json"
    payload = run(output=out, skip_iverilog=True)
    assert out.exists()
    payload2 = run(output=out, skip_iverilog=True)
    assert payload["fit"]["coefficients_train_only"] == payload2["fit"]["coefficients_train_only"]
    assert payload["split"] == payload2["split"]
    assert payload["latency_prediction"] == payload2["latency_prediction"]
    assert payload["selection_quality"] == payload2["selection_quality"]
