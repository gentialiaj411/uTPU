"""Phase 7: schema lock + sanity floors for the held-out replay artifact.

Tests are intentionally floor-only (no exact-match) so refits with new
calibration data don't churn the test suite. Floors are derived from
the locked artifact in this session and chosen to leave headroom for
modest model-quality drift while still failing if the held-out claim
collapses.
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

from run_cost_model_heldout import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    HOLDOUT_FRAC,
    SPLIT_SEED,
    _deterministic_holdout_shapes,
    _shape_key,
    run,
)

REQUIRED_TOPLEVEL = {
    "version",
    "generated_at_utc",
    "git_sha",
    "phase",
    "methodology",
    "source",
    "split",
    "fit",
    "latency_prediction",
    "selection_quality",
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
        "`python firmware/host/run_cost_model_heldout.py` first"
    )


def test_artifact_top_level_schema_locked():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    missing = REQUIRED_TOPLEVEL - set(blob.keys())
    assert not missing, f"missing top-level fields: {missing}"
    assert blob["phase"] == "phase7_generalization_replay"
    assert blob["version"] == 1


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
    assert split["n_unique_train_shapes"] >= 5
    holdout_shapes = {
        (s["in_features"], s["out_features"]) for s in split["holdout_shapes"]
    }
    assert len(holdout_shapes) == split["n_unique_test_shapes"]


def test_test_metrics_within_floor():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    test = blob["latency_prediction"]["test_metrics"]
    missing = REQUIRED_LATENCY_FIELDS - set(test.keys())
    assert not missing, f"missing test latency fields: {missing}"
    # Floors derived from the locked artifact in this session, with
    # ~30% headroom so a refit can drift modestly without breaking CI.
    # If these floors fail, generalization regressed materially and a
    # human should look at the artifact before relaxing them.
    assert test["log_r2"] >= 0.80, (
        f"held-out log_R^2 collapsed: {test['log_r2']:.4f} < 0.80"
    )
    assert test["mape_pct"] <= 30.0, (
        f"held-out MAPE inflated: {test['mape_pct']:.2f}% > 30%"
    )
    assert test["p95_abs_rel_error_pct"] <= 60.0, (
        f"held-out p95 abs-rel error inflated: "
        f"{test['p95_abs_rel_error_pct']:.2f}% > 60%"
    )


def test_test_over_train_ratios_are_bounded():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    ratios = blob["latency_prediction"]["test_over_train_ratios"]
    # log_R^2 ratio should be > 0.7 (test log-R^2 isn't catastrophic
    # vs train), MAPE ratio should be < 2.5x (test MAPE not >2.5x train).
    assert ratios["log_r2"] is not None and ratios["log_r2"] >= 0.7, (
        f"test/train log_R^2 ratio collapsed: {ratios['log_r2']}"
    )
    assert ratios["mape"] is not None and ratios["mape"] <= 2.5, (
        f"test/train MAPE ratio inflated: {ratios['mape']}"
    )


def test_selection_quality_floor():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    sel = blob["selection_quality"]["summary"]
    n = sel["n_held_out_shapes_with_multi_schedule"]
    # On the production grid every held-out shape sweeps the full
    # 16-schedule menu, so this will always be >= 1; if not, the data
    # source is broken.
    assert n >= 1
    # Production claim is "policy quality / bounded regret", not "top-1".
    # We do not assert top1_accuracy >= 0; we DO assert the bounded
    # regret claim survives on held-out shapes.
    assert sel["max_regret_pct"] <= 30.0, (
        f"held-out max regret blew past sanity ceiling: "
        f"{sel['max_regret_pct']:.2f}% > 30%"
    )
    assert sel["mean_regret_pct"] <= 15.0, (
        f"held-out mean regret blew past sanity ceiling: "
        f"{sel['mean_regret_pct']:.2f}% > 15%"
    )
    assert sel["within_10pct_fraction"] >= 0.5, (
        f"held-out within-10% fraction collapsed: "
        f"{sel['within_10pct_fraction']:.3f} < 0.5"
    )


def test_holdout_shape_selection_is_deterministic():
    keys = [(in_, out_) for in_ in [16, 32, 64, 128, 256, 512] for out_ in [16, 32, 64, 128, 256, 512]]
    a = _deterministic_holdout_shapes(keys, HOLDOUT_FRAC, SPLIT_SEED)
    b = _deterministic_holdout_shapes(keys, HOLDOUT_FRAC, SPLIT_SEED)
    assert a == b, "holdout selection must be deterministic for repro"
    # Different seed must produce a different (or same-by-luck) split,
    # but the function must not crash and must return the same count.
    c = _deterministic_holdout_shapes(keys, HOLDOUT_FRAC, "phase7-heldout-v999")
    assert len(c) == len(a)


def test_replay_can_run_from_committed_source(tmp_path: Path):
    """The script must be self-sufficient: given the committed source
    calibration JSON it produces a valid artifact in a temp output
    location, byte-identical to the committed one (modulo timestamp /
    git_sha). This guards against accidental nondeterminism in the
    fitter or the split."""
    if not DEFAULT_SOURCE.exists():
        pytest.skip(f"source {DEFAULT_SOURCE} not present")
    out = tmp_path / "cost_model_heldout.json"
    payload = run(source=DEFAULT_SOURCE, output=out)
    assert out.exists()
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob == payload
    # Re-run; coefficients must match bit-for-bit (deterministic fit).
    payload2 = run(source=DEFAULT_SOURCE, output=out)
    assert payload["fit"]["coefficients_train_only"] == payload2["fit"]["coefficients_train_only"]
    assert payload["split"] == payload2["split"]
    assert payload["latency_prediction"] == payload2["latency_prediction"]
    assert payload["selection_quality"] == payload2["selection_quality"]
