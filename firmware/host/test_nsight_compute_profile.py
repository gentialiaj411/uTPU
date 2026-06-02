"""Tests for the Nsight Compute occupancy / bottleneck profile harness.

Two roles:

1. **Schema lock** for ``bench/results/nsight_compute_profile.json``.
   Whether the artifact landed in ``status="ok"`` (real ncu host),
   ``status="partial"`` (some workloads errored), or
   ``status="nsight_compute_unavailable"`` (ncu missing / CPU host /
   ``--skip-ncu`` stub), the top-level shape must be identical so writeups
   can reference the same keys on either host class.
2. **Bound-classification logic** for the four-way rule
   (compute_bound / memory_bound / launch_overhead_bound / balanced) —
   this rule is the load-bearing distillation in the artifact, and a
   silent re-threshold would invalidate the "is this fused kernel
   memory-bound or compute-bound?" answer to a phone screen question.
"""

import json
from pathlib import Path

import pytest

from run_nsight_compute_profile import (
    COMPUTE_BOUND_THRESHOLD_PCT,
    LAUNCH_OVERHEAD_THRESHOLD_PCT,
    LOCKED_WORKLOAD_NAMES,
    MEMORY_BOUND_THRESHOLD_PCT,
    METHODOLOGY,
    OUTPUT_JSON,
    _classify_bound,
    _compute_aggregate,
    _empty_aggregate,
    build_artifact,
)


# ---------------------------------------------------------------------------
# Bound-classification rule (always runs).
# ---------------------------------------------------------------------------

def test_classify_bound_compute():
    # SM saturated, memory cold -> compute_bound.
    assert _classify_bound(95.0, 30.0) == "compute_bound"
    assert _classify_bound(80.0, 79.99) == "compute_bound"


def test_classify_bound_memory():
    # Memory saturated, SM cold -> memory_bound.
    assert _classify_bound(50.0, 92.0) == "memory_bound"
    assert _classify_bound(79.99, 80.0) == "memory_bound"


def test_classify_bound_launch_overhead():
    # Both under launch-overhead threshold -> launch_overhead_bound.
    assert _classify_bound(15.0, 12.0) == "launch_overhead_bound"
    assert _classify_bound(39.99, 39.99) == "launch_overhead_bound"


def test_classify_bound_balanced():
    # Both nontrivial but neither saturating -> balanced.
    assert _classify_bound(60.0, 60.0) == "balanced"
    assert _classify_bound(45.0, 75.0) == "balanced"


def test_classify_bound_unknown_when_metric_missing():
    assert _classify_bound(None, 50.0) == "unclassified"
    assert _classify_bound(50.0, None) == "unclassified"
    assert _classify_bound(None, None) == "unclassified"


def test_thresholds_are_locked_and_documented_in_methodology():
    # If a future change to the thresholds is made, the methodology block
    # MUST be updated in lockstep (the artifact's bound_classification
    # rule is the contract). This test fails if they drift.
    bc = METHODOLOGY["bound_classification"]
    assert f"sm_throughput >= {COMPUTE_BOUND_THRESHOLD_PCT}%" in bc["compute_bound_iff"]
    assert f"mem_throughput >= {MEMORY_BOUND_THRESHOLD_PCT}%" in bc["memory_bound_iff"]
    assert f"sm_throughput < {LAUNCH_OVERHEAD_THRESHOLD_PCT}%" in bc["launch_overhead_bound_iff"]


# ---------------------------------------------------------------------------
# Aggregate helper.
# ---------------------------------------------------------------------------

def test_empty_aggregate_schema_matches_full_aggregate_schema():
    """The stub-mode `_empty_aggregate` and the real `_compute_aggregate`
    must emit the same set of keys — otherwise the schema-lock test fails
    in one mode but not the other."""
    empty = _empty_aggregate(["a", "b"])
    real = _compute_aggregate([
        {
            "status": "ok",
            "sm_throughput_pct": 50.0,
            "memory_throughput_pct": 60.0,
            "achieved_occupancy_pct": 70.0,
            "bottleneck_classification": "balanced",
        }
    ])
    assert set(empty.keys()) == set(real.keys()), (
        f"_empty_aggregate keys {set(empty.keys())} != _compute_aggregate keys {set(real.keys())}"
    )


def test_compute_aggregate_majority_bound_class():
    rows = [
        {"status": "ok", "sm_throughput_pct": 95, "memory_throughput_pct": 20, "achieved_occupancy_pct": 80, "bottleneck_classification": "compute_bound"},
        {"status": "ok", "sm_throughput_pct": 92, "memory_throughput_pct": 25, "achieved_occupancy_pct": 75, "bottleneck_classification": "compute_bound"},
        {"status": "ok", "sm_throughput_pct": 20, "memory_throughput_pct": 95, "achieved_occupancy_pct": 30, "bottleneck_classification": "memory_bound"},
    ]
    agg = _compute_aggregate(rows)
    assert agg["primary_bottleneck_class"] == "compute_bound"
    assert agg["bottleneck_classification_counts"] == {"compute_bound": 2, "memory_bound": 1}


# ---------------------------------------------------------------------------
# build_artifact() smoke (always runs, lands a stub on non-ncu hosts).
# ---------------------------------------------------------------------------

def test_build_artifact_returns_well_formed_artifact_on_any_host():
    artifact = build_artifact(LOCKED_WORKLOAD_NAMES)
    # Top-level schema is invariant across host classes.
    for key in (
        "generated_at_utc",
        "git_sha",
        "methodology",
        "workloads_requested",
        "environment",
        "status",
        "per_workload",
        "aggregate",
    ):
        assert key in artifact, f"missing top-level key '{key}'"
    assert artifact["status"] in (
        "ok",
        "partial",
        "nsight_compute_unavailable",
    ), f"unexpected status: {artifact['status']}"
    assert artifact["workloads_requested"] == LOCKED_WORKLOAD_NAMES


def test_build_artifact_stub_has_regen_instructions_when_ncu_missing():
    artifact = build_artifact(LOCKED_WORKLOAD_NAMES)
    if artifact["status"] == "ok":
        pytest.skip("ncu is available; stub-mode test")
    assert "regen_instructions" in artifact, "stub mode must have regen_instructions"
    assert "ncu" in artifact["regen_instructions"].lower() or "nsight" in artifact["regen_instructions"].lower()


def test_build_artifact_methodology_keys_locked():
    artifact = build_artifact(LOCKED_WORKLOAD_NAMES)
    method = artifact["methodology"]
    for key in (
        "ncu_command",
        "csv_export_command",
        "workloads_profiled",
        "arm_profiled",
        "metrics_extracted",
        "bound_classification",
        "scope",
        "honest_caveats",
        "stub_behavior",
    ):
        assert key in method, f"methodology missing '{key}'"
    assert method["arm_profiled"] == "fused_region"
    assert set(method["workloads_profiled"]) == set(LOCKED_WORKLOAD_NAMES)


# ---------------------------------------------------------------------------
# Committed-artifact schema lock (skips if artifact not regenerated yet).
# ---------------------------------------------------------------------------

def _load_committed_artifact():
    if not Path(OUTPUT_JSON).exists():
        pytest.skip(f"{OUTPUT_JSON} not regenerated yet (run `python firmware/host/run_nsight_compute_profile.py` or `--skip-ncu` first)")
    return json.loads(Path(OUTPUT_JSON).read_text(encoding="utf-8"))


def test_committed_artifact_top_level_schema_lock():
    artifact = _load_committed_artifact()
    for key in (
        "generated_at_utc",
        "git_sha",
        "methodology",
        "workloads_requested",
        "environment",
        "status",
        "per_workload",
        "aggregate",
    ):
        assert key in artifact, f"committed artifact missing '{key}'"


def test_committed_artifact_status_in_allowed_set():
    artifact = _load_committed_artifact()
    assert artifact["status"] in (
        "ok",
        "partial",
        "nsight_compute_unavailable",
    ), f"unexpected status: {artifact['status']}"


def test_committed_artifact_per_workload_schema_lock():
    artifact = _load_committed_artifact()
    for row in artifact["per_workload"]:
        assert "workload" in row
        assert row.get("status") in ("ok", "skipped", "error"), (
            f"unexpected per-workload status: {row.get('status')}"
        )
        if row.get("status") == "ok":
            for key in (
                "sm_throughput_pct",
                "memory_throughput_pct",
                "achieved_occupancy_pct",
                "bottleneck_classification",
            ):
                assert key in row, f"ok workload missing '{key}': {row}"


def test_committed_artifact_unavailable_mode_no_fabricated_metrics():
    """In stub mode (`status=nsight_compute_unavailable`) the harness must
    NOT have invented any sm/memory/occupancy percentages. Stub-mode rows
    are status='skipped' with no metric fields."""
    artifact = _load_committed_artifact()
    if artifact["status"] == "ok":
        pytest.skip("populated artifact; stub-mode test")
    for row in artifact["per_workload"]:
        for k in (
            "sm_throughput_pct",
            "memory_throughput_pct",
            "achieved_occupancy_pct",
        ):
            if k in row:
                assert row[k] is None, (
                    f"stub artifact fabricated {k} on {row.get('workload')}: {row[k]}"
                )
