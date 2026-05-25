"""Phase 7 remediation P2.2 contract: bench/results/selection_ab.json.

Locks the schema of the selection A/B artifact in both modes:
- `status="ok"` (CUDA host populated the per-shape A/B measurements +
  realized regret),
- `status="cuda_unavailable"` (CPU-only host stub; methodology and
  shape count preserved so the test still gates the schema).

Live-mode (`status="ok"`) assertions are guarded by a runtime check so
the test stays green on a non-CUDA host while still failing the build
if the live artifact drops required fields on a GPU host.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "bench" / "results" / "selection_ab.json"
)


def _load() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip("bench/results/selection_ab.json not present; "
                    "regenerate with run_selection_ab.py")
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_live(artifact: dict) -> bool:
    return artifact.get("status") == "ok"


def test_artifact_has_top_level_required_fields():
    artifact = _load()
    for key in (
        "generated_at_utc",
        "status",
        "shape_count",
        "methodology",
        "per_shape",
        "aggregate",
    ):
        assert key in artifact, f"selection_ab.json missing top-level '{key}'"
    assert artifact["status"] in {"ok", "cuda_unavailable"}, (
        f"unexpected status='{artifact['status']}'"
    )


def test_methodology_block_is_locked():
    artifact = _load()
    method = artifact["methodology"]
    for key in (
        "api",
        "what_it_measures",
        "realized_regret_pct",
        "warmup_iters",
        "measurement_iters",
        "array_size",
        "timing",
        "shape_source",
        "consumed_by_runtime",
        "dtype_caveats",
    ):
        assert key in method, f"selection_ab.json methodology missing '{key}'"
    assert "cost-model-selected schedule" in method["what_it_measures"]
    assert "schedule_source='cost_model'" in method["consumed_by_runtime"], (
        "methodology must explicitly link the A schedule to "
        "schedule_source='cost_model' in CompiledMLPRuntime so a reader "
        "knows the runtime *consumes* the cost-model choice, not just "
        "records it (this is the P2.1 contract)."
    )


def test_stub_path_carries_regenerate_instructions():
    artifact = _load()
    if _is_live(artifact):
        pytest.skip("live artifact mode; regenerate instructions are not required")
    assert "reason" in artifact
    assert "regenerate_with" in artifact
    assert "run_selection_ab.py" in artifact["regenerate_with"]


def test_ok_artifact_per_shape_layout_locked():
    artifact = _load()
    if not _is_live(artifact):
        pytest.skip(f"cuda_unavailable: {artifact.get('reason')}")
    assert artifact["per_shape"], "live artifact must populate per_shape"
    for entry in artifact["per_shape"]:
        for key in (
            "shape", "seed", "cost_model_run", "oracle_run",
            "realized_regret_pct", "schedules_identical", "predicted",
        ):
            assert key in entry, f"per_shape entry missing '{key}'"
        for k in ("out_features", "in_features", "array_size"):
            assert k in entry["shape"]
        for run_key in ("schedule", "kernel_ms", "bit_exact_vs_numpy_oracle"):
            assert run_key in entry["cost_model_run"]
            assert run_key in entry["oracle_run"]
        for stat in ("mean", "median", "stdev", "min", "max", "p95", "samples"):
            assert stat in entry["cost_model_run"]["kernel_ms"]
            assert stat in entry["oracle_run"]["kernel_ms"]


def test_ok_artifact_is_bit_exact_for_both_schedules():
    """The honest contract: A and B must both produce the correct INT4
    output for the shape. If either schedule diverges from the NumPy
    oracle, the gap is meaningless and the test must fail (rather than
    silently report a regret number against wrong outputs).
    """
    artifact = _load()
    if not _is_live(artifact):
        pytest.skip(f"cuda_unavailable: {artifact.get('reason')}")
    for entry in artifact["per_shape"]:
        assert entry["cost_model_run"]["bit_exact_vs_numpy_oracle"], (
            f"cost-model schedule produced incorrect output at "
            f"shape={entry['shape']}"
        )
        assert entry["oracle_run"]["bit_exact_vs_numpy_oracle"], (
            f"oracle schedule produced incorrect output at "
            f"shape={entry['shape']}"
        )


def test_ok_artifact_aggregate_within_5pct_floor():
    """Soft sanity floor: the predicted regret aggregate in
    cost_model_selection.json reports within_5pct_fraction = 0.875.
    The realized aggregate should not be wildly worse than the
    predicted one. Floor at 0.5 so this gate flags catastrophic drift
    (e.g. cost-model not actually consumed end-to-end) but allows for
    measurement noise on a different GPU. This is the P2.1+P2.2
    end-to-end signal: if the runtime is silently re-searching, the
    realized regret distribution will diverge from the predicted one
    far below this floor.
    """
    artifact = _load()
    if not _is_live(artifact):
        pytest.skip(f"cuda_unavailable: {artifact.get('reason')}")
    agg = artifact["aggregate"]
    within_5 = agg.get("realized_within_5pct_fraction")
    assert within_5 is not None
    assert within_5 >= 0.5, (
        f"realized_within_5pct_fraction={within_5} -- if the backend is "
        "actually consuming the cost-model schedule on the same shapes "
        "the calibration ran on, > 0.5 is the soft floor"
    )


def test_ok_artifact_environment_is_recorded():
    artifact = _load()
    if not _is_live(artifact):
        pytest.skip(f"cuda_unavailable: {artifact.get('reason')}")
    env = artifact.get("environment", {})
    for key in ("python_version", "device_name", "torch_version"):
        assert key in env and env[key], f"environment.{key} must be populated on the ok path"


def test_ok_artifact_predicted_link_is_populated_for_every_shape():
    artifact = _load()
    if not _is_live(artifact):
        pytest.skip(f"cuda_unavailable: {artifact.get('reason')}")
    for entry in artifact["per_shape"]:
        predicted = entry.get("predicted")
        assert predicted is not None, (
            f"shape {entry['shape']} missing predicted block from "
            "cost_model_selection.json; the A/B harness must stitch the "
            "two signals so a single artifact reader can compare "
            "predicted regret to realized regret."
        )
        for key in ("predicted_regret_pct", "chosen_schedule", "oracle_schedule"):
            assert key in predicted
