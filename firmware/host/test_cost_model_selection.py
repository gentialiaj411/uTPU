"""Tests for Phase 1: `cost_model.select` as a real selection component
plus a replay regression against the committed selection artifact.

The replay thresholds below were captured from the locked artifact at
``bench/results/cost_model_selection.json`` (`top1_accuracy=0.292`,
`max_regret_pct=8.04`, `within_5pct_fraction=0.875`,
`within_10pct_fraction=1.0`). They are data-derived floors, not invented
gates: tightening them requires re-running the benchmark and updating
this file with the freshly measured numbers.
"""

import json
from pathlib import Path

import pytest

from cost_model import CostModelChoice, _confidence_from_margin_pct, select


def test_select_returns_minimum_predicted_latency():
    candidates = [
        {"threads_per_block": 256, "unroll_factor": 8},
        {"threads_per_block": 32, "unroll_factor": 1},
        {"threads_per_block": 128, "unroll_factor": 4},
    ]
    shape = {"out_features": 256, "in_features": 256, "batch": 1, "array_size": 16, "apply_quant": True}
    choice = select(shape, candidates, target="cuda")
    assert isinstance(choice, CostModelChoice)
    assert choice.rank == 1
    assert choice.candidates_considered == 3
    assert choice.predicted_latency_us > 0.0
    runner = choice.runner_up_predicted_latency_us
    assert runner is not None and runner >= choice.predicted_latency_us
    assert choice.target_name == "cuda"


def test_select_metadata_is_internally_consistent():
    candidates = [
        {"threads_per_block": 32, "unroll_factor": 1},
        {"threads_per_block": 64, "unroll_factor": 2},
        {"threads_per_block": 128, "unroll_factor": 4},
    ]
    shape = {"out_features": 512, "in_features": 1024, "batch": 1, "array_size": 16, "apply_quant": True}
    choice = select(shape, candidates, target="cuda")

    runner_us = choice.runner_up_predicted_latency_us
    assert runner_us is not None
    expected_margin_us = runner_us - choice.predicted_latency_us
    assert choice.margin_us == pytest.approx(expected_margin_us, rel=1e-9, abs=1e-9)
    expected_margin_pct = (expected_margin_us / choice.predicted_latency_us) * 100.0
    assert choice.margin_pct == pytest.approx(expected_margin_pct, rel=1e-9, abs=1e-9)
    assert choice.score == pytest.approx(-choice.predicted_latency_us, rel=1e-12)
    assert choice.confidence == pytest.approx(_confidence_from_margin_pct(choice.margin_pct), rel=1e-9, abs=1e-12)
    assert 0.0 <= choice.confidence <= 1.0


def test_select_single_candidate_has_zero_margin_zero_confidence():
    candidates = [{"threads_per_block": 128, "unroll_factor": 4}]
    shape = {"out_features": 256, "in_features": 256, "batch": 1, "array_size": 16, "apply_quant": True}
    choice = select(shape, candidates, target="cuda")
    assert choice.runner_up_schedule is None
    assert choice.runner_up_predicted_latency_us is None
    assert choice.margin_us == 0.0
    assert choice.margin_pct == 0.0
    assert choice.confidence == 0.0


def test_select_is_stable_on_exact_ties():
    """Coefficient set that zeros every schedule-dependent term, so all
    candidates score identically. Tie-break must be deterministic on
    insertion order."""
    target = {
        "name": "cuda",
        "cost_model_coefficients": {
            "intercept_us": 1.0,
            "memory_us_per_kib": 0.0,
            "cta_memory_us_per_kib": 0.0,
            "underoccupancy_penalty_us": 0.0,
            "tile_tail_penalty_us": 0.0,
            "unroll_gain_us": 0.0,
            "unroll_k_tail_penalty_us": 0.0,
            "unroll_shape_interaction_us": 0.0,
            "small_out_tpb_interaction_us": 0.0,
            "small_out_unroll_interaction_us": 0.0,
            "idle_thread_ratio_us": 0.0,
            "wave_tpb_interaction_us": 0.0,
            "small_out_idle_penalty_us": 0.0,
            "large_k_unroll_gain_us": 0.0,
            "small_out_unroll_penalty_us": 0.0,
            "large_out_small_k_wave_tpb_efficiency_us": 0.0,
        },
    }
    candidates = [
        {"threads_per_block": 256, "unroll_factor": 8},
        {"threads_per_block": 32, "unroll_factor": 1},
        {"threads_per_block": 128, "unroll_factor": 4},
    ]
    shape = {"out_features": 128, "in_features": 128, "batch": 1, "array_size": 16, "apply_quant": True}
    choice_a = select(shape, candidates, target=target)
    choice_b = select(shape, list(reversed(candidates)), target=target)
    # Both calls see identical predicted latencies for every candidate;
    # tie-break picks the first-listed schedule.
    assert choice_a.schedule == candidates[0]
    assert choice_b.schedule == list(reversed(candidates))[0]
    assert choice_a.margin_pct == 0.0
    assert choice_b.margin_pct == 0.0


def test_runtime_plan_records_cuda_choice():
    """`build_graph_runtime_plan(target="cuda")` invokes the default selector
    and records the cost-model choice on each LINEAR op."""
    pytest.importorskip("torch")
    pytest.importorskip("torch.fx")
    import torch
    import torch.fx as fx
    import torch.nn as nn

    from fx_importer import import_fx_graph_module
    from graph_passes import GraphPassManager
    from graph_runtime_plan import build_graph_runtime_plan

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128, bias=False)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 32, bias=False)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    gm = fx.symbolic_trace(M().eval())
    graph = import_fx_graph_module(gm)
    passed = GraphPassManager(target_backend="cuda").run(graph)
    plan = build_graph_runtime_plan(passed.graph, target="cuda")

    linear_ops = [op for op in plan.ops if op.op == "linear"]
    assert linear_ops, "expected at least one LINEAR op in the plan"
    for op in linear_ops:
        assert op.cuda_schedule is not None, f"{op.graph_op}: cost model should commit a schedule"
        assert {"threads_per_block", "unroll_factor"} == set(op.cuda_schedule.keys())
        prov = op.cuda_schedule_provenance
        assert prov is not None and prov["selector"] == "cost_model.select"
        assert prov["candidates_considered"] >= 4
        assert prov["target_name"] == "cuda"


def test_runtime_plan_selector_disables_when_target_is_not_cuda():
    pytest.importorskip("torch")
    pytest.importorskip("torch.fx")
    import torch.fx as fx
    import torch.nn as nn

    from fx_importer import import_fx_graph_module
    from graph_passes import GraphPassManager
    from graph_runtime_plan import build_graph_runtime_plan

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 16, bias=False)

        def forward(self, x):
            return self.fc1(x)

    gm = fx.symbolic_trace(M().eval())
    graph = import_fx_graph_module(gm)
    passed = GraphPassManager(target_backend="utpu").run(graph)
    plan = build_graph_runtime_plan(passed.graph, target="utpu")
    for op in plan.ops:
        if op.op == "linear":
            assert op.cuda_schedule is None
            assert op.cuda_schedule_provenance is None


# ---------------------------------------------------------------------------
# Replay regression against the committed selection artifact.
# Floors below are the actually-measured values minus a small tolerance.
# Re-run `python firmware/host/run_cost_model_selection.py` and update
# this block if the calibration is refit.
# ---------------------------------------------------------------------------
_ARTIFACT_PATH = Path("bench/results/cost_model_selection.json")

_SELECTION_FLOORS = {
    # measured 0.292; floor 0.25 leaves ~4pp headroom for stochastic refit
    "top1_accuracy_min": 0.25,
    # measured 8.04; ceiling 10.0 leaves ~2pp headroom
    "max_regret_pct_max": 10.0,
    # measured 7.38; ceiling 9.0 leaves ~1.6pp headroom
    "p95_regret_pct_max": 9.0,
    # measured 2.56; ceiling 4.0 leaves ~1.4pp headroom
    "mean_regret_pct_max": 4.0,
    # measured 0.875; floor 0.80 leaves ~7.5pp headroom
    "within_5pct_fraction_min": 0.80,
    # measured 1.0; floor 0.95 leaves 5pp headroom
    "within_10pct_fraction_min": 0.95,
    # measured 24; require at least 20 calibrated shapes survive
    "shape_count_min": 20,
}


def test_cost_model_selection_artifact_within_envelope():
    """Lock-in regression for the committed selection artifact. Mirrors
    the pattern used by `test_cost_model_regression.py`: measured floors,
    not invented thresholds."""
    if not _ARTIFACT_PATH.exists():
        pytest.skip(
            "bench/results/cost_model_selection.json missing; "
            "run `python firmware/host/run_cost_model_selection.py` to generate it."
        )
    report = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
    summary = report["summary"]

    assert int(summary["shape_count"]) >= _SELECTION_FLOORS["shape_count_min"], (
        f"shape_count={summary['shape_count']} below floor "
        f"{_SELECTION_FLOORS['shape_count_min']}"
    )
    assert float(summary["top1_accuracy"]) >= _SELECTION_FLOORS["top1_accuracy_min"], (
        f"top1_accuracy={summary['top1_accuracy']:.4f} below floor "
        f"{_SELECTION_FLOORS['top1_accuracy_min']:.4f}"
    )
    assert float(summary["mean_regret_pct"]) <= _SELECTION_FLOORS["mean_regret_pct_max"], (
        f"mean_regret_pct={summary['mean_regret_pct']:.4f} exceeds ceiling "
        f"{_SELECTION_FLOORS['mean_regret_pct_max']:.4f}"
    )
    assert float(summary["p95_regret_pct"]) <= _SELECTION_FLOORS["p95_regret_pct_max"], (
        f"p95_regret_pct={summary['p95_regret_pct']:.4f} exceeds ceiling "
        f"{_SELECTION_FLOORS['p95_regret_pct_max']:.4f}"
    )
    assert float(summary["max_regret_pct"]) <= _SELECTION_FLOORS["max_regret_pct_max"], (
        f"max_regret_pct={summary['max_regret_pct']:.4f} exceeds ceiling "
        f"{_SELECTION_FLOORS['max_regret_pct_max']:.4f}"
    )
    assert float(summary["within_5pct_fraction"]) >= _SELECTION_FLOORS["within_5pct_fraction_min"], (
        f"within_5pct_fraction={summary['within_5pct_fraction']:.4f} below floor "
        f"{_SELECTION_FLOORS['within_5pct_fraction_min']:.4f}"
    )
    assert float(summary["within_10pct_fraction"]) >= _SELECTION_FLOORS["within_10pct_fraction_min"], (
        f"within_10pct_fraction={summary['within_10pct_fraction']:.4f} below floor "
        f"{_SELECTION_FLOORS['within_10pct_fraction_min']:.4f}"
    )


def test_cost_model_selection_artifact_per_shape_consistency():
    if not _ARTIFACT_PATH.exists():
        pytest.skip("artifact missing")
    report = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
    for row in report["per_shape"]:
        chosen = row["chosen_measured_latency_us_median"]
        oracle = row["oracle_measured_latency_us_median"]
        regret = row["regret_pct"]
        # regret must be non-negative and consistent with the recorded medians
        assert regret >= -1e-6
        recomputed = (chosen - oracle) / max(oracle, 1e-9) * 100.0
        assert abs(regret - recomputed) < 1e-3
        # if is_top1 then chosen schedule equals oracle schedule
        if row["is_top1"]:
            assert row["chosen_schedule"] == row["oracle_schedule"]
            assert regret == pytest.approx(0.0, abs=1e-6)
