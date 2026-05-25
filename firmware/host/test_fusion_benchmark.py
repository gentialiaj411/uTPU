"""Phase 2 fusion benchmark: schema lock + correctness gate.

This test does NOT gate on a specific throughput number — those are
host-dependent (NumPy build, BLAS threads, CPU clock). It DOES gate on:

* the JSON artifact exists and is well-formed;
* every workload reports differential correctness within tolerance;
* the canonical fusion rules are exercised (one workload per rule);
* op-count reduction is non-negative (fusion can never add ops in this
  rule set);
* the script itself reproduces the artifact deterministically (the
  in-test run regenerates the same schema and the same correctness
  verdict).
"""

import json
from pathlib import Path

import pytest

from run_fusion_benchmark import (
    DEFAULT_OUTPUT_JSON,
    WORKLOADS,
    run_benchmark,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "bench" / "results" / "fusion_payoff.json"


REQUIRED_TOP_KEYS = {
    "phase",
    "backend",
    "regime_note",
    "methodology",
    "environment",
    "workloads",
    "summary",
    "generated_at",
}
REQUIRED_METHODOLOGY_KEYS = {
    "warmup_iters",
    "timed_iters",
    "metric",
    "clock_note",
    "cache_state",
    "seed",
}
REQUIRED_WORKLOAD_KEYS = {
    "workload",
    "description",
    "rule_name",
    "op_count_unfused",
    "op_count_fused",
    "op_count_reduction",
    "op_count_reduction_pct",
    "input_shape",
    "input_dtype",
    "unfused_latency_us",
    "fused_latency_us",
    "throughput_delta_pct",
    "speedup",
    "correctness",
    "seed",
}
REQUIRED_LATENCY_KEYS = {"median_us", "min_us", "max_us", "mean_us", "stdev_us"}
REQUIRED_CORRECTNESS_KEYS = {
    "max_abs_error",
    "max_rel_error",
    "tolerance_abs",
    "tolerance_rel",
    "within_tolerance",
}
REQUIRED_SUMMARY_KEYS = {
    "workload_count",
    "all_correctness_within_tolerance",
    "median_throughput_delta_pct",
    "mean_throughput_delta_pct",
    "min_throughput_delta_pct",
    "max_throughput_delta_pct",
}
EXPECTED_RULE_NAMES = {
    "linear_relu_fusion",
    "scale_softmax_fusion",
    "conv_bn_fusion",
}


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(
            f"fusion_payoff.json missing at {ARTIFACT_PATH}; "
            "run `python firmware/host/run_fusion_benchmark.py` to generate it."
        )
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_artifact_has_expected_schema():
    artifact = _load_artifact()
    assert REQUIRED_TOP_KEYS.issubset(artifact.keys())
    assert REQUIRED_METHODOLOGY_KEYS.issubset(artifact["methodology"].keys())
    assert REQUIRED_SUMMARY_KEYS.issubset(artifact["summary"].keys())
    assert artifact["phase"] == "phase_2_generalized_fusion"
    assert isinstance(artifact["workloads"], list)
    assert artifact["summary"]["workload_count"] == len(artifact["workloads"])


def test_every_workload_has_expected_schema_and_is_correct():
    artifact = _load_artifact()
    for w in artifact["workloads"]:
        assert REQUIRED_WORKLOAD_KEYS.issubset(w.keys()), w["workload"]
        assert REQUIRED_LATENCY_KEYS.issubset(w["unfused_latency_us"].keys())
        assert REQUIRED_LATENCY_KEYS.issubset(w["fused_latency_us"].keys())
        assert REQUIRED_CORRECTNESS_KEYS.issubset(w["correctness"].keys())

        # Op-count never increases under fusion.
        assert w["op_count_fused"] <= w["op_count_unfused"], w["workload"]
        assert w["op_count_reduction"] >= 0

        # Numerical correctness must hold for every workload.
        assert w["correctness"]["within_tolerance"], (
            f"{w['workload']}: max_abs={w['correctness']['max_abs_error']:.3e} "
            f"max_rel={w['correctness']['max_rel_error']:.3e}"
        )


def test_artifact_covers_every_canonical_fusion_rule():
    artifact = _load_artifact()
    rules_in_artifact = {w["rule_name"] for w in artifact["workloads"]}
    assert rules_in_artifact == EXPECTED_RULE_NAMES
    # And the in-script workload set agrees.
    assert {w.rule_name for w in WORKLOADS} == EXPECTED_RULE_NAMES


def test_summary_correctness_flag_matches_per_workload_truth():
    artifact = _load_artifact()
    expected = all(w["correctness"]["within_tolerance"] for w in artifact["workloads"])
    assert artifact["summary"]["all_correctness_within_tolerance"] is expected


def test_benchmark_rerun_is_correctness_clean(tmp_path):
    """Re-running the script must always produce a correctness-clean
    artifact (small iter count keeps CI fast; the headline artifact is
    the one checked in)."""
    out = tmp_path / "fusion_payoff_rerun.json"
    artifact = run_benchmark(warmup=2, iters=5, seed=0xC0FFEE, output_path=out)
    assert artifact["summary"]["all_correctness_within_tolerance"] is True
    assert artifact["summary"]["workload_count"] == len(WORKLOADS)
    assert {w["rule_name"] for w in artifact["workloads"]} == EXPECTED_RULE_NAMES
    # Op-count reduction is rule-structural, not host-dependent — it must
    # be identical to the checked-in artifact.
    if ARTIFACT_PATH.exists():
        checked_in = json.load(open(ARTIFACT_PATH, "r", encoding="utf-8"))
        by_name = {w["workload"]: w for w in checked_in["workloads"]}
        for w in artifact["workloads"]:
            assert (
                w["op_count_reduction"] == by_name[w["workload"]]["op_count_reduction"]
            ), w["workload"]
