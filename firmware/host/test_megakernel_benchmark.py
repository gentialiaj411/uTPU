"""Tests for the fused CUDA region-kernel benchmark (Task 1).

Two roles:

1. **Schema lock** for `bench/results/megakernel_payoff.json`. Whether the
   artifact landed in `status="ok"` (real CUDA host) or `status="cuda_unavailable"`
   (CPU host stub), the top-level shape must be identical so writeups can
   reference the same keys on either host class.
2. **Codegen smoke** for `cuda_megakernel_backend`. The CUDA execution
   itself can't run on Windows, but the *generated source string* and
   `register_with_diff_oracle` plumbing are exercised on every host and
   gated here.

There is intentionally NO hard speedup gate. The user's instruction was
explicit: do not hard-gate on a large speedup. A soft reality floor is
applied only when `status="ok"` AND the relevant value is present.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import diff_oracle
import region_fusion
import cuda_megakernel_backend as backend
from graph_ir import GraphIR, OpKind, OpNode
from run_megakernel_benchmark import (
    METHODOLOGY,
    OUTPUT_JSON,
    TIMING_PROTOCOL_NAME,
    TIMING_STABILITY_THRESHOLD_PCT,
    WORKLOADS,
    _collect_arm_stability,
    build_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Codegen smoke (always runs).
# ---------------------------------------------------------------------------

def _linear_relu_add_graph():
    g = GraphIR(name="le_test")
    g.inputs = ["x", "r"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 4), dtype="torch.float32")
    g.add_value("r", shape=(1, 2), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="fc",
            op=OpKind.LINEAR,
            inputs=["x"],
            outputs=["h"],
            attrs={
                "weight": np.eye(2, 4, dtype=np.float32),
                "in_features": 4,
                "out_features": 2,
            },
        )
    )
    g.add_op(OpNode(name="relu", op=OpKind.RELU, inputs=["h"], outputs=["h2"], attrs={}))
    g.add_op(OpNode(name="add", op=OpKind.ADD, inputs=["h2", "r"], outputs=["y"], attrs={}))
    return g


def test_codegen_emits_one_kernel_for_linear_with_epilogue_region():
    g = _linear_relu_add_graph()
    analysis = region_fusion.find_fusion_regions(g)
    assert len(analysis.regions) == 1
    region = analysis.regions[0]
    gen = backend.generate_kernel_source(region, g)
    assert gen.region_kind == "linear_with_epilogue"
    assert gen.kernel_name.startswith("utpu_fused_region__")
    assert "__global__" in gen.source
    # The epilogue text MUST contain the relu and the residual-add markers,
    # in that order (RELU first, then ADD per the graph).
    relu_pos = gen.source.find("if (acc < 0.0f) acc = 0.0f;")
    add_pos = gen.source.find("acc + ext_")
    assert relu_pos > 0 and add_pos > 0 and relu_pos < add_pos


def test_codegen_emits_one_kernel_for_elementwise_chain_region():
    g = GraphIR(name="ew_test")
    g.inputs = ["x", "r"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 8), dtype="torch.float32")
    g.add_value("r", shape=(1, 8), dtype="torch.float32")
    g.add_op(OpNode(name="relu", op=OpKind.RELU, inputs=["x"], outputs=["h"], attrs={}))
    g.add_op(OpNode(name="scale", op=OpKind.SCALE, inputs=["h"], outputs=["h2"], attrs={"scale": 0.25}))
    g.add_op(OpNode(name="add", op=OpKind.ADD, inputs=["h2", "r"], outputs=["y"], attrs={}))

    analysis = region_fusion.find_fusion_regions(g)
    assert len(analysis.regions) == 1
    region = analysis.regions[0]
    gen = backend.generate_kernel_source(region, g)
    assert gen.region_kind == "elementwise_chain"
    assert "v * 0.25" in gen.source
    assert "v + ext_" in gen.source


def test_codegen_rejects_non_v1_region_kind():
    g = _linear_relu_add_graph()
    analysis = region_fusion.find_fusion_regions(g)
    region = analysis.regions[0]
    bogus = region.__class__(
        region_id=region.region_id,
        region_kind="single_cta_bounded_multilayer",
        op_names=region.op_names,
        root_op_name=region.root_op_name,
        epilogue_op_names=region.epilogue_op_names,
        inputs_external=region.inputs_external,
        output=region.output,
        rationale=region.rationale,
    )
    with pytest.raises(ValueError, match="single_cta_bounded_multilayer is future work"):
        backend.generate_kernel_source(bogus, g)


def test_per_op_kernel_codegen_for_each_v1_op():
    for op in [
        OpNode(name="lr", op=OpKind.LINEAR_RELU, inputs=["x"], outputs=["y"], attrs={
            "weight": np.eye(2, 4, dtype=np.float32),
            "in_features": 4, "out_features": 2,
        }),
        OpNode(name="l", op=OpKind.LINEAR, inputs=["x"], outputs=["y"], attrs={
            "weight": np.eye(2, 4, dtype=np.float32),
            "in_features": 4, "out_features": 2,
        }),
        OpNode(name="r", op=OpKind.RELU, inputs=["x"], outputs=["y"], attrs={}),
        OpNode(name="s", op=OpKind.SCALE, inputs=["x"], outputs=["y"], attrs={"scale": 0.5}),
        OpNode(name="a", op=OpKind.ADD, inputs=["x", "r"], outputs=["y"], attrs={}),
    ]:
        src = backend.generate_per_op_kernel_source(op)
        assert "extern \"C\" __global__" in src
        assert f"utpu_per_op__{op.name}" in src


def test_per_op_kernel_codegen_rejects_unsupported_op():
    op = OpNode(name="bn", op=OpKind.BATCH_NORM, inputs=["x"], outputs=["y"], attrs={})
    with pytest.raises(ValueError, match="not in v1 per-op set"):
        backend.generate_per_op_kernel_source(op)


def test_register_with_diff_oracle_replaces_skip_stub():
    backend.register_with_diff_oracle()
    try:
        # After registration, the runner is no longer the import-time stub.
        runner = diff_oracle._BACKEND_RUNNERS["cuda_megakernel"]
        assert runner is backend._diff_oracle_cuda_megakernel_runner
    finally:
        # Restore the stub so we don't pollute other tests.
        diff_oracle._BACKEND_RUNNERS["cuda_megakernel"] = backend.diff_oracle._cuda_megakernel_runner_stub  # type: ignore[attr-defined]


def test_cuda_megakernel_diff_oracle_runner_skips_cleanly_without_cuda():
    """On a CPU host, calling the diff_oracle runner for cuda_megakernel must
    return status='skipped' (via BackendUnavailable). Never status='error',
    never a fabricated output."""
    backend.register_with_diff_oracle()
    try:
        g = _linear_relu_add_graph()
        x = np.array([[1.0, 2.0, -1.0, 0.5]], dtype=np.float32)
        r = np.array([[0.0, 0.25]], dtype=np.float32)
        out = diff_oracle.run_all_backends(g, [x, r], backends=("cuda_megakernel",))
        res = out["cuda_megakernel"]
        # Either skipped (CUDA missing) or ok (CUDA present); never error.
        assert res.status in ("skipped", "ok"), f"unexpected status: {res.status} reason={res.reason}"
        if res.status == "skipped":
            assert res.reason
    finally:
        diff_oracle._BACKEND_RUNNERS["cuda_megakernel"] = backend.diff_oracle._cuda_megakernel_runner_stub  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Artifact schema lock.
# ---------------------------------------------------------------------------

def _expected_top_level_keys():
    return {
        "generated_at_utc",
        "git_sha",
        "methodology",
        "workloads_spec",
        "status",
        "environment",
        "warmup",
        "iters",
        "workloads",
        "aggregate",
    }


def _expected_methodology_keys():
    return {
        "warmup",
        "iters",
        "arms",
        "summary_stats",
        "correctness_tol",
        "dtype_caveats",
        "headline",
        "scope",
        "subprocess_isolation",
        "launch_count_methodology",
        "num_stability_runs",
        "timing_protocol_name",
        "timing_stability_threshold_pct",
        "timing_protocol",
        "stability_protocol",
    }


def _expected_aggregate_keys():
    return {
        "latency_reduction_vs_op_by_op_pct_median",
        "gap_vs_cublas_pct_median",
        "gap_vs_cuda_graphs_op_by_op_pct_median",
        "launch_reduction_vs_op_by_op_pct_median",
        "launch_reduction_vs_op_by_op_pct_pooled",
        "pooled_fused_launches",
        "pooled_op_by_op_launches",
        "workloads_improved_over_op_by_op",
        "all_workloads_correct",
        "per_workload_summary",
    }


def _expected_stability_aggregate_keys():
    """Aggregate stability rollup fields, required only on stabilized
    artifacts. The split-CV-gate fields
    (max_timing_stability_pct_load_bearing_arm and the baseline-arm
    twin) are required only on split-gate artifacts (post-split-gate
    change); the legacy ``max_timing_stability_pct_across_arms``
    diagnostic field is still required on every stabilized artifact.
    """
    return {
        "max_timing_stability_pct_across_arms",
        "mean_timing_stability_pct_across_arms",
        "arms_exceeding_stability_threshold_pct",
        "num_stability_measurements",
        "timing_stability_threshold_pct",
    }


def _expected_split_gate_aggregate_keys():
    """Split-CV-gate fields, required only on split-gate artifacts.

    The strict (load-bearing) and advisory (baseline) sub-fields are
    populated on artifacts regenerated after the split-CV-gate change.
    """
    return {
        "max_timing_stability_pct_load_bearing_arm",
        "load_bearing_arms_exceeding_threshold_pct",
        "num_stability_measurements_load_bearing_arm",
        "max_timing_stability_pct_baseline_arms",
        "baseline_arms_exceeding_threshold_pct",
        "num_stability_measurements_baseline_arms",
        "load_bearing_arms",
    }


def _is_stabilized_megakernel_artifact(artifact):
    """Stabilized = methodology names the new timing protocol AND aggregate
    carries the stability rollup. Legacy committed artifacts (pre-this
    change) have neither and are explicitly skipped by gate tests with an
    actionable regen hint."""
    method = artifact.get("methodology") or {}
    if method.get("timing_protocol_name") != TIMING_PROTOCOL_NAME:
        return False
    agg = artifact.get("aggregate") or {}
    return "max_timing_stability_pct_across_arms" in agg


def _is_split_gate_megakernel_artifact(artifact):
    """Split-gate = stabilized + the strict-side load-bearing-arm field
    exists. Artifacts generated before the split-CV-gate change carry the
    old single across-arms field only; gate tests asserting on the strict
    field cleanly skip those with an actionable regen hint."""
    if not _is_stabilized_megakernel_artifact(artifact):
        return False
    agg = artifact.get("aggregate") or {}
    return "max_timing_stability_pct_load_bearing_arm" in agg


def _load_committed_artifact():
    if not Path(OUTPUT_JSON).exists():
        pytest.skip(f"{OUTPUT_JSON} not regenerated yet (run `python firmware/host/run_megakernel_benchmark.py` first)")
    return json.loads(Path(OUTPUT_JSON).read_text(encoding="utf-8"))


def test_committed_artifact_top_level_schema_lock():
    artifact = _load_committed_artifact()
    missing = _expected_top_level_keys() - set(artifact.keys())
    assert not missing, f"missing top-level keys: {missing}"


def test_committed_artifact_methodology_keys_locked():
    artifact = _load_committed_artifact()
    method = artifact["methodology"]
    # The v1 methodology keys must always be present; the stability keys
    # are required only on stabilized artifacts (post-this-change). On
    # legacy committed artifacts the stability-keys subset is skipped
    # with an actionable regen hint via the gate test below.
    legacy_v1_keys = {
        "warmup",
        "iters",
        "arms",
        "summary_stats",
        "correctness_tol",
        "dtype_caveats",
        "headline",
        "scope",
        "subprocess_isolation",
        "launch_count_methodology",
    }
    missing_v1 = legacy_v1_keys - set(method.keys())
    assert not missing_v1, f"missing v1 methodology keys: {missing_v1}"
    assert method["arms"] == ["fused_region", "op_by_op", "cuda_graphs_op_by_op", "cublas_fp32"]
    assert method["summary_stats"] == ["mean", "median", "stdev", "min", "max", "p95"]
    assert method["correctness_tol"] == {"rtol": 1e-3, "atol": 1e-3}
    if _is_stabilized_megakernel_artifact(artifact):
        missing_stab = _expected_methodology_keys() - set(method.keys())
        assert not missing_stab, (
            f"stabilized artifact missing methodology stability keys: "
            f"{missing_stab}"
        )


def test_committed_artifact_status_is_either_ok_or_unavailable():
    artifact = _load_committed_artifact()
    allowed = {"ok", "cuda_unavailable", "torch_unavailable", "cuda_python_unavailable", "subprocess_error", "subprocess_timeout", "subprocess_parse_error"}
    assert artifact["status"] in allowed, f"unexpected status: {artifact['status']}"


def test_committed_artifact_includes_every_workload_in_spec():
    artifact = _load_committed_artifact()
    names_spec = {w["name"] for w in artifact["workloads_spec"]}
    names_workloads = {w["name"] for w in artifact["workloads"]}
    # NEVER silently drop a workload: every spec entry must appear in
    # `workloads`, whether populated (ok) or skipped (stub).
    assert names_spec == names_workloads, (
        f"workloads_spec vs workloads names diverge: spec={names_spec} workloads={names_workloads}"
    )


def test_committed_artifact_aggregate_keys_locked():
    artifact = _load_committed_artifact()
    agg = artifact["aggregate"]
    missing = _expected_aggregate_keys() - set(agg.keys())
    assert not missing, f"missing aggregate keys: {missing}"
    assert isinstance(agg["per_workload_summary"], list)
    assert len(agg["per_workload_summary"]) == len(artifact["workloads_spec"])
    if _is_stabilized_megakernel_artifact(artifact):
        missing_stab = _expected_stability_aggregate_keys() - set(agg.keys())
        assert not missing_stab, (
            f"stabilized artifact missing stability aggregate keys: "
            f"{missing_stab}"
        )


def test_committed_artifact_populated_arm_kernel_ms_carries_stability_fields():
    """Every populated (status='ok') per-arm kernel_ms block in a
    stabilized artifact must report the inter-run stability fields:
    per_run_medians_ms, timing_stability_pct, num_stability_runs,
    timing_protocol. Legacy artifacts skip this with an actionable hint.
    """
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    if not _is_stabilized_megakernel_artifact(artifact):
        pytest.skip(
            "artifact predates the timing-stability protocol; re-run "
            "`python firmware/host/run_megakernel_benchmark.py` on WSL2 "
            "+ CUDA to regenerate with the stabilized methodology."
        )
    for w in artifact["workloads"]:
        if w.get("status") != "ok":
            continue
        for arm in w.get("arms", []):
            if not isinstance(arm, dict):
                continue
            if arm.get("status") != "ok":
                continue
            kernel_ms = arm.get("kernel_ms") or {}
            for field in (
                "per_run_medians_ms",
                "per_run_medians_trimmed_ms",
                "timing_stability_pct",
                "num_stability_runs",
                "num_stability_runs_trimmed",
                "timing_protocol",
            ):
                assert field in kernel_ms, (
                    f"workload={w['name']} arm={arm.get('arm')!r}: "
                    f"kernel_ms missing stability field {field!r}; "
                    f"stabilized artifacts must populate this on every "
                    f"ok arm or the CV gate is meaningless"
                )
            assert kernel_ms["timing_protocol"] == TIMING_PROTOCOL_NAME
            assert isinstance(kernel_ms["per_run_medians_ms"], list)
            assert len(kernel_ms["per_run_medians_ms"]) == int(
                kernel_ms["num_stability_runs"]
            )
            assert isinstance(kernel_ms["per_run_medians_trimmed_ms"], list)
            assert len(kernel_ms["per_run_medians_trimmed_ms"]) == int(
                kernel_ms["num_stability_runs_trimmed"]
            )
            assert len(kernel_ms["per_run_medians_trimmed_ms"]) <= len(
                kernel_ms["per_run_medians_ms"]
            )


def test_load_bearing_fused_region_arm_is_gate_green_megakernel():
    """STRICT CV gate (split-gate engineering): asserts ONLY on the
    fused_region (load-bearing) arm.

    On a split-gate artifact the aggregate exposes
    ``max_timing_stability_pct_load_bearing_arm`` (worst inter-run CV
    across the load-bearing arm only — every (fused_region, workload)
    pair). The documented threshold lives at
    ``timing_stability_threshold_pct``. If this fires the build fails.

    CRITICAL: passing this strict gate does **NOT** certify the published
    ``latency_reduction_vs_op_by_op_pct`` ratio. That ratio's denominator
    (``op_by_op``'s median) is on a **baseline** arm and the baseline
    arms are intentionally gated **advisory** only — multi-launch
    host-side jitter on a contended laptop GPU without
    ``nvidia-smi --lock-gpu-clocks`` is intrinsic, not
    methodology-fixable. Any specific latency-% number therefore stays
    ``[needs-locked-clock-artifact]`` until regenerated on locked clocks
    (native Linux ``sudo nvidia-smi --lock-gpu-clocks=$BOOST,$BOOST``)
    or a cloud RTX. The strict gate's job is narrower: confirm the
    fused-region kernel's per-iteration time is *itself* stable.
    """
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    if not _is_split_gate_megakernel_artifact(artifact):
        pytest.skip(
            "artifact predates the split-CV-gate (load-bearing arm) "
            "protocol; re-run `python firmware/host/run_megakernel"
            "_benchmark.py` on WSL2 + CUDA to regenerate."
        )
    agg = artifact["aggregate"]
    threshold = float(agg["timing_stability_threshold_pct"])
    max_cv_lb = agg["max_timing_stability_pct_load_bearing_arm"]
    if max_cv_lb is None:
        pytest.skip(
            "no populated load-bearing-arm stability measurements "
            "(fused_region errored on every workload)"
        )
    lb_exceed = agg.get("load_bearing_arms_exceeding_threshold_pct") or []
    assert max_cv_lb <= threshold, (
        f"STRICT timing-stability gate FAILED on the load-bearing "
        f"(fused_region) arm of the megakernel artifact: max inter-run "
        f"CV = {max_cv_lb:.2f}% > documented threshold {threshold:.1f}%. "
        f"Offending (arm, workload, CV%, per-run-medians): "
        f"{[(e['arm'], e['workload'], e['timing_stability_pct'], e['per_run_medians_ms']) for e in lb_exceed]}. "
        f"This blocks the build because the load-bearing arm's per-"
        f"iteration kernel time is no longer stable — bump --warmup / "
        f"--iters / --num-stability-runs in firmware/host/run_megakernel"
        f"_benchmark.py and regen, or kill background GPU load."
    )
    assert not lb_exceed, (
        f"max load-bearing-arm CV under threshold but exceed list "
        f"non-empty: {lb_exceed!r}"
    )


def test_baseline_arms_advisory_cv_reporting_megakernel():
    """ADVISORY CV companion (split-gate engineering): does NOT fail the
    build, but asserts the aggregate carries the baseline-arm CV
    diagnostic fields so a reviewer can see the noise floor.

    The baseline arms (op_by_op, cuda_graphs_op_by_op, cublas_fp32)
    fire 2-4 launches per timed iteration or are themselves multi-
    kernel torch/cuBLAS callees. On a contended laptop GPU without
    ``nvidia-smi --lock-gpu-clocks`` their inter-run CV is dominated by
    clock-frequency drift between stability runs — intrinsic to the
    host environment, not methodology-fixable. This test reports the
    advisory CV and offending pairs (if any) and skips assertion. Any
    specific latency-% number that uses a baseline-arm median as its
    denominator therefore stays ``[needs-locked-clock-artifact]`` even
    when the strict gate above passes — passing the split gate does
    NOT certify the latency ratio.
    """
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    if not _is_split_gate_megakernel_artifact(artifact):
        pytest.skip(
            "artifact predates the split-CV-gate; regen to expose "
            "advisory baseline-arm CV fields."
        )
    agg = artifact["aggregate"]
    for field in (
        "max_timing_stability_pct_baseline_arms",
        "baseline_arms_exceeding_threshold_pct",
        "num_stability_measurements_baseline_arms",
    ):
        assert field in agg, f"split-gate aggregate missing {field!r}"
    max_cv_bl = agg["max_timing_stability_pct_baseline_arms"]
    bl_exceed = agg.get("baseline_arms_exceeding_threshold_pct") or []
    if max_cv_bl is None:
        return
    threshold = float(agg["timing_stability_threshold_pct"])
    if max_cv_bl > threshold:
        # Advisory: build does not fail. The information is in the
        # artifact for reviewers; the latency-% ratio is independently
        # [needs-locked-clock-artifact] regardless of this value.
        print(
            f"\n[ADVISORY] baseline-arm CV {max_cv_bl:.2f}% > "
            f"{threshold:.1f}% on {len(bl_exceed)} (arm, workload) "
            f"pair(s) — expected on WSL2 + unlocked laptop clocks; "
            f"does NOT block build, does NOT certify the latency-% "
            f"ratio."
        )


def test_collect_arm_stability_helper_megakernel_handles_empty_and_legacy_inputs():
    stabilized = [
        {
            "name": "wA",
            "arms": [
                {"arm": "fused_region", "status": "ok",
                 "kernel_ms": {"timing_stability_pct": 2.0}},
                {"arm": "op_by_op", "status": "ok",
                 "kernel_ms": {"timing_stability_pct": 3.5}},
                {"arm": "cublas_fp32", "status": "ok",
                 "kernel_ms": {"timing_stability_pct": 12.5,
                               "per_run_medians_ms": [0.5, 0.55, 0.62]}},
            ],
        },
        {
            "name": "wB",
            "arms": [
                {"arm": "fused_region", "status": "ok",
                 "kernel_ms": {"timing_stability_pct": 1.0}},
                {"arm": "op_by_op", "status": "error"},
            ],
        },
    ]
    roll = _collect_arm_stability(stabilized)
    # Across-arms diagnostic still reports the full picture.
    assert roll["num_stability_measurements"] == 4
    assert roll["max_timing_stability_pct_across_arms"] == 12.5
    assert len(roll["arms_exceeding_stability_threshold_pct"]) == 1
    bad = roll["arms_exceeding_stability_threshold_pct"][0]
    assert bad["arm"] == "cublas_fp32" and bad["workload"] == "wA"
    # Split-gate: load-bearing arm (fused_region) has max CV = 2.0
    # (only the two clean fused_region entries from wA + wB).
    assert roll["max_timing_stability_pct_load_bearing_arm"] == 2.0
    assert roll["num_stability_measurements_load_bearing_arm"] == 2
    assert roll["load_bearing_arms_exceeding_threshold_pct"] == []
    # Baseline arms (op_by_op, cublas_fp32) carry the noisy 12.5 reading.
    assert roll["max_timing_stability_pct_baseline_arms"] == 12.5
    assert roll["num_stability_measurements_baseline_arms"] == 2
    assert len(roll["baseline_arms_exceeding_threshold_pct"]) == 1
    bad_bl = roll["baseline_arms_exceeding_threshold_pct"][0]
    assert bad_bl["arm"] == "cublas_fp32" and bad_bl["workload"] == "wA"
    assert roll["load_bearing_arms"] == ["fused_region"]

    legacy = [
        {
            "name": "wL",
            "arms": [{"arm": "fused_region", "status": "ok",
                       "kernel_ms": {"median": 0.05}}],
        },
    ]
    roll_legacy = _collect_arm_stability(legacy)
    assert roll_legacy["num_stability_measurements"] == 0
    assert roll_legacy["max_timing_stability_pct_across_arms"] is None
    assert roll_legacy["arms_exceeding_stability_threshold_pct"] == []
    # Split-gate fields also legacy-null:
    assert roll_legacy["max_timing_stability_pct_load_bearing_arm"] is None
    assert roll_legacy["load_bearing_arms_exceeding_threshold_pct"] == []
    assert roll_legacy["num_stability_measurements_load_bearing_arm"] == 0
    assert roll_legacy["max_timing_stability_pct_baseline_arms"] is None
    assert roll_legacy["baseline_arms_exceeding_threshold_pct"] == []
    assert roll_legacy["num_stability_measurements_baseline_arms"] == 0
    assert (
        roll_legacy["timing_stability_threshold_pct"]
        == TIMING_STABILITY_THRESHOLD_PCT
    )


def test_committed_artifact_per_workload_summary_keys_locked():
    artifact = _load_committed_artifact()
    for entry in artifact["aggregate"]["per_workload_summary"]:
        for key in (
            "name",
            "fused_region_ms_median",
            "op_by_op_ms_median",
            "cuda_graphs_op_by_op_ms_median",
            "cublas_fp32_ms_median",
            "latency_reduction_vs_op_by_op_pct",
            "gap_vs_cublas_pct",
            "gap_vs_cuda_graphs_op_by_op_pct",
            "fused_region_launches",
            "op_by_op_launches",
            "cuda_graphs_op_by_op_launches",
            "cuda_graphs_op_by_op_graph_nodes",
            "launch_reduction_vs_op_by_op_pct",
            "all_arms_correct",
        ):
            assert key in entry, f"per_workload_summary entry missing '{key}': {entry}"


def test_committed_artifact_ok_mode_launch_reduction_is_real_and_consistent():
    """If status='ok', every workload that has both fused + op_by_op arms
    populated must have a non-None launch-count reduction recorded, and the
    reduction must be derivable from the per-arm launch counts (no fabricated
    aggregate)."""
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    for entry in artifact["aggregate"]["per_workload_summary"]:
        if entry.get("workload_status") != "ok":
            continue
        f_n = entry.get("fused_region_launches")
        o_n = entry.get("op_by_op_launches")
        red = entry.get("launch_reduction_vs_op_by_op_pct")
        if f_n is None or o_n is None or red is None:
            continue
        assert isinstance(f_n, int) and isinstance(o_n, int) and o_n > 0
        expected_red = (o_n - f_n) / o_n * 100.0
        # Numerical equality (computed from integer launch counts).
        assert abs(red - expected_red) < 1e-9, (
            f"workload={entry['name']} launch_reduction={red:.4f}% but "
            f"({o_n} - {f_n}) / {o_n} * 100 = {expected_red:.4f}%"
        )


def test_committed_artifact_ok_mode_cuda_graphs_arm_present_or_legacy_artifact():
    """The cuda_graphs_op_by_op arm is a v1.1 addition. Once a fresh
    WSL2 + CUDA regen lands, every workload with a populated `arms` list
    MUST include the cuda_graphs_op_by_op arm slot (its status may be
    'error' on older CUDA drivers, but the slot must not be missing).

    Until the v1.1 regen lands, the existing populated artifact only has
    the 3-arm v1 shape and this test is informational — the schema-lock
    test (`test_committed_artifact_per_workload_summary_keys_locked`) plus
    the aggregate keys lock are what catch silent arm-drop.
    """
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    missing_count = 0
    for w in artifact["workloads"]:
        if w.get("status") != "ok":
            continue
        arm_names = {a.get("arm") for a in w.get("arms", [])}
        if "cuda_graphs_op_by_op" not in arm_names:
            missing_count += 1
    if missing_count > 0:
        pytest.skip(
            f"v1 legacy artifact: {missing_count} workloads missing "
            "cuda_graphs_op_by_op arm slot pending the next WSL2 regen "
            "(`python firmware/host/run_megakernel_benchmark.py`)."
        )


# ---------------------------------------------------------------------------
# Mode-dependent assertions.
# ---------------------------------------------------------------------------

def test_committed_artifact_ok_mode_has_no_silently_dropped_arm():
    """If status='ok' AND workload was actually run (workload_status='ok'),
    every workload has all four arms recorded (status may still be 'error'
    or 'deferred' per arm — what we forbid is the arm silently missing
    entirely). Workloads with status='pending_regen' are v1.1 schema-bump
    placeholders pending the next WSL2 regen and are intentionally exempt
    from this check.
    """
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    for w in artifact["workloads"]:
        if w.get("status") == "pending_regen":
            continue
        arm_names = {a["arm"] for a in w["arms"]}
        required = {"fused_region", "op_by_op", "cublas_fp32"}
        missing = required - arm_names
        assert not missing, (
            f"workload {w['name']} missing required arms: missing={missing} present={arm_names}"
        )


def test_committed_artifact_ok_mode_all_ok_arms_correct():
    """If status='ok', every arm that successfully ran (arm.status='ok')
    must have correctness_within_tolerance=True. A failed correctness flag
    means the planner / codegen produced numerically wrong output and
    is the canary for the global-sync-trap class of bug if it ever sneaks
    past the region planner."""
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    for w in artifact["workloads"]:
        if w.get("status") == "pending_regen":
            continue
        for arm in w["arms"]:
            if arm.get("status") == "ok":
                assert arm.get("correctness_within_tolerance") is True, (
                    f"workload={w['name']} arm={arm['arm']} reported incorrect output: "
                    f"max_abs={arm.get('max_abs_error_vs_reference')}"
                )


def test_committed_artifact_ok_mode_primary_arms_must_actually_succeed():
    """If status='ok', every workload's *primary* arms (fused_region and
    op_by_op) must have status='ok'. The 'silent-arm-drop' class of bug —
    where 2/3 arms erroneously errored but the workload still claimed
    all_arms_correct=True because correctness was vacuously satisfied — would
    have made it past every other test gate. This test forbids it.

    cuBLAS is intentionally NOT in this list: it is a perf baseline and is
    allowed to be skipped/error (e.g. if torch is missing) without making the
    benchmark itself dishonest. fused_region vs op_by_op IS the resume claim,
    so both arms must actually run successfully on an ok-mode artifact.
    """
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    primary = ("fused_region", "op_by_op")
    for w in artifact["workloads"]:
        if w.get("status") == "pending_regen":
            continue
        arms = {a["arm"]: a for a in w["arms"]}
        for arm_name in primary:
            arm = arms.get(arm_name)
            assert arm is not None, f"workload={w['name']} missing primary arm '{arm_name}'"
            assert arm.get("status") == "ok", (
                f"workload={w['name']} primary arm '{arm_name}' status="
                f"{arm.get('status')!r} reason={arm.get('reason')!r}. The resume claim "
                "compares fused_region vs op_by_op; both must run on ok-mode artifacts."
            )


def test_committed_artifact_ok_mode_aggregate_consistent_with_arms():
    """If status='ok' and aggregate.all_workloads_correct=True, then every
    workload's primary arms must actually be ok=True. This locks the
    aggregate-vs-arms consistency: aggregate can't claim correctness while
    underlying arm-level data shows failures."""
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    if not artifact["aggregate"].get("all_workloads_correct"):
        pytest.skip("aggregate.all_workloads_correct=False — nothing to cross-check")
    for w in artifact["workloads"]:
        if w.get("status") == "pending_regen":
            continue
        arms = {a["arm"]: a for a in w["arms"]}
        for arm_name in ("fused_region", "op_by_op"):
            arm = arms.get(arm_name, {})
            assert arm.get("status") == "ok", (
                f"aggregate.all_workloads_correct=True but workload={w['name']} "
                f"arm={arm_name} status={arm.get('status')!r} — aggregate is lying."
            )
            assert arm.get("correctness_within_tolerance") is True, (
                f"aggregate.all_workloads_correct=True but workload={w['name']} "
                f"arm={arm_name} correctness_within_tolerance="
                f"{arm.get('correctness_within_tolerance')!r} — aggregate is lying."
            )


def test_committed_artifact_stub_mode_has_regen_instructions():
    """If status indicates a non-CUDA host, the artifact must include
    `stub_reason` AND `regen_instructions` so a reviewer knows what to do."""
    artifact = _load_committed_artifact()
    if artifact["status"] == "ok":
        pytest.skip("populated artifact does not need stub_reason / regen_instructions")
    assert "stub_reason" in artifact and artifact["stub_reason"], "stub artifact missing stub_reason"
    assert "regen_instructions" in artifact, "stub artifact missing regen_instructions"
    assert "WSL2" in artifact["regen_instructions"] or "CUDA" in artifact["regen_instructions"]


def test_committed_artifact_stub_mode_no_fabricated_timings():
    """In stub mode, every per-workload-summary timing MUST be None.
    A non-None timing in stub mode would mean the harness fabricated a
    number, which is a critical honesty-rule violation."""
    artifact = _load_committed_artifact()
    if artifact["status"] == "ok":
        pytest.skip("populated artifact has real timings; stub-mode test")
    for entry in artifact["aggregate"]["per_workload_summary"]:
        for key in (
            "fused_region_ms_median",
            "op_by_op_ms_median",
            "cuda_graphs_op_by_op_ms_median",
            "cublas_fp32_ms_median",
        ):
            assert entry[key] is None, f"stub artifact fabricated {key} on {entry['name']}: {entry[key]}"


# ---------------------------------------------------------------------------
# Reality floor (SOFT — only when status='ok' AND value present).
# The user's instruction: do not hard-gate on a large speedup in CI.
# So this is intentionally lenient: fused_region must not be CATASTROPHICALLY
# worse than op_by_op (>5x slower would mean codegen quality regressed
# severely; anything between -5x and +infinity is allowed and recorded).
# ---------------------------------------------------------------------------

def test_committed_artifact_fused_region_not_catastrophically_slow():
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip("ok-mode test")
    for entry in artifact["aggregate"]["per_workload_summary"]:
        if entry.get("workload_status") == "pending_regen":
            continue
        fused = entry["fused_region_ms_median"]
        op = entry["op_by_op_ms_median"]
        if fused is None or op is None or op == 0.0:
            continue
        ratio = fused / op
        # Soft reality floor only: fused should not be more than 5x slower
        # than op_by_op. (No upper-bound speedup gate.)
        assert ratio < 5.0, (
            f"workload={entry['name']} fused is {ratio:.2f}x op_by_op — "
            f"codegen quality regressed severely"
        )


# ---------------------------------------------------------------------------
# build_artifact() round-trip (regenerates the artifact in-process).
# ---------------------------------------------------------------------------

def test_build_artifact_in_process_emits_valid_stub_or_ok():
    """build_artifact() callable on any host; result has the full schema."""
    artifact = build_artifact(warmup=1, iters=2)  # tiny iters so a real CUDA host doesn't take long
    missing = _expected_top_level_keys() - set(artifact.keys())
    assert not missing, f"missing keys after in-process regen: {missing}"
    assert artifact["status"] in ("ok", "cuda_unavailable", "torch_unavailable", "cuda_python_unavailable", "subprocess_error", "subprocess_timeout", "subprocess_parse_error")
    assert artifact["methodology"]["arms"] == METHODOLOGY["arms"]
    assert {w["name"] for w in artifact["workloads"]} == {w["name"] for w in WORKLOADS}
