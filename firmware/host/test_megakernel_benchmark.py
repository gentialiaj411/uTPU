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
    WORKLOADS,
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
    }


def _expected_aggregate_keys():
    return {
        "latency_reduction_vs_op_by_op_pct_median",
        "gap_vs_cublas_pct_median",
        "workloads_improved_over_op_by_op",
        "all_workloads_correct",
        "per_workload_summary",
    }


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
    missing = _expected_methodology_keys() - set(method.keys())
    assert not missing, f"missing methodology keys: {missing}"
    assert method["arms"] == ["fused_region", "op_by_op", "cublas_fp32"]
    assert method["summary_stats"] == ["mean", "median", "stdev", "min", "max", "p95"]
    assert method["correctness_tol"] == {"rtol": 1e-3, "atol": 1e-3}


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


def test_committed_artifact_per_workload_summary_keys_locked():
    artifact = _load_committed_artifact()
    for entry in artifact["aggregate"]["per_workload_summary"]:
        for key in (
            "name",
            "fused_region_ms_median",
            "op_by_op_ms_median",
            "cublas_fp32_ms_median",
            "latency_reduction_vs_op_by_op_pct",
            "gap_vs_cublas_pct",
            "all_arms_correct",
        ):
            assert key in entry, f"per_workload_summary entry missing '{key}': {entry}"


# ---------------------------------------------------------------------------
# Mode-dependent assertions.
# ---------------------------------------------------------------------------

def test_committed_artifact_ok_mode_has_no_silently_dropped_arm():
    """If status='ok', every workload has all three arms recorded (status
    may still be 'error' or 'deferred' per arm — what we forbid is the arm
    silently missing entirely)."""
    artifact = _load_committed_artifact()
    if artifact["status"] != "ok":
        pytest.skip(f"status='{artifact['status']}' — ok-mode test")
    for w in artifact["workloads"]:
        arm_names = {a["arm"] for a in w["arms"]}
        assert arm_names == {"fused_region", "op_by_op", "cublas_fp32"}, (
            f"workload {w['name']} missing arms: present={arm_names}"
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
        for key in ("fused_region_ms_median", "op_by_op_ms_median", "cublas_fp32_ms_median"):
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
