"""Phase 7 tests: cuBLAS / cuBLASLt IMMA INT8 / Inductor baseline harness.

These tests intentionally do **not** gate on a performance number —
the gap depends on the GPU + driver of whichever host regenerates the
artifact. They lock the *contract* of the harness so the writeup and
the CLAIMS_MATRIX row cannot drift from the JSON layout:

* The locked shape set (``run_cublas_baseline.SHAPES``) is exactly the
  six shapes the writeup references and contains documented small /
  medium / large regions so a single number can't be cherry-picked.
* The artifact's top-level keys, methodology block, and per-shape
  layout are identical in ``status="ok"`` and ``status="cuda_unavailable"``
  modes.
* The dtype caveats are present, explicit, and call out the FP32
  Inductor / INT32 cuBLAS mismatches so a downstream reader cannot
  mistake the comparison for like-for-like. They also explicitly
  document the cuBLASLt IMMA INT8 GEMM arm (the dtype-matched
  apples-to-apples comparison) and its N=1→N=8 alignment caveat.
* The Torch subprocess script exists, parses its ``--shapes-json``
  argument, emits a ``cublaslt_int8`` per-shape arm, and emits a
  well-formed stub when CUDA is missing.
* The ``--recompute-aggregate-only`` mode refreshes the aggregate
  block in-place from existing per-shape data (used to surface newly
  added IMMA fields without re-running CUDA), and does not invent
  numbers for arms that weren't actually measured.

Running the harness on the current (no-CUDA) host produces a stub
artifact with ``status="cuda_unavailable"``; these tests validate
*that* stub, plus they validate the schema contract that the live
artifact will satisfy when re-generated on a CUDA host.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from run_cublas_baseline import (  # noqa: E402
    DEFAULT_ITERS,
    DEFAULT_NUM_STABILITY_RUNS,
    DEFAULT_WARMUP,
    OUTPUT_JSON,
    SHAPES,
    SUBPROCESS_SCRIPT,
    TIMING_PROTOCOL_NAME,
    TIMING_STABILITY_THRESHOLD_PCT,
    _aggregate,
    _collect_arm_stability,
    _cublaslt_int8_per_column_median,
    _gap_pct,
    _methodology_block,
    recompute_aggregate_in_place,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBPROCESS_PATH = Path(SUBPROCESS_SCRIPT)


# ---------------------------------------------------------------------------
# Locked shape set
# ---------------------------------------------------------------------------


def test_locked_shape_set_is_exactly_six_shapes():
    assert len(SHAPES) == 6
    for entry in SHAPES:
        assert set(entry.keys()) == {"M", "K"}
        assert isinstance(entry["M"], int) and entry["M"] > 0
        assert isinstance(entry["K"], int) and entry["K"] > 0


def test_locked_shape_set_spans_small_medium_large_regions():
    sizes = [s["M"] * s["K"] for s in SHAPES]
    assert min(sizes) <= 16 * 16, "no small-regime shape in SHAPES"
    assert max(sizes) >= 512 * 256, "no large-regime shape in SHAPES"
    assert len({s["M"] * s["K"] for s in SHAPES}) >= 5, (
        "SHAPES has too many duplicates; a single number could be cherry-picked"
    )


def test_locked_shape_set_includes_anchor_shapes():
    pairs = {(s["M"], s["K"]) for s in SHAPES}
    assert (16, 16) in pairs
    assert (512, 512) in pairs


# ---------------------------------------------------------------------------
# Methodology block (independent of CUDA availability)
# ---------------------------------------------------------------------------


def test_methodology_block_contains_required_fields():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    required = {
        "shape_set",
        "shape_set_description",
        "warmup_iterations",
        "timed_iterations",
        "num_stability_runs",
        "timing_protocol_name",
        "timing_stability_threshold_pct",
        "timing_protocol",
        "stability_protocol",
        "tflops_definition",
        "baseline_arms",
        "cublaslt_int8_alignment_caveat",
        "dtype_caveats",
        "isolation",
        "rng_seed_per_shape",
        "scope",
    }
    assert required <= set(block.keys())
    assert block["shape_set"] == SHAPES
    assert block["warmup_iterations"] == DEFAULT_WARMUP
    assert block["timed_iterations"] == DEFAULT_ITERS
    assert block["num_stability_runs"] == DEFAULT_NUM_STABILITY_RUNS
    assert block["timing_protocol_name"] == TIMING_PROTOCOL_NAME
    assert (
        float(block["timing_stability_threshold_pct"])
        == TIMING_STABILITY_THRESHOLD_PCT
    )


def test_methodology_timing_protocol_documents_cuda_events_not_wall_clock():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    protocol = block["timing_protocol"].lower()
    assert "cuda event" in protocol or "torch.cuda.event" in protocol, (
        "timing_protocol must document CUDA-event timing (not wall-clock)"
    )
    assert "elapsed_time" in protocol, (
        "timing_protocol must reference start.elapsed_time(end)"
    )
    assert "perf_counter" not in protocol, (
        "timing_protocol must NOT claim wall-clock perf_counter as the active "
        "protocol — that was the old methodology"
    )


def test_methodology_stability_protocol_documents_n_runs_and_cv_gate():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    stab = block["stability_protocol"].lower()
    assert "back-to-back" in stab
    assert "per_run_medians" in stab or "per-run medians" in stab
    assert "timing_stability_pct" in stab or "cv" in stab, (
        "stability_protocol must name the CV / timing_stability_pct field"
    )
    assert (
        f"{TIMING_STABILITY_THRESHOLD_PCT}" in stab
        or f"{TIMING_STABILITY_THRESHOLD_PCT:.1f}" in stab
    ), "stability_protocol must document the CV threshold inline"


def test_methodology_dtype_caveats_call_out_inductor_and_cublas_dtypes():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    caveats_blob = " ".join(block["dtype_caveats"]).lower()
    assert "int8" in caveats_blob and "int32" in caveats_blob and "int4" in caveats_blob
    assert "fp32" in caveats_blob or "float32" in caveats_blob
    assert "inductor" in caveats_blob
    assert "cublas" in caveats_blob


def test_methodology_baseline_arms_lists_imma_int8_arm():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    arms_blob = " ".join(block["baseline_arms"]).lower()
    assert "cublaslt" in arms_blob, "baseline_arms must name the cuBLASLt IMMA INT8 arm"
    assert "imma" in arms_blob, "baseline_arms must name IMMA explicitly"
    assert "int8" in arms_blob and "int32" in arms_blob, (
        "baseline_arms must document the INT8-input / INT32-accumulator path"
    )
    assert "torch._int_mm" in arms_blob, (
        "baseline_arms must name the torch._int_mm entry point so the "
        "reader can audit which Python surface dispatches to IMMA"
    )


def test_methodology_imma_alignment_caveat_is_explicit_about_n_padding():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    caveat = block["cublaslt_int8_alignment_caveat"].lower()
    assert "n=1" in caveat or "n = 1" in caveat or "n=8" in caveat, (
        "alignment caveat must state the N=1 → N=8 padding explicitly"
    )
    assert "n_padded" in caveat or "ms_per_n_column" in caveat, (
        "alignment caveat must explain per-N-column derivation"
    )
    assert "favorable to cublas" in caveat or "favorable to cuBLAS".lower() in caveat, (
        "alignment caveat must explicitly state the per-column estimate "
        "is slightly favorable to cuBLAS (honesty disclosure)"
    )


def test_methodology_explicitly_disclaims_hardware_claim():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    assert "no physical-board claim" in block["scope"].lower() or \
           "sim/host-measured" in block["scope"].lower()


# ---------------------------------------------------------------------------
# Live artifact schema lock (works for both ok + cuda_unavailable modes)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> Dict[str, Any]:
    if not OUTPUT_JSON.exists():
        pytest.skip(f"baseline artifact not regenerated yet: {OUTPUT_JSON}")
    return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))


def test_artifact_has_top_level_contract_keys(artifact: Dict[str, Any]):
    required = {
        "version",
        "generated_at_utc",
        "git_sha",
        "methodology",
        "status",
        "shapes_requested",
        "per_shape",
        "aggregate",
    }
    assert required <= set(artifact.keys())
    assert artifact["version"] == 1
    assert artifact["status"] in {"ok", "cuda_unavailable"}


def test_artifact_shape_set_matches_locked_constant(artifact: Dict[str, Any]):
    assert artifact["shapes_requested"] == SHAPES
    assert artifact["methodology"]["shape_set"] == SHAPES


def test_artifact_methodology_dtype_caveats_present_in_either_mode(artifact: Dict[str, Any]):
    caveats = artifact["methodology"].get("dtype_caveats")
    assert isinstance(caveats, list) and len(caveats) >= 4, (
        "expected at least 4 dtype caveats: uTPU, cuBLAS, cuBLASLt IMMA INT8, Inductor"
    )
    caveats_blob = " ".join(caveats).lower()
    assert "cublaslt" in caveats_blob, (
        "dtype_caveats must call out the cuBLASLt IMMA INT8 arm; "
        "missing the dtype-matched apples-to-apples row makes the writeup "
        "drift from the schema again"
    )


def test_artifact_methodology_baseline_arms_locked(artifact: Dict[str, Any]):
    arms = artifact["methodology"].get("baseline_arms")
    assert isinstance(arms, list) and len(arms) >= 3, (
        "expected at least 3 baseline arms: cuBLAS GEMV, cuBLASLt IMMA INT8, Inductor"
    )
    blob = " ".join(arms).lower()
    assert "cublaslt" in blob and "imma" in blob and "int8" in blob


def test_cuda_unavailable_stub_has_actionable_instructions(artifact: Dict[str, Any]):
    if artifact["status"] != "cuda_unavailable":
        pytest.skip("artifact regenerated with status='ok'; stub gate not applicable")
    instr = artifact.get("instructions")
    assert isinstance(instr, list) and len(instr) >= 3
    blob = " ".join(instr).lower()
    assert "cuda" in blob
    assert "run_cublas_baseline.py" in blob


def test_ok_artifact_per_shape_layout_matches_lock(artifact: Dict[str, Any]):
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; live shape gate not applicable")
    per = artifact["per_shape"]
    assert len(per) == len(SHAPES)
    for entry in per:
        assert "shape" in entry and set(entry["shape"].keys()) == {"M", "K", "N"}
        assert "utpu" in entry
        assert "cublas" in entry
        assert "cublaslt_int8" in entry, (
            f"per_shape entry for shape={entry.get('shape')} missing "
            f"cublaslt_int8 arm; the dtype-matched IMMA INT8 GEMM arm "
            f"must be present (even if its value is None or has "
            f"imma_unavailable_reason)"
        )
        assert "inductor" in entry
        assert "gap_vs_cublas_pct_median" in entry
        assert "gap_vs_cublaslt_int8_pct_median" in entry, (
            f"per_shape entry for shape={entry.get('shape')} missing "
            f"gap_vs_cublaslt_int8_pct_median; the dtype-matched gap "
            f"field must be present (None is fine if IMMA was unavailable)"
        )
        assert "gap_vs_inductor_pct_median" in entry
        utpu = entry["utpu"]
        for key in [
            "backend", "dtype_W", "dtype_x", "dtype_accum", "dtype_out",
            "kernel_ms", "samples_ms", "int_mac_tflops_median",
            "bit_exact_match_vs_numpy_reference",
        ]:
            assert key in utpu, f"utpu entry missing field {key}"
        for stat in ["mean", "median", "stdev", "min", "max", "p95", "samples"]:
            assert stat in utpu["kernel_ms"], f"utpu.kernel_ms missing {stat}"


def test_ok_artifact_aggregate_gap_keys_match_lock(artifact: Dict[str, Any]):
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; aggregate gate not applicable")
    agg = artifact["aggregate"]
    assert agg is not None
    gap_keys = [
        "cublas_gap_pct_median_of_shapes",
        "cublas_gap_pct_mean_of_shapes",
        "cublas_gap_pct_max_of_shapes",
        "cublas_gap_pct_min_of_shapes",
        "cublaslt_int8_gap_pct_median_of_shapes",
        "cublaslt_int8_gap_pct_mean_of_shapes",
        "cublaslt_int8_gap_pct_max_of_shapes",
        "cublaslt_int8_gap_pct_min_of_shapes",
        "inductor_gap_pct_median_of_shapes",
        "inductor_gap_pct_mean_of_shapes",
        "shapes_compared_vs_cublas",
        "shapes_compared_vs_cublaslt_int8",
        "shapes_compared_vs_inductor",
    ]
    for key in gap_keys:
        assert key in agg, f"aggregate missing field {key}"

    # Stability rollup keys are required only on stabilized artifacts
    # (i.e., artifacts regenerated after the timing-stability protocol
    # landed). Legacy populated artifacts pre-date the rollup and the
    # gate test below will skip them with an actionable regen hint.
    if _is_stabilized_artifact(artifact):
        for key in [
            "max_timing_stability_pct_across_arms",
            "mean_timing_stability_pct_across_arms",
            "arms_exceeding_stability_threshold_pct",
            "num_stability_measurements",
            "timing_stability_threshold_pct",
        ]:
            assert key in agg, (
                f"stabilized artifact aggregate missing stability field "
                f"{key!r}; once methodology.timing_protocol_name is set "
                f"all stability rollup fields must be present"
            )


# ---------------------------------------------------------------------------
# Inter-run timing stability — load-bearing CV gate
# ---------------------------------------------------------------------------


def _is_stabilized_artifact(artifact: Dict[str, Any]) -> bool:
    """A populated artifact counts as 'stabilized' if its methodology
    names the new timing protocol AND its aggregate carries the
    timing-stability rollup. Legacy populated artifacts (pre-this-change)
    have neither; we treat them as not-yet-stabilized and skip the gate."""
    methodology = artifact.get("methodology") or {}
    if methodology.get("timing_protocol_name") != TIMING_PROTOCOL_NAME:
        return False
    agg = artifact.get("aggregate") or {}
    return "max_timing_stability_pct_across_arms" in agg


def _is_split_gate_artifact(artifact: Dict[str, Any]) -> bool:
    """Split-gate = stabilized + the strict-side load-bearing-arm field
    exists. Artifacts generated before the split-CV-gate change carry the
    old single across-arms field only; gate tests asserting on the strict
    field cleanly skip those with an actionable regen hint."""
    if not _is_stabilized_artifact(artifact):
        return False
    agg = artifact.get("aggregate") or {}
    return "max_timing_stability_pct_load_bearing_arm" in agg


def test_ok_artifact_populated_arm_kernel_ms_carries_stability_fields(
    artifact: Dict[str, Any],
):
    """Every populated (status='ok') per-arm-per-shape kernel_ms block in
    a stabilized artifact must report the inter-run stability fields:

    * ``per_run_medians_ms`` (list of N >= 1 floats)
    * ``timing_stability_pct`` (CV across per-run medians)
    * ``num_stability_runs`` (int >= 1)
    * ``timing_protocol`` (matches TIMING_PROTOCOL_NAME)

    On legacy artifacts that predate this protocol the fields may be
    absent — the test skips on those, but a stabilized artifact must
    have them populated everywhere or the gate is meaningless.
    """
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; stability fields not applicable")
    if not _is_stabilized_artifact(artifact):
        pytest.skip(
            "artifact predates timing-stability protocol; "
            "re-run firmware/host/run_cublas_baseline.py on CUDA to regen"
        )
    arm_names = ("utpu", "cublas", "cublaslt_int8", "inductor")
    for entry in artifact["per_shape"]:
        for arm_name in arm_names:
            arm = entry.get(arm_name)
            if not isinstance(arm, dict):
                continue
            if "imma_unavailable_reason" in arm or "error" in arm:
                continue
            kernel_ms = arm.get("kernel_ms")
            if not isinstance(kernel_ms, dict):
                continue
            for field in (
                "per_run_medians_ms",
                "per_run_medians_trimmed_ms",
                "timing_stability_pct",
                "num_stability_runs",
                "num_stability_runs_trimmed",
                "timing_protocol",
            ):
                assert field in kernel_ms, (
                    f"shape={entry['shape']} arm={arm_name}: kernel_ms "
                    f"missing required stability field {field!r}; the "
                    f"timing-stability protocol must populate this on "
                    f"every populated arm or the CV gate is meaningless"
                )
            assert isinstance(kernel_ms["per_run_medians_ms"], list)
            assert len(kernel_ms["per_run_medians_ms"]) == int(
                kernel_ms["num_stability_runs"]
            ), (
                f"shape={entry['shape']} arm={arm_name}: "
                f"per_run_medians_ms length "
                f"({len(kernel_ms['per_run_medians_ms'])}) does not match "
                f"num_stability_runs ({kernel_ms['num_stability_runs']})"
            )
            assert isinstance(kernel_ms["per_run_medians_trimmed_ms"], list)
            assert len(kernel_ms["per_run_medians_trimmed_ms"]) == int(
                kernel_ms["num_stability_runs_trimmed"]
            ), (
                f"shape={entry['shape']} arm={arm_name}: "
                f"per_run_medians_trimmed_ms length "
                f"({len(kernel_ms['per_run_medians_trimmed_ms'])}) does "
                f"not match num_stability_runs_trimmed "
                f"({kernel_ms['num_stability_runs_trimmed']})"
            )
            assert len(kernel_ms["per_run_medians_trimmed_ms"]) <= len(
                kernel_ms["per_run_medians_ms"]
            ), (
                f"shape={entry['shape']} arm={arm_name}: trimmed list "
                f"cannot be longer than the full per-run-medians list"
            )
            assert kernel_ms["timing_protocol"] == TIMING_PROTOCOL_NAME


def test_load_bearing_utpu_arm_is_gate_green(artifact: Dict[str, Any]):
    """STRICT CV gate (split-gate engineering): asserts ONLY on the
    uTPU (load-bearing) arm.

    On a split-gate artifact the aggregate exposes
    ``max_timing_stability_pct_load_bearing_arm`` (worst inter-run CV
    across only the uTPU arm — every (utpu, shape) pair). The
    documented threshold lives at ``timing_stability_threshold_pct``;
    if the gate fires the build fails.

    CRITICAL: passing this strict gate does **NOT** certify any
    specific ``gap_vs_*_pct`` number (IMMA-INT8 / cuBLAS-FP / Inductor
    gaps). Every gap's denominator is on a **baseline** arm and the
    baseline arms are intentionally gated **advisory** only —
    multi-launch / multi-kernel-callee host-side jitter on a contended
    laptop GPU without ``nvidia-smi --lock-gpu-clocks`` is intrinsic,
    not methodology-fixable. Any specific %-claim therefore stays
    ``[needs-locked-clock-artifact]`` until regenerated on locked
    clocks (native Linux
    ``sudo nvidia-smi --lock-gpu-clocks=$BOOST,$BOOST``) or a cloud
    RTX. The strict gate's job is narrower: confirm uTPU's
    per-iteration kernel time is *itself* stable.
    """
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; CV gate not applicable")
    if not _is_split_gate_artifact(artifact):
        pytest.skip(
            "artifact predates the split-CV-gate (load-bearing arm) "
            "protocol; re-run firmware/host/run_cublas_baseline.py on "
            "CUDA to regen with the strict + advisory split fields."
        )
    agg = artifact["aggregate"]
    threshold = float(agg["timing_stability_threshold_pct"])
    max_cv_lb = agg["max_timing_stability_pct_load_bearing_arm"]
    if max_cv_lb is None:
        pytest.skip(
            "no populated load-bearing-arm stability measurements "
            "(uTPU errored on every shape)"
        )
    lb_exceed = agg.get("load_bearing_arms_exceeding_threshold_pct") or []
    assert max_cv_lb <= threshold, (
        f"STRICT timing-stability gate FAILED on the load-bearing "
        f"(uTPU) arm: max inter-run CV = {max_cv_lb:.2f}% > documented "
        f"threshold {threshold:.1f}%. Offending (arm, shape, CV%, "
        f"per-run-medians-ms): "
        f"{[(e['arm'], e['shape'], e['timing_stability_pct'], e['per_run_medians_ms']) for e in lb_exceed]}. "
        f"Remediation: bump --warmup / --iters / --num-stability-runs "
        f"in firmware/host/run_cublas_baseline.py and regen, or kill "
        f"background GPU load."
    )
    assert not lb_exceed, (
        f"max load-bearing-arm CV under threshold but exceed list "
        f"non-empty: {lb_exceed!r}"
    )


def test_baseline_arms_advisory_cv_reporting(artifact: Dict[str, Any]):
    """ADVISORY CV companion (split-gate engineering): does NOT fail the
    build, but asserts the aggregate carries the baseline-arm CV
    diagnostic fields so a reviewer can see the noise floor.

    The baseline arms (cublas, cublaslt_int8, inductor) are multi-
    launch / multi-kernel-callee torch wrappers; on a contended laptop
    GPU without ``nvidia-smi --lock-gpu-clocks`` their inter-run CV is
    dominated by clock-frequency drift between stability runs —
    intrinsic to the host environment, not methodology-fixable. This
    test reports the advisory CV and offending pairs (if any) and skips
    assertion. The IMMA-INT8 / cuBLAS-FP / Inductor gap %-claims
    therefore stay ``[needs-locked-clock-artifact]`` even when the
    strict gate above passes — passing the split gate does NOT certify
    any specific gap-vs-* number.
    """
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; CV gate not applicable")
    if not _is_split_gate_artifact(artifact):
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
        # artifact for reviewers; gap_vs_* %-claims are independently
        # [needs-locked-clock-artifact] regardless of this value.
        print(
            f"\n[ADVISORY] baseline-arm CV {max_cv_bl:.2f}% > "
            f"{threshold:.1f}% on {len(bl_exceed)} (arm, shape) "
            f"pair(s) — expected on WSL2 + unlocked laptop clocks; "
            f"does NOT block build, does NOT certify any gap_vs_* "
            f"%-claim."
        )


def test_collect_arm_stability_helper_handles_empty_and_legacy_inputs():
    """Unit test the rollup helper against three input classes:

    1. Stabilized populated input → max + mean + count populated, threshold field set.
    2. Legacy input with no stability fields → all rollup values None, count=0.
    3. Mixed input (one arm has stability, others don't) → only the populated CV counts.
    """
    stabilized = [
        {
            "shape": {"M": 16, "K": 16, "N": 1},
            "utpu": {"kernel_ms": {"timing_stability_pct": 2.5}},
            "cublas": {"kernel_ms": {"timing_stability_pct": 4.0}},
            "cublaslt_int8": {"kernel_ms": {"timing_stability_pct": 3.0}},
            "inductor": {"kernel_ms": {"timing_stability_pct": 12.0,
                                       "per_run_medians_ms": [0.1, 0.13, 0.11]}},
        },
        {
            "shape": {"M": 64, "K": 64, "N": 1},
            "utpu": {"kernel_ms": {"timing_stability_pct": 1.5}},
            "cublas": {"kernel_ms": {"timing_stability_pct": 2.0}},
            "cublaslt_int8": {"imma_unavailable_reason": "M <= 16"},
        },
    ]
    roll = _collect_arm_stability(stabilized)
    # Across-arms diagnostic still reports the full picture.
    assert roll["num_stability_measurements"] == 6
    assert roll["max_timing_stability_pct_across_arms"] == 12.0
    assert roll["mean_timing_stability_pct_across_arms"] == pytest.approx(
        (2.5 + 4.0 + 3.0 + 12.0 + 1.5 + 2.0) / 6.0
    )
    assert len(roll["arms_exceeding_stability_threshold_pct"]) == 1
    bad = roll["arms_exceeding_stability_threshold_pct"][0]
    assert bad["arm"] == "inductor"
    assert bad["timing_stability_pct"] == 12.0
    assert bad["per_run_medians_ms"] == [0.1, 0.13, 0.11]
    # Split-gate: load-bearing arm (utpu) has max CV = max(2.5, 1.5) = 2.5.
    assert roll["max_timing_stability_pct_load_bearing_arm"] == 2.5
    assert roll["num_stability_measurements_load_bearing_arm"] == 2
    assert roll["load_bearing_arms_exceeding_threshold_pct"] == []
    # Baseline arms have max CV = 12.0 (inductor exceeds threshold).
    assert roll["max_timing_stability_pct_baseline_arms"] == 12.0
    assert roll["num_stability_measurements_baseline_arms"] == 4
    assert len(roll["baseline_arms_exceeding_threshold_pct"]) == 1
    assert roll["baseline_arms_exceeding_threshold_pct"][0]["arm"] == "inductor"
    assert roll["load_bearing_arms"] == ["utpu"]

    legacy = [
        {
            "shape": {"M": 16, "K": 16, "N": 1},
            "utpu": {"kernel_ms": {"median": 0.05}},
            "cublas": {"kernel_ms": {"median": 0.04}},
        },
    ]
    roll_legacy = _collect_arm_stability(legacy)
    assert roll_legacy["num_stability_measurements"] == 0
    assert roll_legacy["max_timing_stability_pct_across_arms"] is None
    assert roll_legacy["mean_timing_stability_pct_across_arms"] is None
    assert roll_legacy["arms_exceeding_stability_threshold_pct"] == []
    # Split-gate fields are legacy-null too:
    assert roll_legacy["max_timing_stability_pct_load_bearing_arm"] is None
    assert roll_legacy["load_bearing_arms_exceeding_threshold_pct"] == []
    assert roll_legacy["num_stability_measurements_load_bearing_arm"] == 0
    assert roll_legacy["max_timing_stability_pct_baseline_arms"] is None
    assert roll_legacy["baseline_arms_exceeding_threshold_pct"] == []
    assert roll_legacy["num_stability_measurements_baseline_arms"] == 0
    assert roll_legacy["timing_stability_threshold_pct"] == TIMING_STABILITY_THRESHOLD_PCT

    mixed = [
        {
            "shape": {"M": 16, "K": 16, "N": 1},
            "utpu": {"kernel_ms": {"timing_stability_pct": 5.0}},
            "cublas": {"kernel_ms": {"median": 0.04}},
        },
    ]
    roll_mixed = _collect_arm_stability(mixed)
    assert roll_mixed["num_stability_measurements"] == 1
    assert roll_mixed["max_timing_stability_pct_across_arms"] == 5.0
    assert roll_mixed["max_timing_stability_pct_load_bearing_arm"] == 5.0
    assert roll_mixed["num_stability_measurements_load_bearing_arm"] == 1
    assert roll_mixed["max_timing_stability_pct_baseline_arms"] is None
    assert roll_mixed["num_stability_measurements_baseline_arms"] == 0
    assert roll_mixed["arms_exceeding_stability_threshold_pct"] == []


def test_ok_artifact_cublaslt_int8_arm_is_either_populated_or_explicitly_unavailable(
    artifact: Dict[str, Any],
):
    """For each shape in the live artifact, the ``cublaslt_int8`` entry
    must either:

    * Be populated with a measured kernel_ms + ms_per_n_column_median +
      INT8/INT32 dtype tags + an alignment_note (the IMMA arm actually
      ran), OR
    * Be ``None`` (legacy artifact pre-IMMA-arm; will be filled on next
      CUDA regen), OR
    * Be a stub entry with ``imma_unavailable_reason`` or ``error``
      explaining exactly why IMMA could not be measured (the IMMA arm
      ran but cuBLASLt rejected the call).

    Silent IMMA misses (entry present, no timings, no failure reason)
    are not allowed — those would let a downstream reader assume IMMA
    "passed" without measurement.
    """
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; live IMMA gate not applicable")
    for entry in artifact["per_shape"]:
        arm = entry.get("cublaslt_int8")
        shape = entry.get("shape")
        if arm is None:
            continue
        if "imma_unavailable_reason" in arm:
            assert isinstance(arm["imma_unavailable_reason"], str) and len(
                arm["imma_unavailable_reason"]
            ) > 0
            assert arm.get("backend") == "cublaslt_imma_int8"
            continue
        if "error" in arm:
            assert isinstance(arm["error"], str) and len(arm["error"]) > 0
            continue
        assert arm.get("backend") == "cublaslt_imma_int8", (
            f"shape {shape}: cublaslt_int8 arm with backend tag "
            f"{arm.get('backend')!r} (expected 'cublaslt_imma_int8')"
        )
        for key in [
            "dtype_W", "dtype_x", "dtype_accum", "dtype_out",
            "alignment_note", "kernel_ms", "ms_per_n_column_median",
            "ms_per_n_column_mean", "samples_ms",
        ]:
            assert key in arm, (
                f"shape {shape}: populated cublaslt_int8 arm missing field {key!r}"
            )
        assert arm["dtype_W"] == "int8"
        assert arm["dtype_x"] == "int8"
        assert arm["dtype_accum"] == "int32"
        assert arm["dtype_out"] == "int32", (
            f"shape {shape}: cublaslt_int8 dtype_out={arm['dtype_out']!r} "
            f"(expected 'int32'); the IMMA arm must use INT32 accumulator "
            f"output to be dtype-matched to the uTPU INT32 accumulator"
        )
        shape_block = arm.get("shape", {})
        assert int(shape_block.get("N", 1)) == 1, (
            f"shape {shape}: cublaslt_int8.shape.N must be 1 (uTPU GEMV); "
            f"got {shape_block.get('N')!r}"
        )
        n_padded = int(shape_block.get("N_padded", 0))
        assert n_padded >= 8 and n_padded % 8 == 0, (
            f"shape {shape}: cublaslt_int8.shape.N_padded={n_padded} "
            f"violates IMMA alignment (must be positive multiple of 8)"
        )
        per_col = arm["ms_per_n_column_median"]
        full_median = arm["kernel_ms"]["median"]
        assert per_col > 0.0 and full_median > 0.0
        assert abs(per_col * n_padded - full_median) < 1e-9, (
            f"shape {shape}: ms_per_n_column_median * N_padded "
            f"({per_col} * {n_padded} = {per_col*n_padded}) does not "
            f"equal kernel_ms.median ({full_median}); the per-column "
            f"derivation must be exact (no rounding, no fabrication)"
        )


def test_ok_artifact_cublaslt_int8_gap_uses_per_column_not_full_n(
    artifact: Dict[str, Any],
):
    """If the IMMA arm ran for a shape, ``gap_vs_cublaslt_int8_pct_median``
    must be derived from ``ms_per_n_column_median`` (the GEMV-equivalent
    figure), not from ``kernel_ms.median`` (the full N=N_padded wall
    time). Mixing the two would inflate the gap by ``N_padded``×.
    """
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; live IMMA gap derivation gate not applicable")
    for entry in artifact["per_shape"]:
        arm = entry.get("cublaslt_int8")
        if not arm or "imma_unavailable_reason" in arm or "error" in arm:
            continue
        gap_pct = entry.get("gap_vs_cublaslt_int8_pct_median")
        if gap_pct is None:
            continue
        utpu_median = entry["utpu"]["kernel_ms"]["median"]
        per_col = arm["ms_per_n_column_median"]
        expected_gap = (utpu_median - per_col) / per_col * 100.0
        assert abs(gap_pct - expected_gap) < 1e-6, (
            f"shape {entry['shape']}: gap_vs_cublaslt_int8_pct_median "
            f"({gap_pct}) does not match the per-column derivation "
            f"({expected_gap}); the gap must use ms_per_n_column_median, "
            f"not the full-N wall time"
        )


def test_ok_artifact_aggregate_imma_counts_consistent_with_per_shape(
    artifact: Dict[str, Any],
):
    """``shapes_compared_vs_cublaslt_int8`` must equal the number of
    per-shape entries that successfully populated
    ``gap_vs_cublaslt_int8_pct_median`` — the count cannot be inflated
    or quietly trimmed."""
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; aggregate-vs-per-shape gate not applicable")
    agg = artifact["aggregate"]
    counted = sum(
        1
        for e in artifact["per_shape"]
        if e.get("gap_vs_cublaslt_int8_pct_median") is not None
    )
    assert agg.get("shapes_compared_vs_cublaslt_int8") == counted, (
        f"shapes_compared_vs_cublaslt_int8="
        f"{agg.get('shapes_compared_vs_cublaslt_int8')} != "
        f"actual non-null per-shape count={counted}"
    )
    if counted == 0:
        for key in [
            "cublaslt_int8_gap_pct_median_of_shapes",
            "cublaslt_int8_gap_pct_mean_of_shapes",
            "cublaslt_int8_gap_pct_max_of_shapes",
            "cublaslt_int8_gap_pct_min_of_shapes",
        ]:
            assert agg.get(key) is None, (
                f"aggregate field {key} is non-null but no per-shape "
                f"IMMA gap was populated; refusing fabricated number"
            )
    else:
        for key in [
            "cublaslt_int8_gap_pct_median_of_shapes",
            "cublaslt_int8_gap_pct_mean_of_shapes",
            "cublaslt_int8_gap_pct_max_of_shapes",
            "cublaslt_int8_gap_pct_min_of_shapes",
        ]:
            assert agg.get(key) is not None, (
                f"aggregate field {key} is null but {counted} per-shape "
                f"IMMA gaps were populated; aggregate must be derivable"
            )


def test_ok_artifact_records_bit_exact_match_per_shape(artifact: Dict[str, Any]):
    """On a real GPU run the uTPU kernel must remain bit-exact vs the
    Python NumPy reference. We do not gate the cuBLAS / Inductor
    numerical agreement here (different dtypes), only our own kernel."""
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; bit-exact gate not applicable")
    for entry in artifact["per_shape"]:
        assert entry["utpu"]["bit_exact_match_vs_numpy_reference"] is True, (
            f"uTPU kernel not bit-exact vs NumPy reference for shape {entry['shape']}"
        )


def test_ok_artifact_cublas_dtype_is_explicit_no_silent_fallback(artifact: Dict[str, Any]):
    """If the Torch build cannot run INT32 cuBLAS GEMV, the subprocess
    falls back to FP32 — but the per-shape entry must record the
    fallback (``dtype_fallback_reason`` non-empty + ``backend`` named
    ``cublas_gemv_fp32_fallback`` + ``dtype_W=="float32"``) so a reader
    cannot mistake the gap for an apples-to-apples INT32 comparison.
    """
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; live dtype contract not applicable")
    for entry in artifact["per_shape"]:
        cublas = entry.get("cublas")
        if cublas is None or "error" in cublas:
            continue
        dtype_W = cublas.get("dtype_W")
        if dtype_W == "int32":
            assert cublas.get("backend") == "cublas_gemv_int32"
            assert cublas.get("dtype_accum") == "int32"
            assert "dtype_fallback_reason" not in cublas
        elif dtype_W == "float32":
            assert cublas.get("backend") == "cublas_gemv_fp32_fallback", (
                f"shape {entry['shape']}: fp32 dtype without fp32_fallback "
                f"backend tag — this would silently downgrade the dtype "
                f"comparison; got backend={cublas.get('backend')!r}"
            )
            assert cublas.get("dtype_accum") == "float32"
            reason = cublas.get("dtype_fallback_reason")
            assert isinstance(reason, str) and len(reason) > 0, (
                f"shape {entry['shape']}: fp32 fallback path missing "
                f"dtype_fallback_reason — refusing silent dtype switch."
            )
        else:
            pytest.fail(
                f"shape {entry['shape']}: cublas dtype_W={dtype_W!r} is "
                f"neither int32 (locked dtype) nor float32 (documented "
                f"fallback). Unknown dtype path is not allowed."
            )


# ---------------------------------------------------------------------------
# Torch subprocess (locked entry point)
# ---------------------------------------------------------------------------


def test_torch_subprocess_script_exists_and_is_python():
    assert SUBPROCESS_PATH.exists(), f"missing {SUBPROCESS_PATH}"
    text = SUBPROCESS_PATH.read_text(encoding="utf-8")
    assert "def run_baselines" in text
    assert "--shapes-json" in text


def test_torch_subprocess_defines_cublaslt_int8_imma_arm():
    """The subprocess must expose a cuBLASLt IMMA INT8 GEMM arm. If this
    test ever drifts, the parent runner will silently fall back to a
    two-arm (cuBLAS-fallback + Inductor) artifact and the headline
    "dtype-matched cuBLAS comparison" claim becomes vacuous."""
    text = SUBPROCESS_PATH.read_text(encoding="utf-8")
    assert "_run_cublaslt_imma_int8_gemm" in text, (
        "_run_cublaslt_imma_int8_gemm function missing from subprocess; "
        "without it the cuBLASLt IMMA INT8 arm cannot be emitted"
    )
    assert "torch._int_mm" in text, (
        "torch._int_mm dispatch missing from subprocess; cuBLASLt IMMA "
        "INT8 GEMM cannot be measured via Torch without it"
    )
    assert "cublaslt_imma_int8" in text, (
        "cuBLASLt IMMA backend tag missing from subprocess"
    )
    assert "N_padded" in text, (
        "N_padded alignment metadata missing from subprocess; the IMMA "
        "N=1→N=8 alignment caveat must be recorded per-shape"
    )


def test_torch_subprocess_imma_arm_refuses_misaligned_shapes_without_running():
    """Smoke-test the IMMA pre-flight checks in isolation: if the
    caller asks for a shape that violates IMMA alignment (K % 16 != 0
    or N_padded % 8 != 0), the subprocess must return an
    ``imma_unavailable_reason`` entry rather than dispatching the
    matmul. This locks the "no silent IMMA misses" contract.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not importable in this environment")
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA unavailable; alignment-rejection smoke test requires the "
            "subprocess to actually reach the IMMA preflight (which is "
            "behind the CUDA gate inside the subprocess wrapper)"
        )

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_cublas_baseline_torch_subprocess", str(SUBPROCESS_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    bad_k = mod._run_cublaslt_imma_int8_gemm(torch, M=32, K=15, warmup=1, iters=1)
    assert bad_k.get("imma_unavailable_reason"), (
        "subprocess accepted K=15 (not a multiple of 16) without "
        "flagging IMMA misalignment"
    )
    bad_n = mod._run_cublaslt_imma_int8_gemm(
        torch, M=32, K=16, warmup=1, iters=1, n_pad=7
    )
    assert bad_n.get("imma_unavailable_reason"), (
        "subprocess accepted N_padded=7 (not a multiple of 8) without "
        "flagging IMMA misalignment"
    )
    bad_m = mod._run_cublaslt_imma_int8_gemm(torch, M=16, K=16, warmup=1, iters=1)
    assert bad_m.get("imma_unavailable_reason"), (
        "subprocess accepted M=16 without flagging the torch._int_mm "
        "M > 16 wrapper constraint; without this preflight the "
        "smallest shape in SHAPES dispatches into the runtime check "
        "and surfaces as a noisy `error` field instead of a clean "
        "imma_unavailable_reason"
    )
    reason = bad_m["imma_unavailable_reason"].lower()
    assert "m=16" in reason or "m<=16" in reason or "m <= 16" in reason
    assert "torch._int_mm" in bad_m["imma_unavailable_reason"], (
        "M-preflight reason must name torch._int_mm so the reader knows "
        "this is a PyTorch wrapper constraint, not a hardware IMMA limit"
    )

    # Boundary above the wrapper limit must NOT be rejected by the
    # M-preflight (M=17 satisfies torch._int_mm; whether it actually
    # runs is a separate hardware question gated by the live dispatch
    # below). We only assert preflight didn't reject at the M boundary.
    edge_m = mod._run_cublaslt_imma_int8_gemm(torch, M=17, K=16, warmup=1, iters=1)
    if "imma_unavailable_reason" in edge_m:
        reason_edge = edge_m["imma_unavailable_reason"].lower()
        assert "m=17" not in reason_edge and "m <= 16" not in reason_edge, (
            f"M=17 must not be rejected by the M-preflight; got reason: "
            f"{edge_m['imma_unavailable_reason']!r}"
        )


def test_cublaslt_int8_per_column_median_helper_rejects_unavailable_and_error_paths():
    assert _cublaslt_int8_per_column_median(None) is None
    assert _cublaslt_int8_per_column_median({}) is None
    assert _cublaslt_int8_per_column_median(
        {"imma_unavailable_reason": "GPU SM does not support IMMA"}
    ) is None
    assert _cublaslt_int8_per_column_median({"error": "subprocess crashed"}) is None
    assert _cublaslt_int8_per_column_median({"ms_per_n_column_median": 0.0}) is None
    assert _cublaslt_int8_per_column_median({"ms_per_n_column_median": -1.0}) is None
    assert _cublaslt_int8_per_column_median({"ms_per_n_column_median": "0.5"}) == 0.5
    assert _cublaslt_int8_per_column_median({"ms_per_n_column_median": 0.5}) == 0.5


def test_gap_pct_helper_rejects_nonpositive_or_missing_inputs():
    assert _gap_pct(0.5, None) is None
    assert _gap_pct(0.5, 0.0) is None
    assert _gap_pct(0.5, -1.0) is None
    assert _gap_pct(0.0, 0.5) is None
    assert _gap_pct(0.5, 0.5) == 0.0
    assert abs(_gap_pct(0.6, 0.5) - 20.0) < 1e-9


def test_aggregate_correctly_handles_mixed_populated_and_missing_imma_arms():
    per_shape = [
        {
            "shape": {"M": 16, "K": 16, "N": 1},
            "gap_vs_cublas_pct_median": 10.0,
            "gap_vs_cublaslt_int8_pct_median": 5.0,
            "gap_vs_inductor_pct_median": -20.0,
        },
        {
            "shape": {"M": 64, "K": 64, "N": 1},
            "gap_vs_cublas_pct_median": 30.0,
            "gap_vs_cublaslt_int8_pct_median": None,
            "gap_vs_inductor_pct_median": -40.0,
        },
        {
            "shape": {"M": 128, "K": 128, "N": 1},
            "gap_vs_cublas_pct_median": 50.0,
            "gap_vs_cublaslt_int8_pct_median": 25.0,
            "gap_vs_inductor_pct_median": None,
        },
    ]
    agg = _aggregate(per_shape)
    assert agg["shapes_compared_vs_cublas"] == 3
    assert agg["shapes_compared_vs_cublaslt_int8"] == 2
    assert agg["shapes_compared_vs_inductor"] == 2
    assert agg["cublaslt_int8_gap_pct_median_of_shapes"] == 15.0
    assert agg["cublaslt_int8_gap_pct_mean_of_shapes"] == 15.0
    assert agg["cublaslt_int8_gap_pct_max_of_shapes"] == 25.0
    assert agg["cublaslt_int8_gap_pct_min_of_shapes"] == 5.0


def test_aggregate_collapses_to_none_when_no_imma_shapes_populated():
    per_shape = [
        {
            "shape": {"M": 16, "K": 16, "N": 1},
            "gap_vs_cublas_pct_median": 10.0,
            "gap_vs_cublaslt_int8_pct_median": None,
            "gap_vs_inductor_pct_median": -20.0,
        },
    ]
    agg = _aggregate(per_shape)
    assert agg["shapes_compared_vs_cublaslt_int8"] == 0
    for key in [
        "cublaslt_int8_gap_pct_median_of_shapes",
        "cublaslt_int8_gap_pct_mean_of_shapes",
        "cublaslt_int8_gap_pct_max_of_shapes",
        "cublaslt_int8_gap_pct_min_of_shapes",
    ]:
        assert agg[key] is None, (
            f"aggregate field {key} must be None when no IMMA shapes "
            f"were populated (no fabrication)"
        )


def test_recompute_aggregate_in_place_refreshes_methodology_without_inventing_imma_data(
    tmp_path,
):
    """``--recompute-aggregate-only`` must:

    1. Refresh ``methodology`` (so newly-added fields like
       ``cublaslt_int8_alignment_caveat`` land on legacy artifacts).
    2. Refresh ``aggregate`` from existing per-shape data.
    3. Record ``aggregate_recomputed`` with a non-empty reason.
    4. NOT fabricate IMMA timings for legacy artifacts whose per-shape
       entries lack a ``cublaslt_int8`` arm.
    """
    legacy = {
        "version": 1,
        "generated_at_utc": "2026-05-01T00:00:00+00:00",
        "git_sha": "legacy-sha",
        "methodology": {
            "shape_set": SHAPES,
            "warmup_iterations": DEFAULT_WARMUP,
            "timed_iterations": DEFAULT_ITERS,
        },
        "status": "ok",
        "environment": {"legacy": True},
        "shapes_requested": SHAPES,
        "per_shape": [
            {
                "shape": {"M": 16, "K": 16, "N": 1},
                "utpu": {
                    "backend": "utpu_blocked_fc_nvrtc_int4",
                    "dtype_W": "int8",
                    "dtype_x": "int8",
                    "dtype_accum": "int32",
                    "dtype_out": "int4_quantized",
                    "kernel_ms": {
                        "mean": 0.05, "median": 0.05, "stdev": 0.001,
                        "min": 0.04, "max": 0.06, "p95": 0.058, "samples": 50,
                    },
                    "samples_ms": [0.05] * 10,
                    "int_mac_tflops_median": 0.01,
                    "bit_exact_match_vs_numpy_reference": True,
                },
                "cublas": {
                    "backend": "cublas_gemv_fp32_fallback",
                    "kernel_ms": {"median": 0.04},
                },
                "inductor": {
                    "backend": "inductor_linear_fp32",
                    "kernel_ms": {"median": 0.08},
                },
                "gap_vs_cublas_pct_median": 25.0,
                "gap_vs_inductor_pct_median": -37.5,
            },
        ],
        "aggregate": {
            "cublas_gap_pct_median_of_shapes": 25.0,
            "cublas_gap_pct_mean_of_shapes": 25.0,
            "cublas_gap_pct_max_of_shapes": 25.0,
            "cublas_gap_pct_min_of_shapes": 25.0,
            "inductor_gap_pct_median_of_shapes": -37.5,
            "inductor_gap_pct_mean_of_shapes": -37.5,
            "shapes_compared_vs_cublas": 1,
            "shapes_compared_vs_inductor": 1,
        },
        "torch_subprocess": None,
    }
    out = tmp_path / "legacy_cublas_baseline.json"
    out.write_text(json.dumps(legacy), encoding="utf-8")
    rc = recompute_aggregate_in_place(out)
    assert rc == 0
    refreshed = json.loads(out.read_text(encoding="utf-8"))

    methodology = refreshed["methodology"]
    assert "cublaslt_int8_alignment_caveat" in methodology
    assert "baseline_arms" in methodology

    agg = refreshed["aggregate"]
    assert agg["shapes_compared_vs_cublas"] == 1
    assert agg["shapes_compared_vs_cublaslt_int8"] == 0
    assert agg["cublaslt_int8_gap_pct_median_of_shapes"] is None
    assert agg["cublaslt_int8_gap_pct_mean_of_shapes"] is None

    entry = refreshed["per_shape"][0]
    assert entry["gap_vs_cublaslt_int8_pct_median"] is None, (
        "legacy artifact without cublaslt_int8 per-shape entry must "
        "produce a None IMMA gap — no fabrication"
    )

    recomputed = refreshed.get("aggregate_recomputed")
    assert isinstance(recomputed, dict)
    assert isinstance(recomputed.get("reason"), str) and len(recomputed["reason"]) > 0
    assert isinstance(recomputed.get("recomputed_at_utc"), str)


def test_recompute_aggregate_in_place_refuses_on_stub_artifact(tmp_path):
    stub = {
        "version": 1,
        "status": "cuda_unavailable",
        "methodology": {"shape_set": SHAPES, "warmup_iterations": 10, "timed_iterations": 50},
        "per_shape": [],
        "aggregate": None,
    }
    out = tmp_path / "stub_cublas_baseline.json"
    out.write_text(json.dumps(stub), encoding="utf-8")
    rc = recompute_aggregate_in_place(out)
    assert rc == 1, (
        "recompute-aggregate-only on a status='cuda_unavailable' stub "
        "must refuse with a non-zero return; re-running CUDA is the "
        "only valid path to populate the IMMA arm"
    )
    # File should not have been rewritten meaningfully (stub still stub).
    after = json.loads(out.read_text(encoding="utf-8"))
    assert after.get("status") == "cuda_unavailable"


def test_torch_subprocess_emits_well_formed_stub_when_cuda_missing():
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not importable in this environment")
    import torch as torch_mod
    if torch_mod.cuda.is_available():
        pytest.skip("CUDA available; stub path not exercisable")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(SUBPROCESS_PATH),
                "--shapes-json",
                json.dumps([{"M": 16, "K": 16}]),
                "--output",
                str(tmp_path),
                "--warmup",
                "1",
                "--iters",
                "1",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, f"subprocess failed: {proc.stderr}"
        payload = json.loads(tmp_path.read_text(encoding="utf-8"))
        assert payload["status"] in {"cuda_unavailable", "torch_unavailable"}
        assert payload.get("shapes_requested") == [{"M": 16, "K": 16}]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
