"""Phase 3 tests for the tiling controller.

Coverage:

* Pure tile-math (no simulator): BufferCapacityModel + plan_linear_tiling.
* Differential correctness via ISA simulator vs NumPy oracle on
  representative over-capacity workloads.
* Confirmation that the un-tiled lowering actually fails on the
  over-capacity workload (so the test is solving a real problem).
* cost_model_select policy plumbing (with injected selector to keep the
  test independent of the Phase 1 calibration artifact).
* Schema-lock + correctness gate on `bench/results/tiling_correctness.json`.

This file does NOT gate on a specific cycle count (host-dependent).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from tiling_controller import (
    BufferCapacityModel,
    TilingError,
    TilingPlan,
    _decode_tile_outputs_from_fetch_bytes,
    _estimate_tile_cycles,
    execute_tiled_linear,
    numpy_oracle_int4,
    plan_linear_tiling,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TILING_ARTIFACT_PATH = REPO_ROOT / "bench" / "results" / "tiling_correctness.json"


# ---------------------------------------------------------------------------
# Pure tile-math: BufferCapacityModel
# ---------------------------------------------------------------------------


def test_buffer_capacity_model_canonical_constants():
    cap = BufferCapacityModel(buffer_capacity_words=512, array_size=16)
    assert cap.weight_scratch_words == 64
    assert cap.input_scratch_words == 4
    assert cap.per_output_block_stride_words == 4
    assert cap.finalize_footprint_words == 64
    assert cap.default_layout == {"weight_addr": 0, "input_addr": 64, "result_addr": 68}
    assert cap.max_out_blocks_per_tile() == 96
    assert cap.max_m_per_tile() == 1536


def test_buffer_capacity_model_scales_with_capacity():
    cap_small = BufferCapacityModel(buffer_capacity_words=256, array_size=16)
    cap_large = BufferCapacityModel(buffer_capacity_words=2048, array_size=16)
    assert cap_large.max_m_per_tile() > cap_small.max_m_per_tile()
    cap_smaller_array = BufferCapacityModel(buffer_capacity_words=512, array_size=8)
    assert cap_smaller_array.max_m_per_tile() > cap_small.max_m_per_tile()


def test_buffer_capacity_model_rejects_invalid_array_size():
    with pytest.raises(ValueError):
        BufferCapacityModel(buffer_capacity_words=512, array_size=7)
    with pytest.raises(ValueError):
        BufferCapacityModel(buffer_capacity_words=0, array_size=16)


def test_peak_words_for_tile_respects_buffer():
    cap = BufferCapacityModel(buffer_capacity_words=512, array_size=16)
    assert cap.peak_words_for_tile(cap.max_m_per_tile()) == 512
    assert cap.peak_words_for_tile(16) <= 512


# ---------------------------------------------------------------------------
# plan_linear_tiling: heuristic policy
# ---------------------------------------------------------------------------


def test_plan_under_capacity_uses_single_tile():
    plan = plan_linear_tiling(
        out_features=256, in_features=512, array_size=16, buffer_capacity_words=512
    )
    assert plan.num_m_tiles == 1
    assert plan.m_tile == 256
    assert plan.tile_partition == ((0, 256),)
    assert plan.fits_in_buffer is True


def test_plan_at_boundary_uses_single_max_tile():
    plan = plan_linear_tiling(
        out_features=1536, in_features=512, array_size=16, buffer_capacity_words=512
    )
    assert plan.num_m_tiles == 1
    assert plan.m_tile == 1536
    assert plan.peak_buffer_words_per_tile == 512


def test_plan_over_capacity_partitions_into_max_fit_tiles():
    plan = plan_linear_tiling(
        out_features=2048, in_features=512, array_size=16, buffer_capacity_words=512
    )
    assert plan.m_tile == 1536
    assert plan.num_m_tiles == 2
    assert plan.tile_partition == ((0, 1536), (1536, 2048))
    last_tile_size = plan.tile_partition[-1][1] - plan.tile_partition[-1][0]
    assert last_tile_size == 512  # remainder tile must be smaller than m_tile


def test_plan_deeply_over_capacity_uses_n_tiles():
    plan = plan_linear_tiling(
        out_features=4096, in_features=768, array_size=16, buffer_capacity_words=512
    )
    assert plan.num_m_tiles == 3
    assert sum(end - start for start, end in plan.tile_partition) == 4096


def test_plan_partition_covers_full_layer_exactly():
    """No tile overlaps, no gaps, last tile may be smaller. Property
    holds for any shape × buffer combination."""
    for M, N, buf in [(513, 64, 512), (2048, 512, 512), (4096, 1024, 1024), (1000, 256, 256)]:
        plan = plan_linear_tiling(
            out_features=M, in_features=N, array_size=16, buffer_capacity_words=buf
        )
        starts = [s for s, _ in plan.tile_partition]
        ends = [e for _, e in plan.tile_partition]
        assert starts[0] == 0
        assert ends[-1] == M
        for i in range(1, len(plan.tile_partition)):
            assert starts[i] == ends[i - 1]
        assert all(0 < (e - s) <= plan.m_tile for s, e in plan.tile_partition)


def test_plan_raises_when_buffer_too_small_for_a_single_block():
    # 2 * weight_scratch (=128) + input_scratch (=4) = 132 minimum; below
    # that a single output block does not fit.
    with pytest.raises(TilingError):
        plan_linear_tiling(
            out_features=16,
            in_features=16,
            array_size=16,
            buffer_capacity_words=100,
        )


def test_plan_estimated_cycles_are_consistent():
    plan = plan_linear_tiling(
        out_features=2048, in_features=512, array_size=16, buffer_capacity_words=512
    )
    expected_per_tile = _estimate_tile_cycles(
        m_tile=plan.m_tile, in_features=512, array_size=16
    )["per_tile_cycles"]
    assert plan.estimated_per_tile_cycles == expected_per_tile
    assert plan.estimated_total_cycles == expected_per_tile * plan.num_m_tiles


# ---------------------------------------------------------------------------
# cost_model_select policy plumbing
# ---------------------------------------------------------------------------


def test_plan_cost_model_select_uses_injected_selector():
    captured = {}

    def fake_selector(candidates, shape):
        captured["candidate_count"] = len(candidates)
        captured["shape"] = shape
        return min(candidates, key=lambda c: c["m_tile"])

    plan = plan_linear_tiling(
        out_features=2048,
        in_features=512,
        array_size=16,
        buffer_capacity_words=512,
        policy="cost_model_select",
        cost_model_selector=fake_selector,
    )
    assert captured["candidate_count"] > 0
    assert captured["shape"]["out_features"] == 2048
    assert plan.policy == "cost_model_select"
    # injected selector picks the smallest candidate -> many tiles
    assert plan.m_tile == 16
    assert plan.num_m_tiles == 2048 // 16
    assert plan.cost_model_provenance["selector"] == "user_injected"


def test_plan_cost_model_select_default_path_falls_back_gracefully():
    """The default cost_model.select path should either succeed and
    select a legal m_tile, or fall back without raising. Either is
    acceptable for the planner contract."""
    plan = plan_linear_tiling(
        out_features=2048,
        in_features=512,
        array_size=16,
        buffer_capacity_words=512,
        policy="cost_model_select",
    )
    assert plan.policy == "cost_model_select"
    assert plan.m_tile % 16 == 0
    assert 16 <= plan.m_tile <= 1536
    prov = plan.cost_model_provenance
    assert prov is not None
    assert prov["selector"] in {"cost_model.select", "fallback_lowest_total_cycles"}


def test_plan_rejects_unknown_policy():
    with pytest.raises(ValueError):
        plan_linear_tiling(
            out_features=64, in_features=64, array_size=16,
            buffer_capacity_words=512, policy="bogus",
        )


# ---------------------------------------------------------------------------
# Differential correctness via ISA simulator vs NumPy oracle
# ---------------------------------------------------------------------------


def _random_int4(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.integers(-8, 8, size=shape, dtype=np.int8)


def test_untiled_lowering_actually_fails_on_over_capacity_layer():
    """Sanity: the un-tiled lowering MUST fail on the over-capacity
    workload. If this test ever passes by accident, the tiling test
    below is solving a phantom problem."""
    rng = np.random.default_rng(0xDEAD)
    W = _random_int4(rng, (2048, 512))
    x = _random_int4(rng, (512,))
    with pytest.raises((ValueError, RuntimeError)):
        lower_blocked_fc_program_utpu(
            weights_int4=W,
            activations_int4=x,
            out_features=2048,
            in_features=512,
            array_size=16,
            apply_relu=False,
            apply_quant=True,
            weight_addr=0,
            input_addr=64,
            result_addr=68,
        )


def test_tiled_execution_matches_oracle_bit_identical_under_capacity():
    rng = np.random.default_rng(0x101)
    W = _random_int4(rng, (256, 512))
    x = _random_int4(rng, (512,))
    plan = plan_linear_tiling(
        out_features=256, in_features=512, array_size=16, buffer_capacity_words=512
    )
    result = execute_tiled_linear(
        plan=plan, weights_int4=W, activations_int4=x, apply_relu=False, apply_quant=True
    )
    oracle = numpy_oracle_int4(
        weights_int4=W, activations_int4=x, apply_relu=False, apply_quant=True
    )
    assert np.array_equal(result.output_int4, oracle)


def test_tiled_execution_matches_oracle_bit_identical_at_boundary():
    rng = np.random.default_rng(0x202)
    W = _random_int4(rng, (1536, 512))
    x = _random_int4(rng, (512,))
    plan = plan_linear_tiling(
        out_features=1536, in_features=512, array_size=16, buffer_capacity_words=512
    )
    result = execute_tiled_linear(
        plan=plan, weights_int4=W, activations_int4=x, apply_relu=False, apply_quant=True
    )
    oracle = numpy_oracle_int4(
        weights_int4=W, activations_int4=x, apply_relu=False, apply_quant=True
    )
    assert np.array_equal(result.output_int4, oracle)


def test_tiled_execution_matches_oracle_bit_identical_over_capacity():
    """The headline correctness test. Same shape that fails un-tiled
    lowering above."""
    rng = np.random.default_rng(0xDEAD)
    W = _random_int4(rng, (2048, 512))
    x = _random_int4(rng, (512,))
    plan = plan_linear_tiling(
        out_features=2048, in_features=512, array_size=16, buffer_capacity_words=512
    )
    result = execute_tiled_linear(
        plan=plan, weights_int4=W, activations_int4=x, apply_relu=False, apply_quant=True
    )
    oracle = numpy_oracle_int4(
        weights_int4=W, activations_int4=x, apply_relu=False, apply_quant=True
    )
    assert result.output_int4.shape == (2048,)
    assert np.array_equal(result.output_int4, oracle)
    # And the plan must have actually tiled this (not silently dropped to 1 tile).
    assert plan.num_m_tiles >= 2


def test_tiled_execution_with_relu_matches_oracle_bit_identical_over_capacity():
    rng = np.random.default_rng(0xBEEF)
    W = _random_int4(rng, (3072, 1024))
    x = _random_int4(rng, (1024,))
    plan = plan_linear_tiling(
        out_features=3072, in_features=1024, array_size=16, buffer_capacity_words=512
    )
    result = execute_tiled_linear(
        plan=plan, weights_int4=W, activations_int4=x, apply_relu=True, apply_quant=True
    )
    oracle = numpy_oracle_int4(
        weights_int4=W, activations_int4=x, apply_relu=True, apply_quant=True
    )
    assert np.array_equal(result.output_int4, oracle)


# ---------------------------------------------------------------------------
# Artifact schema lock + correctness gate
# ---------------------------------------------------------------------------


REQUIRED_TOP_KEYS = {
    "phase",
    "scope",
    "buffer_model",
    "environment",
    "workloads",
    "summary",
    "generated_at",
}
REQUIRED_BUFFER_MODEL_KEYS = {
    "array_size",
    "buffer_capacity_words",
    "max_out_blocks_per_tile",
    "max_m_per_tile",
    "weight_scratch_words",
    "input_scratch_words",
    "finalize_footprint_words",
    "default_layout",
}
REQUIRED_WORKLOAD_KEYS = {
    "name",
    "description",
    "out_features",
    "in_features",
    "apply_relu",
    "seed",
    "array_size",
    "buffer_capacity_words",
    "plan",
    "execution",
    "correctness",
}
REQUIRED_PLAN_KEYS = {
    "m_tile",
    "num_m_tiles",
    "tile_partition",
    "in_blocks_per_tile",
    "peak_buffer_words_per_tile",
    "fits_in_buffer",
    "policy",
    "estimated_per_tile_cycles",
    "estimated_total_cycles",
    "candidates_considered",
    "cost_model_provenance",
    "notes",
}
REQUIRED_EXECUTION_KEYS = {
    "num_tiles",
    "total_sim_cycles_sequential",
    "total_program_words",
    "per_tile_cycles",
    "per_tile_program_words",
    "fits_instruction_bram_per_tile",
}
REQUIRED_CORRECTNESS_KEYS = {
    "max_abs_error_int4",
    "mismatch_count",
    "total_outputs",
    "bit_identical_to_oracle",
}
REQUIRED_SUMMARY_KEYS = {
    "workload_count",
    "all_bit_identical_to_oracle",
    "max_max_abs_error_int4",
    "max_num_tiles",
    "max_total_sim_cycles_sequential",
}


def _load_artifact() -> dict:
    if not TILING_ARTIFACT_PATH.exists():
        pytest.skip(
            f"tiling_correctness.json missing at {TILING_ARTIFACT_PATH}; "
            "run `python firmware/host/run_tiling_correctness.py` to generate it."
        )
    with open(TILING_ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_tiling_artifact_schema():
    artifact = _load_artifact()
    assert REQUIRED_TOP_KEYS.issubset(artifact.keys())
    assert artifact["phase"] == "phase_3_tiling_controller"
    assert REQUIRED_BUFFER_MODEL_KEYS.issubset(artifact["buffer_model"].keys())
    assert REQUIRED_SUMMARY_KEYS.issubset(artifact["summary"].keys())
    assert isinstance(artifact["workloads"], list)
    assert artifact["summary"]["workload_count"] == len(artifact["workloads"])


def test_tiling_artifact_every_workload_bit_identical():
    artifact = _load_artifact()
    for w in artifact["workloads"]:
        assert REQUIRED_WORKLOAD_KEYS.issubset(w.keys()), w["name"]
        assert REQUIRED_PLAN_KEYS.issubset(w["plan"].keys()), w["name"]
        assert REQUIRED_EXECUTION_KEYS.issubset(w["execution"].keys()), w["name"]
        assert REQUIRED_CORRECTNESS_KEYS.issubset(w["correctness"].keys()), w["name"]
        assert w["correctness"]["bit_identical_to_oracle"], (
            f"{w['name']}: max_abs_err={w['correctness']['max_abs_error_int4']}, "
            f"mismatches={w['correctness']['mismatch_count']}"
        )
        assert w["correctness"]["max_abs_error_int4"] == 0
        assert w["correctness"]["mismatch_count"] == 0


def test_tiling_artifact_covers_at_least_one_over_capacity_workload():
    """Defensive: at least one workload must actually exercise tiling
    (num_m_tiles > 1) and must be bit-identical. Without this, the
    artifact could pass schema + correctness while only running
    under-capacity layers."""
    artifact = _load_artifact()
    over_capacity = [w for w in artifact["workloads"] if w["plan"]["num_m_tiles"] > 1]
    assert over_capacity, "expected at least one workload with num_m_tiles > 1"
    for w in over_capacity:
        assert w["correctness"]["bit_identical_to_oracle"], w["name"]


def test_tiling_artifact_summary_flag_matches_truth():
    artifact = _load_artifact()
    expected = all(
        w["correctness"]["bit_identical_to_oracle"] for w in artifact["workloads"]
    )
    assert artifact["summary"]["all_bit_identical_to_oracle"] is expected


def test_decode_tile_outputs_round_trip():
    """Encoder + simulator finalize + decoder must round-trip through
    the fetch-byte protocol exactly."""
    rng = np.random.default_rng(0xC0DE)
    W = _random_int4(rng, (32, 16))
    x = _random_int4(rng, (16,))
    lowered = lower_blocked_fc_program_utpu(
        weights_int4=W,
        activations_int4=x,
        out_features=32,
        in_features=16,
        array_size=16,
        apply_relu=False,
        apply_quant=True,
        weight_addr=0,
        input_addr=64,
        result_addr=68,
    )
    res = simulate_program_bytes(lowered["program"], array_size=16, buffer_size=512)
    decoded = _decode_tile_outputs_from_fetch_bytes(
        res.fetch_bytes, m_tile=32, array_size=16
    )
    oracle = numpy_oracle_int4(
        weights_int4=W, activations_int4=x, apply_relu=False, apply_quant=True
    )
    assert np.array_equal(decoded, oracle)
