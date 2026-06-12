import json
from pathlib import Path

import numpy as np

from isa_encoder import DEFAULT_CFG, IsaConfig
from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from run_batched_gemm_correctness import (
    ARRAY_SIZE,
    BATCHED_BUFFER_SIZE,
    BATCHED_CFG,
    BATCHED_INPUT_ADDR,
    BATCHED_RESULT_ADDR,
    BATCHED_WEIGHT_ADDR,
    LEGACY_INPUT_ADDR,
    LEGACY_RESULT_ADDR,
    LEGACY_WEIGHT_ADDR,
    build_artifact,
    expected_fetch_bytes_for_batched_blocked_fc,
)


ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "bench" / "results" / "batched_gemm_correctness.json"


def _gen(out_features: int, in_features: int, batch_size: int):
    seed = 0xB470 + out_features * 37 + in_features * 13 + batch_size
    rng = np.random.default_rng(seed)
    w = rng.integers(low=-8, high=8, size=(out_features, in_features), dtype=np.int8)
    x = rng.integers(low=-8, high=8, size=(batch_size, in_features), dtype=np.int8)
    return w, x


def test_batched_gemm_matches_numpy_oracle_shape_batch_sweep():
    for out_features, in_features, batch_size in ((16, 16, 4), (32, 16, 4), (16, 32, 4)):
        weights, activations = _gen(out_features, in_features, batch_size)
        lowered = lower_blocked_fc_program_utpu(
            weights,
            activations,
            out_features,
            in_features,
            ARRAY_SIZE,
            False,
            True,
            BATCHED_WEIGHT_ADDR,
            BATCHED_INPUT_ADDR,
            BATCHED_RESULT_ADDR,
            cfg=BATCHED_CFG,
        )
        sim = simulate_program_bytes(
            lowered["program"],
            array_size=ARRAY_SIZE,
            buffer_size=BATCHED_BUFFER_SIZE,
            cfg=BATCHED_CFG,
        )
        expected = expected_fetch_bytes_for_batched_blocked_fc(
            weights,
            activations,
            out_features=out_features,
            in_features=in_features,
            array_size=ARRAY_SIZE,
            apply_relu=False,
        )
        assert sim.fetch_bytes == expected
        assert sim.total_macs == ARRAY_SIZE * ARRAY_SIZE * batch_size * lowered["in_blocks"] * lowered["out_blocks"]


def test_batched_gemm_b1_program_is_byte_identical():
    out_features, in_features = 32, 16
    weights, activations = _gen(out_features, in_features, 1)
    legacy = lower_blocked_fc_program_utpu(
        weights,
        activations[0],
        out_features,
        in_features,
        ARRAY_SIZE,
        False,
        True,
        LEGACY_WEIGHT_ADDR,
        LEGACY_INPUT_ADDR,
        LEGACY_RESULT_ADDR,
        cfg=DEFAULT_CFG,
    )
    singleton_batch = lower_blocked_fc_program_utpu(
        weights,
        activations,
        out_features,
        in_features,
        ARRAY_SIZE,
        False,
        True,
        LEGACY_WEIGHT_ADDR,
        LEGACY_INPUT_ADDR,
        LEGACY_RESULT_ADDR,
        cfg=DEFAULT_CFG,
    )
    assert legacy["program"] == singleton_batch["program"]


def test_batched_gemm_is_deterministic():
    weights, activations = _gen(16, 16, 8)
    lowered = lower_blocked_fc_program_utpu(
        weights,
        activations,
        16,
        16,
        ARRAY_SIZE,
        False,
        True,
        BATCHED_WEIGHT_ADDR,
        BATCHED_INPUT_ADDR,
        BATCHED_RESULT_ADDR,
        cfg=BATCHED_CFG,
    )
    sim1 = simulate_program_bytes(
        lowered["program"],
        array_size=ARRAY_SIZE,
        buffer_size=BATCHED_BUFFER_SIZE,
        cfg=BATCHED_CFG,
    )
    sim2 = simulate_program_bytes(
        lowered["program"],
        array_size=ARRAY_SIZE,
        buffer_size=BATCHED_BUFFER_SIZE,
        cfg=BATCHED_CFG,
    )
    assert sim1.fetch_bytes == sim2.fetch_bytes
    assert sim1.cycle_count_sequential == sim2.cycle_count_sequential


def test_batched_gemm_artifact_schema_and_invariants():
    if not ARTIFACT_PATH.exists():
        artifact = build_artifact()
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["status"] == "ok"
    assert data["aggregate"]["all_cases_bit_exact_vs_oracle"] is True
    assert data["aggregate"]["all_cases_deterministic"] is True
    assert data["aggregate"]["all_b1_programs_byte_identical"] is True
    for check in data["aggregate"]["weight_load_amortization_checks"]:
        assert check["weight_load_fraction_nonincreasing"] is True
