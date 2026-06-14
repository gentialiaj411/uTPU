from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from isa_encoder import DEFAULT_CFG, IsaConfig
from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "batched_gemm_correctness.json"

ARRAY_SIZE = 16
LEGACY_WEIGHT_ADDR = 256
LEGACY_INPUT_ADDR = 0
LEGACY_RESULT_ADDR = 320

BATCHED_CFG = IsaConfig(address_width=12, compute_data_width=4)
BATCHED_WEIGHT_ADDR = 0
BATCHED_INPUT_ADDR = 128
BATCHED_RESULT_ADDR = 256
BATCHED_BUFFER_SIZE = 1 << BATCHED_CFG.address_width

SHAPE_BATCH_SWEEP: Sequence[Tuple[int, int, int]] = (
    (16, 16, 1),
    (16, 16, 4),
    (16, 16, 8),
    (32, 16, 1),
    (32, 16, 4),
    (16, 32, 1),
    (16, 32, 4),
)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _pack_int4_fetch_bytes(vals: np.ndarray) -> List[int]:
    flat = vals.astype(np.int8).tolist()
    out: List[int] = []
    for i in range(0, len(flat), 4):
        chunk = flat[i : i + 4]
        while len(chunk) < 4:
            chunk.append(0)
        out.append((chunk[0] & 0xF) | ((chunk[1] & 0xF) << 4))
        out.append((chunk[2] & 0xF) | ((chunk[3] & 0xF) << 4))
    return out


def expected_fetch_bytes_for_batched_blocked_fc(
    weights_int4: np.ndarray,
    activations_int4: np.ndarray,
    *,
    out_features: int,
    in_features: int,
    array_size: int,
    apply_relu: bool,
) -> List[int]:
    w = np.asarray(weights_int4, dtype=np.int8)
    x = np.asarray(activations_int4, dtype=np.int8)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    acc = x.astype(np.int32) @ w.astype(np.int32).T
    q = np.clip(acc, -8, 7).astype(np.int8)
    if apply_relu:
        q = np.where(q < 0, q >> 2, q).astype(np.int8)

    batch_size = int(x.shape[0])
    out_blocks = (out_features + array_size - 1) // array_size
    out_padded = out_blocks * array_size
    q_pad = np.zeros((batch_size, out_padded), dtype=np.int8)
    q_pad[:, :out_features] = q

    fetch: List[int] = []
    for ob in range(out_blocks):
        o0 = ob * array_size
        o1 = o0 + array_size
        block = q_pad[:, o0:o1].T  # [array_size, batch]
        fetch.extend(_pack_int4_fetch_bytes(block.flatten(order="F")))
    return fetch


def _gen_case_tensors(out_features: int, in_features: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    seed = 0xB470 + out_features * 37 + in_features * 13 + batch_size
    rng = np.random.default_rng(seed)
    w = rng.integers(low=-8, high=8, size=(out_features, in_features), dtype=np.int8)
    x = rng.integers(low=-8, high=8, size=(batch_size, in_features), dtype=np.int8)
    return w, x


def _build_case(out_features: int, in_features: int, batch_size: int) -> Dict[str, object]:
    weights, activations = _gen_case_tensors(out_features, in_features, batch_size)
    cfg = DEFAULT_CFG if batch_size == 1 else BATCHED_CFG
    weight_addr = LEGACY_WEIGHT_ADDR if batch_size == 1 else BATCHED_WEIGHT_ADDR
    input_addr = LEGACY_INPUT_ADDR if batch_size == 1 else BATCHED_INPUT_ADDR
    result_addr = LEGACY_RESULT_ADDR if batch_size == 1 else BATCHED_RESULT_ADDR
    buffer_size = 512 if batch_size == 1 else BATCHED_BUFFER_SIZE

    lowered = lower_blocked_fc_program_utpu(
        weights,
        activations if batch_size > 1 else activations[0],
        out_features,
        in_features,
        ARRAY_SIZE,
        False,
        True,
        weight_addr,
        input_addr,
        result_addr,
        cfg=cfg,
    )
    sim1 = simulate_program_bytes(
        lowered["program"],
        array_size=ARRAY_SIZE,
        buffer_size=buffer_size,
        cfg=cfg,
    )
    sim2 = simulate_program_bytes(
        lowered["program"],
        array_size=ARRAY_SIZE,
        buffer_size=buffer_size,
        cfg=cfg,
    )
    expected = expected_fetch_bytes_for_batched_blocked_fc(
        weights,
        activations,
        out_features=out_features,
        in_features=in_features,
        array_size=ARRAY_SIZE,
        apply_relu=False,
    )
    identity_with_singleton_batch = None
    if batch_size == 1:
        singleton = lower_blocked_fc_program_utpu(
            weights,
            activations,
            out_features,
            in_features,
            ARRAY_SIZE,
            False,
            True,
            weight_addr,
            input_addr,
            result_addr,
            cfg=cfg,
        )
        identity_with_singleton_batch = bool(singleton["program"] == lowered["program"])

    weight_load_cycles = int(sim1.executed_ops["load_weights"])
    total_cycles = int(sim1.cycle_count_sequential)
    return {
        "shape": {"out_features": int(out_features), "in_features": int(in_features)},
        "batch_size": int(batch_size),
        "program_instruction_words": int(lowered["program_instruction_words"]),
        "program_byte_length": len(lowered["program"]),
        "bit_exact_vs_oracle": bool(sim1.fetch_bytes == expected),
        "deterministic_fetch_bytes": bool(sim1.fetch_bytes == sim2.fetch_bytes),
        "deterministic_cycle_count": bool(sim1.cycle_count_sequential == sim2.cycle_count_sequential),
        "cycle_count_sequential": total_cycles,
        "compute_runs": int(sim1.compute_runs),
        "total_macs": int(sim1.total_macs),
        "array_utilization": float(sim1.array_utilization),
        "weight_load_cycles": weight_load_cycles,
        "weight_load_fraction_of_total_cycles": (weight_load_cycles / total_cycles) if total_cycles > 0 else 0.0,
        "load_inputs_cycles": int(sim1.executed_ops["load_inputs"]),
        "legacy_b1_program_byte_identity": identity_with_singleton_batch,
        "expected_fetch_bytes": expected,
        "actual_fetch_bytes": list(sim1.fetch_bytes),
    }


def build_artifact() -> Dict[str, object]:
    cases = [_build_case(out_f, in_f, batch) for (out_f, in_f, batch) in SHAPE_BATCH_SWEEP]
    grouped: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for case in cases:
        shape = case["shape"]
        key = (int(shape["out_features"]), int(shape["in_features"]))
        grouped.setdefault(key, []).append(case)

    amortization_checks = []
    for key, shape_cases in grouped.items():
        by_batch = sorted(shape_cases, key=lambda c: int(c["batch_size"]))
        fractions = [float(c["weight_load_fraction_of_total_cycles"]) for c in by_batch]
        amortization_checks.append(
            {
                "shape": {"out_features": key[0], "in_features": key[1]},
                "batch_sizes": [int(c["batch_size"]) for c in by_batch],
                "weight_load_fraction_nonincreasing": bool(
                    all(lhs >= rhs for lhs, rhs in zip(fractions, fractions[1:]))
                ),
            }
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "status": "ok",
        "methodology": {
            "scope": (
                "Host-side ISA lowering + python ISA simulator correctness for blocked-FC "
                "batched GEMM. B=1 vector lowering stays byte-identical to the pre-batching "
                "dc7a517 vector path (pinned by firmware/host/fixtures/b1_legacy_program_bytes.json "
                "and test_batched_gemm_b1_program_matches_pre_batching_legacy_bytes); B=1 vector "
                "and 1-row matrix activations emit identical programs; B>1 uses extended-address "
                "encoding and widened buffer addresses."
            ),
            "array_size": ARRAY_SIZE,
            "legacy_cfg": {
                "address_width": DEFAULT_CFG.address_width,
                "compute_data_width": DEFAULT_CFG.compute_data_width,
            },
            "batched_cfg": {
                "address_width": BATCHED_CFG.address_width,
                "compute_data_width": BATCHED_CFG.compute_data_width,
                "buffer_size": BATCHED_BUFFER_SIZE,
            },
            "shape_batch_sweep": [
                {"out_features": out_f, "in_features": in_f, "batch_size": batch}
                for (out_f, in_f, batch) in SHAPE_BATCH_SWEEP
            ],
        },
        "cases": cases,
        "aggregate": {
            "all_cases_bit_exact_vs_oracle": bool(all(c["bit_exact_vs_oracle"] for c in cases)),
            "all_cases_deterministic": bool(
                all(c["deterministic_fetch_bytes"] and c["deterministic_cycle_count"] for c in cases)
            ),
            "all_b1_programs_byte_identical": bool(
                all(c["legacy_b1_program_byte_identity"] for c in cases if c["batch_size"] == 1)
            ),
            "weight_load_amortization_checks": amortization_checks,
        },
    }


def main() -> None:
    artifact = build_artifact()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[batched_gemm_correctness] wrote {OUTPUT_JSON}")
    print(
        f"[batched_gemm_correctness] all_cases_bit_exact_vs_oracle="
        f"{artifact['aggregate']['all_cases_bit_exact_vs_oracle']}"
    )


if __name__ == "__main__":
    main()
