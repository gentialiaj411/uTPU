"""Phase 3 evidence script: tile a set of `Linear(M, N)` layers onto the
current fixed-capacity unified buffer, simulate every tile through the
Python ISA simulator, and verify the concatenated INT4 output is
bit-identical to a NumPy int32-accumulated oracle.

The workload set includes:

* an under-capacity layer (`M=256, N=512`) — fits in a single tile;
  proves the planner does not over-partition.
* a boundary-capacity layer (`M=1536, N=512`) — exactly at the max
  `m_tile`; proves the buffer math is tight.
* an over-capacity layer (`M=2048, N=512`) — fails un-tiled lowering at
  ISA-encode time (`Address >= 512 out of range`); tiling resolves it.
* a deeply over-capacity layer (`M=4096, N=768`) — 3 tiles; proves the
  pass scales beyond 2x capacity.
* an asymmetric layer (`M=3072, N=1024`) with `apply_relu=True` —
  exercises the `_run_finalize` leaky-relu path inside every tile.

For every workload we record: layer shape, buffer capacity, chosen
`m_tile`, tile partition, peak buffer words per tile, ISA-sim cycles,
total program words, max-abs-error vs oracle, and a
`bit_identical_to_oracle` boolean.

Determinism: every workload seeds its own RNG; the artifact is
byte-stable across reruns on the same Python/NumPy.

Honest scoping: this is **simulator-only** evidence. No RTL changes
were made and no on-board execution is claimed. The companion test
(`test_tiling_controller.py`) consumes this artifact and gates on
correctness, not on a specific cycle count.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tiling_controller import (
    BufferCapacityModel,
    execute_tiled_linear,
    numpy_oracle_int4,
    plan_linear_tiling,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "tiling_correctness.json"

DEFAULT_ARRAY_SIZE = 16
DEFAULT_BUFFER_WORDS = 512


WORKLOADS: List[Dict[str, Any]] = [
    {
        "name": "linear_256x512_under_capacity",
        "out_features": 256,
        "in_features": 512,
        "apply_relu": False,
        "seed": 0x101,
        "description": "Fits in a single tile (max_m_per_tile=1536); proves planner does not over-partition.",
    },
    {
        "name": "linear_1536x512_at_capacity",
        "out_features": 1536,
        "in_features": 512,
        "apply_relu": False,
        "seed": 0x202,
        "description": "Exactly at max_m_per_tile boundary for AS=16/BS=512; tight buffer-math test.",
    },
    {
        "name": "linear_2048x512_over_capacity",
        "out_features": 2048,
        "in_features": 512,
        "apply_relu": False,
        "seed": 0x303,
        "description": "Un-tiled lowering fails at encode time (9-bit address overflow); tiling resolves it.",
    },
    {
        "name": "linear_4096x768_2x_over_capacity",
        "out_features": 4096,
        "in_features": 768,
        "apply_relu": False,
        "seed": 0x404,
        "description": "3 tiles; proves the pass scales beyond 2x capacity with bit-exact output.",
    },
    {
        "name": "linear_3072x1024_relu_over_capacity",
        "out_features": 3072,
        "in_features": 1024,
        "apply_relu": True,
        "seed": 0x505,
        "description": "Exercises _run_finalize leaky-relu path inside every tile; multi-tile + activation.",
    },
]


def _run_workload(spec: Dict[str, Any], array_size: int, buffer_words: int) -> Dict[str, Any]:
    rng = np.random.default_rng(spec["seed"])
    M = int(spec["out_features"])
    N = int(spec["in_features"])
    weights = rng.integers(-8, 8, size=(M, N), dtype=np.int8)
    activations = rng.integers(-8, 8, size=(N,), dtype=np.int8)

    plan = plan_linear_tiling(
        out_features=M,
        in_features=N,
        array_size=array_size,
        buffer_capacity_words=buffer_words,
        policy="max_fit_heuristic",
        layer_name=spec["name"],
    )

    result = execute_tiled_linear(
        plan=plan,
        weights_int4=weights,
        activations_int4=activations,
        apply_relu=bool(spec["apply_relu"]),
        apply_quant=True,
    )
    oracle = numpy_oracle_int4(
        weights_int4=weights,
        activations_int4=activations,
        apply_relu=bool(spec["apply_relu"]),
        apply_quant=True,
    )
    diff = np.abs(result.output_int4.astype(np.int64) - oracle.astype(np.int64))
    max_abs_err = int(diff.max()) if diff.size else 0
    mismatches = int(np.count_nonzero(diff))

    return {
        "name": spec["name"],
        "description": spec["description"],
        "out_features": M,
        "in_features": N,
        "apply_relu": bool(spec["apply_relu"]),
        "seed": int(spec["seed"]),
        "array_size": int(array_size),
        "buffer_capacity_words": int(buffer_words),
        "plan": plan.to_dict(),
        "execution": {
            "num_tiles": len(result.per_tile),
            "total_sim_cycles_sequential": int(result.total_sim_cycles_sequential),
            "total_program_words": int(result.total_program_words),
            "per_tile_cycles": [t.sim_cycles_sequential for t in result.per_tile],
            "per_tile_program_words": [t.program_words for t in result.per_tile],
            "fits_instruction_bram_per_tile": [
                bool(t.fits_instruction_bram) for t in result.per_tile
            ],
        },
        "correctness": {
            "max_abs_error_int4": max_abs_err,
            "mismatch_count": mismatches,
            "total_outputs": int(result.output_int4.size),
            "bit_identical_to_oracle": bool(max_abs_err == 0 and mismatches == 0),
        },
    }


def run_tiling_correctness(
    *,
    array_size: int = DEFAULT_ARRAY_SIZE,
    buffer_words: int = DEFAULT_BUFFER_WORDS,
    output_path: Path = DEFAULT_OUTPUT_JSON,
) -> Dict[str, Any]:
    capacity = BufferCapacityModel(
        buffer_capacity_words=buffer_words, array_size=array_size
    )
    workloads = [
        _run_workload(spec, array_size=array_size, buffer_words=buffer_words)
        for spec in WORKLOADS
    ]
    summary = {
        "workload_count": len(workloads),
        "all_bit_identical_to_oracle": all(
            w["correctness"]["bit_identical_to_oracle"] for w in workloads
        ),
        "max_max_abs_error_int4": max(
            w["correctness"]["max_abs_error_int4"] for w in workloads
        ),
        "max_num_tiles": max(w["plan"]["num_m_tiles"] for w in workloads),
        "max_total_sim_cycles_sequential": max(
            w["execution"]["total_sim_cycles_sequential"] for w in workloads
        ),
    }

    artifact = {
        "phase": "phase_3_tiling_controller",
        "scope": (
            "Simulator-only correctness evidence. NumPy oracle vs ISA simulator "
            "execution of the tiled program. No RTL changes; no on-board claim."
        ),
        "buffer_model": {
            "array_size": int(capacity.array_size),
            "buffer_capacity_words": int(capacity.buffer_capacity_words),
            "max_out_blocks_per_tile": int(capacity.max_out_blocks_per_tile()),
            "max_m_per_tile": int(capacity.max_m_per_tile()),
            "weight_scratch_words": int(capacity.weight_scratch_words),
            "input_scratch_words": int(capacity.input_scratch_words),
            "finalize_footprint_words": int(capacity.finalize_footprint_words),
            "default_layout": capacity.default_layout,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "workloads": workloads,
        "summary": summary,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    return artifact


def _print_summary(artifact: Dict[str, Any]) -> None:
    bm = artifact["buffer_model"]
    print(
        f"array_size={bm['array_size']}  buffer_capacity_words={bm['buffer_capacity_words']}  "
        f"max_m_per_tile={bm['max_m_per_tile']}"
    )
    for w in artifact["workloads"]:
        p = w["plan"]
        e = w["execution"]
        c = w["correctness"]
        print(
            f"  - {w['name']:<48} M={w['out_features']:<5} N={w['in_features']:<5} "
            f"m_tile={p['m_tile']:<5} tiles={p['num_m_tiles']:<3} "
            f"peak={p['peak_buffer_words_per_tile']:<4} cycles={e['total_sim_cycles_sequential']:<9} "
            f"max_abs_err={c['max_abs_error_int4']}  bit_identical={c['bit_identical_to_oracle']}"
        )
    s = artifact["summary"]
    print(
        f"summary: workloads={s['workload_count']}  "
        f"all_bit_identical_to_oracle={s['all_bit_identical_to_oracle']}  "
        f"max_max_abs_error_int4={s['max_max_abs_error_int4']}  "
        f"max_num_tiles={s['max_num_tiles']}"
    )


if __name__ == "__main__":
    artifact = run_tiling_correctness()
    _print_summary(artifact)
    print(f"\nartifact -> {DEFAULT_OUTPUT_JSON.relative_to(REPO_ROOT)}")
