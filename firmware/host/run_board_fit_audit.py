"""Phase 7 remediation P3 — board-fit audit artifact generator.

Why this exists: the existing artifact `bench/results/tiling_correctness.json`
reports `fits_instruction_bram_per_tile = [false, ...]` for every
workload, because the lowering hard-coded `prog_depth = 1024` (the
PROG_DEPTH of the current RTL top.sv default). That made "board
execution" structurally blocked — the smallest workload in that
artifact (`256x512`) lowers to 106 129 instruction words, ~100x the
2 KiB BRAM.

This audit makes the gap honest and parameter-driven:

1. We lower a representative shape grid (the same one
   `tiling_correctness.json` uses plus a "tiny demos that already fit"
   sub-grid that the worker scanned).
2. For each shape we report `program_instruction_words` (independent
   of board; that's the size the encoder produces).
3. For each `BoardConfig` we report `fits = words <= prog_depth` per
   shape, and an aggregate `shapes_fit_count` per board.

The output `bench/results/board_fit_audit.json` is the single source
of truth a reader can consult to answer "which shapes fit on which
board?". Nothing is invented — the lowering already produces
`program_instruction_words` deterministically and the boards are the
three reference configs in `firmware/host/board_config.py`.

Board-execution unlock path (documented honestly):

- `pynqz2_baseline` (`PROG_DEPTH = 1024`, today's bitstream): only
  tiny demos fit. The audit will list them per shape.
- `pynqz2_bram_max` (`PROG_DEPTH = 8192`, synthesis-time change):
  covers single-tile MLPs in the (M, K) <= (64, 128) range. Bitstream
  re-synth required.
- `vu13p_uram` (`PROG_DEPTH = 131072`, larger part + URAM): covers the
  full Phase 3 shape grid in a single tile. Different bitstream + board.

Run: `python firmware/host/run_board_fit_audit.py`
Tests: `firmware/host/test_board_fit_audit.py`
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from board_config import BoardConfig
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_PATH = REPO_ROOT / "bench" / "results" / "board_fit_audit.json"


SHAPE_GRID: List[Tuple[int, int, str]] = [
    (16, 16, "demo_smallest"),
    (16, 32, "demo_tiny"),
    (16, 64, "demo_small_K"),
    (32, 32, "demo_2x2_blocks"),
    (32, 64, "first_overflow_pynqz2_baseline"),
    (64, 64, "phase1_calibration_small"),
    (64, 128, "single_tile_mlp_class"),
    (128, 128, "single_tile_mlp_class"),
    (256, 16, "phase1_calibration"),
    (256, 256, "phase1_calibration"),
    (256, 512, "phase3_under_capacity"),
    (512, 256, "phase1_calibration_large"),
    (512, 512, "phase1_calibration_largest"),
    (1536, 512, "phase3_at_capacity_largest_single_tile"),
]


def _lower(out_features: int, in_features: int, array_size: int, prog_depth: int, seed: int) -> Dict[str, Any]:
    # Use the same buffer layout the Phase 3 tiling controller uses so the
    # `program_instruction_words` we report is directly comparable to the
    # `per_tile_program_words` values in `bench/results/tiling_correctness.json`.
    # weight_addr=0, input_addr=64, result_addr=68 keeps all addresses under
    # the default 9-bit address bus for every shape in SHAPE_GRID.
    weight_addr = 0
    input_addr = 64
    result_addr = 68

    rng = np.random.default_rng(seed)
    weights = rng.integers(-8, 8, size=(out_features, in_features), dtype=np.int8)
    activations = rng.integers(-8, 8, size=(in_features,), dtype=np.int8)
    return lower_blocked_fc_program_utpu(
        weights_int4=weights,
        activations_int4=activations,
        out_features=out_features,
        in_features=in_features,
        array_size=array_size,
        apply_relu=False,
        apply_quant=True,
        weight_addr=weight_addr,
        input_addr=input_addr,
        result_addr=result_addr,
        prog_depth=int(prog_depth),
    )


def _per_shape_audit(
    boards: List[BoardConfig],
    grid: List[Tuple[int, int, str]],
    array_size: int = 16,
    seed: int = 0xB041D,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for out_features, in_features, tag in grid:
        result = _lower(
            out_features, in_features, array_size,
            prog_depth=10**9,
            seed=int(seed) ^ (out_features * 2654435761 + in_features * 7),
        )
        words = int(result["program_instruction_words"])
        rows.append({
            "shape": {
                "out_features": int(out_features),
                "in_features": int(in_features),
                "array_size": int(array_size),
            },
            "tag": tag,
            "program_instruction_words": words,
            "block_ops": int(result["block_ops"]),
            "out_blocks": int(result["out_blocks"]),
            "in_blocks": int(result["in_blocks"]),
            "fits_per_board": {
                b.name: bool(words <= b.prog_depth)
                for b in boards
            },
        })
    return rows


def _aggregate(per_shape: List[Dict[str, Any]], boards: List[BoardConfig]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {
        "shape_count": len(per_shape),
        "per_board": {},
    }
    for board in boards:
        fit_shapes = [
            (row["shape"]["out_features"], row["shape"]["in_features"])
            for row in per_shape if row["fits_per_board"][board.name]
        ]
        agg["per_board"][board.name] = {
            "prog_depth": int(board.prog_depth),
            "shapes_fit_count": len(fit_shapes),
            "shapes_fit": [{"out_features": m, "in_features": k} for (m, k) in fit_shapes],
            "shapes_fit_fraction": float(len(fit_shapes)) / float(len(per_shape)) if per_shape else 0.0,
        }
    return agg


def _methodology(boards: List[BoardConfig]) -> Dict[str, Any]:
    return {
        "api": "firmware/host/run_board_fit_audit.py",
        "what_it_measures": (
            "For each (board_config, shape) pair, does the blocked-FC "
            "lowering's instruction stream fit in the board's instruction "
            "BRAM (PROG_DEPTH)? The shape grid covers the tiny demos that "
            "already fit on the shipping pynqz2_baseline bitstream, the "
            "Phase 1 cost-model calibration grid, and the Phase 3 tiling-"
            "correctness shapes."
        ),
        "fit_criterion": (
            "program_instruction_words <= board.prog_depth "
            "(each 16-bit instruction word; the encoder produces a deterministic "
            "byte stream so the count is reproducible per shape)."
        ),
        "boards": [b.as_dict() for b in boards],
        "shape_source": "firmware/host/run_board_fit_audit.py::SHAPE_GRID",
        "lowering_api": "firmware/host/lowering_blocked_fc_utpu.py::lower_blocked_fc_program_utpu",
        "notes": [
            "This audit does NOT execute the program; it only checks the "
            "encoded program length against PROG_DEPTH. Bit-exactness vs "
            "the NumPy oracle is gated by tiling_correctness.json + "
            "test_tiling_controller.py for the in-bound shapes.",
            "The RTL parameter top.sv::PROG_DEPTH was already parameterised; "
            "before this audit only the host hard-coded 1024 in 5+ places. "
            "lowering_blocked_fc_utpu.py now takes prog_depth as a kwarg "
            "(DEFAULT_PROG_DEPTH=1024) so the lowering's fit-flag matches "
            "the board the user is targeting.",
            "Board-execution unlock path: synthesise top.sv with PROG_DEPTH "
            "overridden to a board with capacity (pynqz2_bram_max / "
            "vu13p_uram). No host or ISA change is required.",
        ],
    }


def _build_payload(artix_prog_depth: int | None = None) -> Dict[str, Any]:
    boards = BoardConfig.reference_set(artix_prog_depth=artix_prog_depth)
    per_shape = _per_shape_audit(boards, SHAPE_GRID)
    return {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "phase": "phase_7_remediation_p3_board_fit_audit",
        "scope": (
            "Reports which boards (PROG_DEPTH) admit which shapes' lowered "
            "instruction streams. No execution; no claim about correctness "
            "beyond the existing tiling-correctness artifact for in-bound "
            "shapes. Includes artix_a7100t_bram_max when PROG_DEPTH sweep closes."
        ),
        "methodology": _methodology(boards),
        "per_shape": per_shape,
        "aggregate": _aggregate(per_shape, boards),
    }


def _largest_closing_artix_prog_depth() -> int | None:
    """Largest closed PROG_DEPTH that the two-byte UART length can fill.

    Sweep may close 131072 BRAM, but UPLOAD_LEN_MAX caps uploadable words at
    65535 — do not advertise a board depth the host protocol cannot fill.
    """
    sweep = REPO_ROOT / "bench" / "results" / "prog_depth_sweep.json"
    if not sweep.exists():
        return None
    try:
        data = json.loads(sweep.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    points = data.get("summary", {}).get("prog_depth_points") or data.get("points") or []
    closed = [
        int(p["PROG_DEPTH"])
        for p in points
        if p.get("status") == "closed"
        and int(p.get("BUFFER_SIZE") or 0) == 4096
        and int(p.get("PROG_DEPTH") or 0) <= 65536
    ]
    return max(closed) if closed else data.get("summary", {}).get(
        "largest_closing_PROG_DEPTH_at_buffer_4096"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH))
    parser.add_argument(
        "--artix-prog-depth",
        type=int,
        default=None,
        help="Override artix_a7100t_bram_max PROG_DEPTH (default: from prog_depth_sweep.json or 65536)",
    )
    args = parser.parse_args()
    artix = args.artix_prog_depth
    if artix is None:
        artix = _largest_closing_artix_prog_depth()
    payload = _build_payload(artix_prog_depth=artix)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    agg = payload["aggregate"]
    for board_name, board_info in agg["per_board"].items():
        print(
            f"[board_fit_audit] {board_name}: "
            f"prog_depth={board_info['prog_depth']} "
            f"fit={board_info['shapes_fit_count']}/{agg['shape_count']} "
            f"({board_info['shapes_fit_fraction']:.0%})"
        )
    print(f"[board_fit_audit] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
