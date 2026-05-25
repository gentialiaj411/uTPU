"""Phase 7 remediation P3 contract: bench/results/board_fit_audit.json.

Locks the schema of the board-fit audit and the **minimum reality
floor** the project commits to:

- At least one shape MUST fit in the `pynqz2_baseline` configuration
  (PROG_DEPTH=1024). Without this floor, "board execution" is
  vapourware on the shipping bitstream. The smallest workload from
  the audit (16x16) must fit.
- Every shape that fits at `pynqz2_baseline` MUST also fit at the
  larger boards (monotonicity in PROG_DEPTH).
- At least one of the Phase 3 tiling shapes (e.g. 256x256) MUST fit
  at `vu13p_uram` (PROG_DEPTH=131072). Without this, the URAM-class
  config is also vapourware.

These are sanity floors, not aspirational targets — if they fail, the
audit silently regressed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "bench" / "results" / "board_fit_audit.json"
)


def _load() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip("bench/results/board_fit_audit.json not present; "
                    "regenerate with `python firmware/host/run_board_fit_audit.py`")
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_artifact_top_level_schema():
    art = _load()
    for k in ("generated_at_utc", "phase", "scope", "methodology", "per_shape", "aggregate"):
        assert k in art, f"board_fit_audit.json missing top-level '{k}'"
    assert art["phase"] == "phase_7_remediation_p3_board_fit_audit"


def test_methodology_required_fields():
    art = _load()
    m = art["methodology"]
    for k in ("api", "what_it_measures", "fit_criterion", "boards", "shape_source", "lowering_api", "notes"):
        assert k in m
    board_names = {b["name"] for b in m["boards"]}
    assert {"pynqz2_baseline", "pynqz2_bram_max", "vu13p_uram"} <= board_names, (
        f"reference board set incomplete: {board_names}"
    )
    for b in m["boards"]:
        for k in ("name", "prog_depth", "buffer_size", "array_size", "data_width_bits", "notes"):
            assert k in b
        assert int(b["prog_depth"]) > 0


def test_per_shape_layout_locked():
    art = _load()
    assert art["per_shape"], "audit must report at least one shape"
    for row in art["per_shape"]:
        for k in ("shape", "tag", "program_instruction_words", "block_ops",
                  "out_blocks", "in_blocks", "fits_per_board"):
            assert k in row
        for k in ("out_features", "in_features", "array_size"):
            assert k in row["shape"]
        for board_name in ("pynqz2_baseline", "pynqz2_bram_max", "vu13p_uram"):
            assert board_name in row["fits_per_board"]
            assert isinstance(row["fits_per_board"][board_name], bool)
        assert int(row["program_instruction_words"]) > 0


def test_aggregate_per_board_locked():
    art = _load()
    agg = art["aggregate"]
    assert "shape_count" in agg and agg["shape_count"] == len(art["per_shape"])
    for board_name in ("pynqz2_baseline", "pynqz2_bram_max", "vu13p_uram"):
        assert board_name in agg["per_board"]
        info = agg["per_board"][board_name]
        for k in ("prog_depth", "shapes_fit_count", "shapes_fit", "shapes_fit_fraction"):
            assert k in info
        assert int(info["shapes_fit_count"]) == len(info["shapes_fit"])


def test_pynqz2_baseline_has_at_least_one_fitting_shape():
    """Reality floor: the smallest shape (16x16) MUST fit the shipping
    PROG_DEPTH=1024 bitstream, otherwise "board execution" is
    structurally blocked on the shipping config."""
    art = _load()
    baseline = art["aggregate"]["per_board"]["pynqz2_baseline"]
    assert baseline["shapes_fit_count"] >= 1, (
        "pynqz2_baseline (PROG_DEPTH=1024) admits zero shapes -- "
        "without at least one fitting shape there is no demo path "
        "for board bring-up on the shipping bitstream"
    )
    fit_shapes = {(s["out_features"], s["in_features"]) for s in baseline["shapes_fit"]}
    assert (16, 16) in fit_shapes, (
        "the canonical smallest shape (16, 16) does not fit the "
        "pynqz2_baseline PROG_DEPTH=1024; if this fails, either the "
        "lowering grew or PROG_DEPTH was reduced -- regression."
    )


def test_fit_is_monotone_in_prog_depth():
    """If shape S fits at board B with prog_depth=D, then S must also
    fit at every board with prog_depth > D. Catches accidentally
    swapping board entries in `aggregate.per_board`.
    """
    art = _load()
    boards_sorted = sorted(
        art["aggregate"]["per_board"].items(),
        key=lambda kv: kv[1]["prog_depth"],
    )
    fits_by_board: dict[str, set] = {}
    for row in art["per_shape"]:
        for board_name, fits in row["fits_per_board"].items():
            fits_by_board.setdefault(board_name, set())
            if fits:
                fits_by_board[board_name].add(
                    (row["shape"]["out_features"], row["shape"]["in_features"])
                )
    for i in range(len(boards_sorted) - 1):
        smaller_name = boards_sorted[i][0]
        larger_name = boards_sorted[i + 1][0]
        assert fits_by_board[smaller_name] <= fits_by_board[larger_name], (
            f"non-monotone: shapes fitting {smaller_name} "
            f"(prog_depth={boards_sorted[i][1]['prog_depth']}) are not a "
            f"subset of those fitting {larger_name} "
            f"(prog_depth={boards_sorted[i + 1][1]['prog_depth']})"
        )


def test_uram_board_admits_typical_phase3_shape():
    """Reality floor: the URAM-class config must admit at least one
    Phase 3 tiling shape (256x256 is the smallest one that the
    `pynqz2_baseline` cannot accommodate). Without this, the
    "scale-up" board config is itself vapourware.
    """
    art = _load()
    uram = art["aggregate"]["per_board"]["vu13p_uram"]
    fit_shapes = {(s["out_features"], s["in_features"]) for s in uram["shapes_fit"]}
    assert (256, 256) in fit_shapes, (
        "vu13p_uram (PROG_DEPTH=131072) does not admit (256, 256); "
        "either the URAM config is too small or the shape grid was "
        "trimmed -- regression."
    )
