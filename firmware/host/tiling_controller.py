"""Phase 3: tiling controller for `Linear(M, N)` onto a fixed on-chip
buffer.

What this module does
---------------------
The current uTPU blocked-FC lowering (`lower_blocked_fc_program_utpu`)
already streams K-dim weight/input blocks through small scratch
windows in the unified buffer, so the K dimension is implicitly tiled
to the systolic-array `array_size`. What it does NOT tile is the
**output (M) dimension**: every output block reserves an `(array_size //
4)`-word slot in the result-staging region, and the final `RUN
quantize` writes `(array_size * array_size) // 4` words from that slot.
For large `out_features` the result-staging region overruns
`BUFFER_SIZE`.

This module adds a buffer-capacity-parameterized tiling pass that
partitions the M dimension into output tiles small enough to fit in
the unified buffer, lowers each tile as an independent
`lower_blocked_fc_program_utpu` program, runs each program through the
ISA simulator, and stitches per-tile INT4 outputs back into a single
output vector — equivalent to the un-tiled NumPy oracle.

Buffer-fit math (must match `lower_blocked_fc_program_utpu` +
`UTPUISASimulator._run_finalize` exactly)
-----------------------------------------
Per-tile address layout (compatible with the existing lowering's
defaults: `weight_addr=0, input_addr=WS, result_addr=WS+IS`):

    WS = (array_size * array_size) // 4   # weight scratch / finalize footprint
    IS = array_size // 4                   # input scratch words
    PB = array_size // 4                   # per-output-block result stride

The simulator's `RUN quantize` writes `WS` words from `result_addr +
(ob * PB)` for `ob` in `[0, out_blocks)`. The last write's
final address must satisfy:

    result_addr + (out_blocks - 1) * PB + WS - 1  <  buffer_capacity_words

Substituting `result_addr = WS + IS`:

    2 * WS + IS + (out_blocks - 1) * PB  <=  buffer_capacity_words

Solving for the maximum number of output blocks per tile:

    max_out_blocks_per_tile = (buffer_capacity_words - 2*WS - IS) // PB + 1
    max_m_per_tile          = max_out_blocks_per_tile * array_size

Tile-size selection policies
----------------------------
* ``max_fit_heuristic`` (default): largest legal `m_tile` (= a multiple
  of `array_size`) up to `out_features`. Minimises tile count, which
  amortises per-tile setup overhead. Deterministic.
* ``cost_model_select``: enumerate legal candidates
  ``{array_size * k : 1 <= k <= max_out_blocks_per_tile}`` and hand
  them to the Phase 1 ``cost_model.select`` API for the predicted
  schedule winner. **TODO/VERIFY:** the Phase 1 cost model is
  calibrated on CUDA blocked-FC schedules, not uTPU tile cycles. The
  current implementation uses the same analytical cycle estimator that
  ``max_fit_heuristic`` reports for `estimated_total_cycles`, but
  surfaces the predicted choice via the cost-model API so the
  selection plumbing is exercised end-to-end. Re-calibration on uTPU
  tile measurements is a Phase 5+ task.

This module does NOT modify RTL; it does NOT widen the unified buffer;
it does NOT claim on-board execution. The verification path is the
existing Python ISA simulator (`isa_simulator.simulate_program_bytes`)
against a NumPy oracle. All claims are sim-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu


# ---------------------------------------------------------------------------
# Buffer-capacity model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferCapacityModel:
    """Per-tile buffer footprint model, locked to the
    `lower_blocked_fc_program_utpu` storage layout and the
    `UTPUISASimulator._run_finalize` write pattern. Tiles produced by
    `plan_linear_tiling` using this model are guaranteed (by
    construction, exercised by tests) to lower without a
    `buffer address out of range` violation.
    """

    buffer_capacity_words: int
    array_size: int
    int4_per_word: int = 4

    def __post_init__(self) -> None:
        if self.array_size <= 0 or self.array_size % self.int4_per_word != 0:
            raise ValueError(
                f"array_size must be a positive multiple of {self.int4_per_word}; "
                f"got {self.array_size}"
            )
        if self.buffer_capacity_words <= 0:
            raise ValueError("buffer_capacity_words must be positive")

    @property
    def weight_scratch_words(self) -> int:
        return (self.array_size * self.array_size) // self.int4_per_word

    @property
    def input_scratch_words(self) -> int:
        return self.array_size // self.int4_per_word

    @property
    def per_output_block_stride_words(self) -> int:
        return self.array_size // self.int4_per_word

    @property
    def finalize_footprint_words(self) -> int:
        return (self.array_size * self.array_size) // self.int4_per_word

    @property
    def default_layout(self) -> Dict[str, int]:
        weight_addr = 0
        input_addr = weight_addr + self.weight_scratch_words
        result_addr = input_addr + self.input_scratch_words
        return {
            "weight_addr": weight_addr,
            "input_addr": input_addr,
            "result_addr": result_addr,
        }

    def max_out_blocks_per_tile(self) -> int:
        ws = self.weight_scratch_words
        is_ = self.input_scratch_words
        pb = self.per_output_block_stride_words
        budget = self.buffer_capacity_words - 2 * ws - is_
        if budget < 0:
            return 0
        return budget // pb + 1

    def max_m_per_tile(self) -> int:
        return self.max_out_blocks_per_tile() * self.array_size

    def peak_words_for_tile(self, m_tile: int) -> int:
        """Peak buffer-word index touched while executing a tile with
        `m_tile` output features. Equals `last_write_end + 1`."""
        out_blocks = math.ceil(m_tile / self.array_size)
        layout = self.default_layout
        last_write_end = (
            layout["result_addr"]
            + (out_blocks - 1) * self.per_output_block_stride_words
            + self.finalize_footprint_words
        )
        return int(last_write_end)


# ---------------------------------------------------------------------------
# Cost estimate (used by both heuristic and cost-model paths)
# ---------------------------------------------------------------------------


def _estimate_tile_cycles(
    m_tile: int,
    in_features: int,
    array_size: int,
) -> Dict[str, int]:
    """Analytical per-tile cycle estimate that mirrors the structure of
    `lower_blocked_fc_program_utpu`. Per-tile work:

    * For each output block (out_blocks_in_tile = ceil(m_tile / array_size))
      * For each K block (in_blocks = ceil(in_features / array_size))
        * Store weight tile     (weight_scratch_words STOREs)
        * LOAD_WEIGHTS          (1 cycle)
        * Store input tile      (input_scratch_words STOREs)
        * LOAD_INPUTS           (1 cycle)
        * RUN compute (acc)     (1 cycle)
      * RUN finalize (q+relu)   (1 cycle)
      * 2 * (array_size // 4)   FETCHes

    The simulator costs every store/load/run/fetch as 1 cycle each, so
    this is a faithful, deterministic upper bound on the simulator's
    `cycle_count_sequential`.
    """
    out_blocks_in_tile = math.ceil(m_tile / array_size)
    in_blocks = math.ceil(in_features / array_size)
    weight_scratch_words = (array_size * array_size) // 4
    input_scratch_words = array_size // 4
    per_k_cycles = (
        weight_scratch_words  # store weight tile
        + 1                   # loadWeights
        + input_scratch_words # store input tile
        + 1                   # loadInputs
        + 1                   # run compute
    )
    per_ob_cycles = in_blocks * per_k_cycles + 1 + 2 * (array_size // 4)
    return {
        "out_blocks_in_tile": int(out_blocks_in_tile),
        "in_blocks": int(in_blocks),
        "per_tile_cycles": int(out_blocks_in_tile * per_ob_cycles),
    }


# ---------------------------------------------------------------------------
# Plan + planner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TilingPlan:
    layer_name: str
    out_features: int
    in_features: int
    array_size: int
    buffer_capacity_words: int
    m_tile: int
    num_m_tiles: int
    tile_partition: Tuple[Tuple[int, int], ...]
    in_blocks_per_tile: int
    peak_buffer_words_per_tile: int
    fits_in_buffer: bool
    policy: str
    estimated_per_tile_cycles: int
    estimated_total_cycles: int
    candidates_considered: int
    cost_model_provenance: Optional[Dict[str, Any]] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "array_size": self.array_size,
            "buffer_capacity_words": self.buffer_capacity_words,
            "m_tile": self.m_tile,
            "num_m_tiles": self.num_m_tiles,
            "tile_partition": [list(t) for t in self.tile_partition],
            "in_blocks_per_tile": self.in_blocks_per_tile,
            "peak_buffer_words_per_tile": self.peak_buffer_words_per_tile,
            "fits_in_buffer": self.fits_in_buffer,
            "policy": self.policy,
            "estimated_per_tile_cycles": self.estimated_per_tile_cycles,
            "estimated_total_cycles": self.estimated_total_cycles,
            "candidates_considered": self.candidates_considered,
            "cost_model_provenance": self.cost_model_provenance,
            "notes": self.notes,
        }


class TilingError(ValueError):
    """Raised when no legal tiling exists for the requested layer on the
    requested buffer."""


def _legal_candidate_m_tiles(
    out_features: int, capacity: BufferCapacityModel
) -> List[int]:
    """Multiples of `array_size` from `array_size` to
    `min(out_features padded, max_m_per_tile)`."""
    array_size = capacity.array_size
    max_m_fit = capacity.max_m_per_tile()
    if max_m_fit <= 0:
        return []
    out_padded = math.ceil(out_features / array_size) * array_size
    cap = min(max_m_fit, out_padded)
    candidates: List[int] = []
    m = array_size
    while m <= cap:
        candidates.append(m)
        m += array_size
    return candidates


def _partition_for(m_tile: int, out_features: int) -> Tuple[Tuple[int, int], ...]:
    parts: List[Tuple[int, int]] = []
    start = 0
    while start < out_features:
        end = min(start + m_tile, out_features)
        parts.append((start, end))
        start = end
    return tuple(parts)


def plan_linear_tiling(
    *,
    out_features: int,
    in_features: int,
    array_size: int = 16,
    buffer_capacity_words: int = 512,
    policy: str = "max_fit_heuristic",
    layer_name: str = "linear",
    cost_model_selector: Optional[
        Callable[[List[Dict[str, Any]], Dict[str, Any]], Dict[str, Any]]
    ] = None,
) -> TilingPlan:
    """Plan an `out_features x in_features` Linear layer onto a unified
    buffer with `buffer_capacity_words` 16-bit words.

    Parameters
    ----------
    policy
        ``"max_fit_heuristic"`` (default) — pick the largest legal
        `m_tile`. ``"cost_model_select"`` — hand the legal candidate
        list to the Phase 1 ``cost_model.select`` API via
        ``cost_model_selector`` (or the default, which lazily imports
        ``cost_model.select``).
    cost_model_selector
        Optional injection point for tests. Signature:
        ``(candidates, shape) -> chosen_candidate_dict``. Each
        candidate carries ``{"m_tile": int, "predicted_cycles": int,
        "tiles": int}``.

    Raises
    ------
    TilingError
        If `buffer_capacity_words` is too small to fit even a single
        `array_size`-sized output tile.
    """
    if policy not in {"max_fit_heuristic", "cost_model_select"}:
        raise ValueError(
            f"unknown policy '{policy}'; expected 'max_fit_heuristic' or 'cost_model_select'"
        )

    capacity = BufferCapacityModel(
        buffer_capacity_words=buffer_capacity_words, array_size=array_size
    )
    candidates_m = _legal_candidate_m_tiles(out_features, capacity)
    if not candidates_m:
        raise TilingError(
            f"buffer_capacity_words={buffer_capacity_words} is too small for "
            f"array_size={array_size}: need at least "
            f"{2 * capacity.weight_scratch_words + capacity.input_scratch_words} words "
            "for one output block."
        )

    candidate_summaries: List[Dict[str, Any]] = []
    for m in candidates_m:
        est = _estimate_tile_cycles(m_tile=m, in_features=in_features, array_size=array_size)
        tiles = math.ceil(out_features / m)
        candidate_summaries.append(
            {
                "m_tile": int(m),
                "predicted_per_tile_cycles": int(est["per_tile_cycles"]),
                "tiles": int(tiles),
                "predicted_total_cycles": int(est["per_tile_cycles"] * tiles),
                "peak_buffer_words_per_tile": int(capacity.peak_words_for_tile(m)),
            }
        )

    cost_model_provenance: Optional[Dict[str, Any]] = None
    if policy == "max_fit_heuristic":
        chosen = max(candidate_summaries, key=lambda c: c["m_tile"])
        notes = "max-fit heuristic: largest m_tile that fits the buffer"
    else:
        # cost_model_select: hand candidates to a selector. The default
        # selector lazily imports cost_model.select and adapts the
        # candidate schema. Either way, ties / cost-model failure falls
        # back to the lowest predicted_total_cycles (deterministic).
        chosen, cost_model_provenance = _select_via_cost_model(
            candidates=candidate_summaries,
            layer_shape={
                "in_features": in_features,
                "out_features": out_features,
                "array_size": array_size,
            },
            selector=cost_model_selector,
        )
        notes = (
            "cost_model_select: candidates handed to Phase 1 cost_model.select; "
            "TODO/VERIFY: Phase 1 cost model is calibrated on CUDA blocked-FC, "
            "not uTPU tile cycles. Selection plumbing is end-to-end exercised "
            "but the chosen m_tile is currently equivalent to lowest analytical "
            "per-tile-cycle estimate. Re-calibration is a Phase 5+ task."
        )

    m_tile_chosen = int(chosen["m_tile"])
    partition = _partition_for(m_tile_chosen, out_features)
    per_tile_cycles = int(chosen["predicted_per_tile_cycles"])
    return TilingPlan(
        layer_name=layer_name,
        out_features=int(out_features),
        in_features=int(in_features),
        array_size=int(array_size),
        buffer_capacity_words=int(buffer_capacity_words),
        m_tile=m_tile_chosen,
        num_m_tiles=len(partition),
        tile_partition=partition,
        in_blocks_per_tile=int(math.ceil(in_features / array_size)),
        peak_buffer_words_per_tile=int(capacity.peak_words_for_tile(m_tile_chosen)),
        fits_in_buffer=bool(capacity.peak_words_for_tile(m_tile_chosen) <= buffer_capacity_words),
        policy=policy,
        estimated_per_tile_cycles=per_tile_cycles,
        estimated_total_cycles=int(per_tile_cycles * len(partition)),
        candidates_considered=len(candidate_summaries),
        cost_model_provenance=cost_model_provenance,
        notes=notes,
    )


def _select_via_cost_model(
    *,
    candidates: List[Dict[str, Any]],
    layer_shape: Dict[str, Any],
    selector: Optional[
        Callable[[List[Dict[str, Any]], Dict[str, Any]], Dict[str, Any]]
    ],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Adapter from the local candidate schema to ``cost_model.select``.

    Returns ``(chosen_candidate, provenance)``. If the cost model is
    unavailable or raises, falls back deterministically to lowest
    ``predicted_total_cycles``.
    """
    fallback = min(candidates, key=lambda c: (c["predicted_total_cycles"], c["m_tile"]))
    if selector is not None:
        chosen = selector(candidates, layer_shape)
        return chosen, {
            "selector": "user_injected",
            "fell_back_to_heuristic": False,
        }
    try:
        from cost_model import select as cost_model_select

        # cost_model.select expects {shape}, [{schedule_dict}], target.
        # We hand it a per-candidate schedule dict carrying m_tile +
        # tiles + a fabricated `threads_per_block`/`unroll_factor` so the
        # Phase 1 analytical model can score it; the chosen m_tile is
        # what we keep.
        cm_candidates = [
            {
                "m_tile": c["m_tile"],
                "threads_per_block": 128,
                "unroll_factor": 1,
                "tiles": c["tiles"],
            }
            for c in candidates
        ]
        choice = cost_model_select(
            shape={
                "in_features": layer_shape["in_features"],
                "out_features": layer_shape["out_features"],
            },
            candidates=cm_candidates,
            target="cuda",
        )
        chosen_m = int(choice.schedule["m_tile"])
        chosen = next(c for c in candidates if c["m_tile"] == chosen_m)
        return chosen, {
            "selector": "cost_model.select",
            "predicted_latency_us": float(choice.predicted_latency_us),
            "rank": int(choice.rank),
            "candidates_considered": int(choice.candidates_considered),
            "margin_pct": float(choice.margin_pct),
            "confidence": float(choice.confidence),
            "fell_back_to_heuristic": False,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return fallback, {
            "selector": "fallback_lowest_total_cycles",
            "fell_back_to_heuristic": True,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Tiled execution: lower each tile, simulate, stitch INT4 outputs
# ---------------------------------------------------------------------------


def _decode_tile_outputs_from_fetch_bytes(
    fetch_bytes: List[int], m_tile: int, array_size: int
) -> np.ndarray:
    """Inverse of `_pack_int4` + the fetch ordering in
    `lower_blocked_fc_program_utpu`. Per output block we read
    `2 * (array_size // 4)` bytes: pairs of (low_byte, high_byte) of
    `array_size // 4` consecutive 16-bit words; each word packs 4 INT4
    nibbles (low nibble = output[widx*4 + 0]).
    """
    out_blocks = math.ceil(m_tile / array_size)
    words_per_block = array_size // 4
    bytes_per_block = 2 * words_per_block
    expected = out_blocks * bytes_per_block
    if len(fetch_bytes) != expected:
        raise ValueError(
            f"unexpected fetch_byte count: got {len(fetch_bytes)}, expected {expected} "
            f"(out_blocks={out_blocks}, words_per_block={words_per_block})"
        )
    outputs = np.zeros(out_blocks * array_size, dtype=np.int32)
    idx = 0
    for ob in range(out_blocks):
        block_outs: List[int] = []
        for widx in range(words_per_block):
            lo = fetch_bytes[idx]
            hi = fetch_bytes[idx + 1]
            idx += 2
            word = (hi << 8) | lo
            for i in range(4):
                nibble = (word >> (4 * i)) & 0xF
                value = nibble - 16 if nibble >= 8 else nibble
                block_outs.append(value)
        outputs[ob * array_size : (ob + 1) * array_size] = block_outs
    return outputs[:m_tile]


@dataclass
class TileExecutionResult:
    tile_index: int
    m_start: int
    m_end: int
    program_words: int
    fits_instruction_bram: bool
    sim_cycles_sequential: int
    output_int4: np.ndarray  # shape (m_end - m_start,)
    fetch_bytes_count: int


@dataclass
class TiledExecutionResult:
    plan: TilingPlan
    per_tile: List[TileExecutionResult]
    output_int4: np.ndarray
    total_sim_cycles_sequential: int
    total_program_words: int


def execute_tiled_linear(
    *,
    plan: TilingPlan,
    weights_int4: np.ndarray,
    activations_int4: np.ndarray,
    apply_relu: bool,
    apply_quant: bool = True,
) -> TiledExecutionResult:
    """Lower each tile from `plan` to a uTPU ISA program, simulate it,
    decode the per-tile INT4 outputs, and concatenate.

    Requires `apply_quant=True` because the current lowering only
    finalises through the quantized buffer-output path. `apply_relu` is
    forwarded per-tile (the same flag is applied to every tile, which
    matches a single Linear+optional ReLU at the IR level).
    """
    if not apply_quant:
        raise ValueError(
            "execute_tiled_linear requires apply_quant=True (lowering finalises "
            "through the quantized output path)"
        )

    weights = np.asarray(weights_int4, dtype=np.int8)
    x = np.asarray(activations_int4, dtype=np.int8).flatten()
    if weights.shape != (plan.out_features, plan.in_features):
        raise ValueError(
            f"weights shape {weights.shape} does not match plan "
            f"({plan.out_features}, {plan.in_features})"
        )
    if x.shape[0] != plan.in_features:
        raise ValueError(
            f"activations length {x.shape[0]} does not match plan in_features "
            f"{plan.in_features}"
        )

    capacity = BufferCapacityModel(
        buffer_capacity_words=plan.buffer_capacity_words,
        array_size=plan.array_size,
    )
    layout = capacity.default_layout

    per_tile_results: List[TileExecutionResult] = []
    out_int4_concat = np.zeros(plan.out_features, dtype=np.int32)
    total_cycles = 0
    total_words = 0

    for tile_idx, (m0, m1) in enumerate(plan.tile_partition):
        w_tile = weights[m0:m1, :]
        lowered = lower_blocked_fc_program_utpu(
            weights_int4=w_tile,
            activations_int4=x,
            out_features=m1 - m0,
            in_features=plan.in_features,
            array_size=plan.array_size,
            apply_relu=apply_relu,
            apply_quant=apply_quant,
            weight_addr=layout["weight_addr"],
            input_addr=layout["input_addr"],
            result_addr=layout["result_addr"],
        )
        sim_result = simulate_program_bytes(
            lowered["program"],
            array_size=plan.array_size,
            buffer_size=plan.buffer_capacity_words,
        )
        if not sim_result.halted:
            raise RuntimeError(
                f"tile {tile_idx} simulation did not halt cleanly "
                f"(pc={sim_result.pc}, words={lowered['program_instruction_words']})"
            )
        tile_outputs = _decode_tile_outputs_from_fetch_bytes(
            sim_result.fetch_bytes, m_tile=m1 - m0, array_size=plan.array_size
        )
        out_int4_concat[m0:m1] = tile_outputs
        per_tile_results.append(
            TileExecutionResult(
                tile_index=tile_idx,
                m_start=int(m0),
                m_end=int(m1),
                program_words=int(lowered["program_instruction_words"]),
                fits_instruction_bram=bool(lowered["fits_instruction_bram"]),
                sim_cycles_sequential=int(sim_result.cycle_count_sequential),
                output_int4=tile_outputs,
                fetch_bytes_count=len(sim_result.fetch_bytes),
            )
        )
        total_cycles += int(sim_result.cycle_count_sequential)
        total_words += int(lowered["program_instruction_words"])

    return TiledExecutionResult(
        plan=plan,
        per_tile=per_tile_results,
        output_int4=out_int4_concat,
        total_sim_cycles_sequential=total_cycles,
        total_program_words=total_words,
    )


# ---------------------------------------------------------------------------
# NumPy oracle (un-tiled int32-accumulated reference)
# ---------------------------------------------------------------------------


def numpy_oracle_int4(
    *,
    weights_int4: np.ndarray,
    activations_int4: np.ndarray,
    apply_relu: bool,
    apply_quant: bool = True,
    alpha_shift: int = 2,
) -> np.ndarray:
    """Reference: full-precision int32 accumulation of `weights @ inputs`
    followed by the same `_run_finalize` quantize+leaky-relu rule the
    simulator uses (clip to int4, leaky_relu via right-shift on
    negatives). Matches `UTPUISASimulator._run_finalize` semantics
    exactly so a passing tiled-vs-oracle test isolates the tiling
    pass's correctness (not numerical drift in the oracle).
    """
    w = np.asarray(weights_int4, dtype=np.int32)
    x = np.asarray(activations_int4, dtype=np.int32).flatten()
    accum = (w @ x).astype(np.int64)

    if apply_quant:
        clipped = np.clip(accum, -8, 7).astype(np.int32)
    else:
        clipped = accum.astype(np.int32)

    if apply_relu:
        # _run_finalize: if q < 0: q = q >> alpha_shift  (arithmetic shift)
        out = np.where(clipped < 0, clipped >> alpha_shift, clipped)
    else:
        out = clipped
    return out.astype(np.int32)
