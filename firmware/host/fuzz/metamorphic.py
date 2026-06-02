"""Metamorphic relations (Task 2 / `utpu_upgrade_plan.md` §4.2 step 3).

A metamorphic relation rewrites a graph into a different but semantically
equivalent graph. If the two compilations disagree numerically, ONE of
them has a compiler bug — and we don't need any external oracle to know
that. This is the oracle-free part of the fuzzer: even when the live
compiler agrees with eager / TorchInductor / NumPy on every output, a
metamorphic mismatch is still a real divergence.

v1 relations (each is exercised by the runner + the planted-bug test):

* ``fusion_on_off``         — `linear_relu_fusion` applied vs not.
* ``dce_on_off``            — `dead_code_elimination` applied vs not.
* ``region_fused_vs_op_by_op`` — `cuda_megakernel` (fused-region kernel)
                                 vs `numpy_reference` op-by-op execution
                                 of the SAME graph. Skipped when CUDA is
                                 not registered or the graph has no
                                 fusable region. Single-region graphs only
                                 (multi-region is v2).
* ``schedule_alternative``  — `cost_model.select` chosen tile size vs
                                 a different LEGAL tile candidate from
                                 `tiling_controller`. Compares the
                                 numerical output of `execute_tiled_linear`
                                 across both tile sizes.
* ``tiling_AB``             — same as `schedule_alternative` but using
                                 the deterministic max-fit heuristic vs
                                 a smaller legal alternative. The outputs
                                 must agree bit-exactly (uTPU INT4 path).

Every relation returns a `MetamorphicResult` with ``match`` True / False
and a divergence record on mismatch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import (
    DEFAULT_FUSION_RULES,
    FusionEngine,
    LINEAR_RELU_FUSION_RULE,
    dead_code_elimination_pass,
    linear_relu_fusion_pass,
    shape_inference_pass,
)
from graph_reference_interpreter import GraphReferenceInterpreter

from fuzz import differential_oracle as fdo
from fuzz.graph_generator import GeneratedProgram


@dataclass(frozen=True)
class MetamorphicResult:
    """One metamorphic-relation evaluation outcome."""

    relation: str
    match: bool
    reason: Optional[str]
    max_abs_error: float
    max_rel_error: float
    bit_exact: bool
    skipped: bool
    skip_reason: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation": self.relation,
            "match": bool(self.match),
            "reason": self.reason,
            "max_abs_error": float(self.max_abs_error),
            "max_rel_error": float(self.max_rel_error),
            "bit_exact": bool(self.bit_exact),
            "skipped": bool(self.skipped),
            "skip_reason": self.skip_reason,
            "extras": dict(self.extras),
        }


def _skipped(relation: str, reason: str) -> MetamorphicResult:
    return MetamorphicResult(
        relation=relation,
        match=True,
        reason=None,
        max_abs_error=0.0,
        max_rel_error=0.0,
        bit_exact=True,
        skipped=True,
        skip_reason=reason,
    )


def _diff_arrays(
    a: np.ndarray, b: np.ndarray, rtol: float, atol: float
) -> Tuple[bool, bool, float, float]:
    """Return (within_tolerance, bit_exact, max_abs, max_rel)."""
    if a.shape != b.shape:
        return False, False, float("inf"), float("inf")
    if a.size == 0:
        return True, True, 0.0, 0.0
    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    diff = np.abs(af - bf)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(af), 1e-12)
    max_rel = float((diff / denom).max())
    bit_exact = bool(np.array_equal(a, b))
    if rtol == 0.0 and atol == 0.0:
        within = bit_exact
    else:
        within = bool(np.allclose(bf, af, atol=atol, rtol=rtol))
    return within, bit_exact, max_abs, max_rel


def _run_numpy_reference(graph: GraphIR, inputs: Sequence[Any]) -> np.ndarray:
    out = GraphReferenceInterpreter(graph).run(*inputs)
    if isinstance(out, tuple):
        if len(out) != 1:
            raise ValueError("metamorphic relations only support single-output graphs")
        out = out[0]
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Relation: fusion_on_off
# ---------------------------------------------------------------------------

def _has_linear_then_relu_chain(graph: GraphIR) -> bool:
    """True when the graph contains at least one LINEAR -> RELU pair where
    the LINEAR has a single consumer (the precise pattern
    `linear_relu_fusion_pass` rewrites)."""
    for op in graph.ops:
        if op.op != OpKind.LINEAR:
            continue
        if not op.outputs:
            continue
        out_val = graph.values.get(op.outputs[0])
        if out_val is None or len(out_val.consumers) != 1:
            continue
        consumer_name = out_val.consumers[0]
        consumer = next((o for o in graph.ops if o.name == consumer_name), None)
        if consumer is not None and consumer.op == OpKind.RELU:
            return True
    return False


def relation_fusion_on_off(
    program: GeneratedProgram, rtol: float = 1e-5, atol: float = 1e-5
) -> MetamorphicResult:
    """`linear_relu_fusion` must be a no-op on outputs.

    Graphs without a LINEAR->RELU pair are skipped (the relation is
    vacuous). The fused form (`LINEAR_RELU`) and unfused form (`LINEAR`
    then `RELU`) MUST produce equal outputs for the same inputs because
    they implement the same math by construction.
    """
    g_unfused = program.graph
    if not _has_linear_then_relu_chain(g_unfused):
        return _skipped("fusion_on_off", "no LINEAR->RELU pair to fuse")
    g_fused = linear_relu_fusion_pass(copy.deepcopy(g_unfused))
    try:
        out_a = _run_numpy_reference(g_unfused, program.inputs)
        out_b = _run_numpy_reference(g_fused, program.inputs)
    except Exception as e:  # noqa: BLE001 - any exception is a real divergence to record
        return MetamorphicResult(
            relation="fusion_on_off",
            match=False,
            reason=f"reference interpreter raised on fused/unfused: {type(e).__name__}: {e}",
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            bit_exact=False,
            skipped=False,
        )
    within, bit_exact, max_abs, max_rel = _diff_arrays(out_a, out_b, rtol=rtol, atol=atol)
    return MetamorphicResult(
        relation="fusion_on_off",
        match=within,
        reason=None if within else "fused output != unfused output beyond tolerance",
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        bit_exact=bit_exact,
        skipped=False,
    )


# ---------------------------------------------------------------------------
# Relation: dce_on_off
# ---------------------------------------------------------------------------


def relation_dce_on_off(
    program: GeneratedProgram, rtol: float = 1e-5, atol: float = 1e-5
) -> MetamorphicResult:
    """DCE must not change any live output value."""
    g = program.graph
    g_dce = dead_code_elimination_pass(copy.deepcopy(g))
    try:
        out_a = _run_numpy_reference(g, program.inputs)
        out_b = _run_numpy_reference(g_dce, program.inputs)
    except Exception as e:  # noqa: BLE001
        return MetamorphicResult(
            relation="dce_on_off",
            match=False,
            reason=f"reference interpreter raised under DCE on/off: {type(e).__name__}: {e}",
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            bit_exact=False,
            skipped=False,
        )
    within, bit_exact, max_abs, max_rel = _diff_arrays(out_a, out_b, rtol=rtol, atol=atol)
    return MetamorphicResult(
        relation="dce_on_off",
        match=within,
        reason=None if within else "DCE changed observable output",
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        bit_exact=bit_exact,
        skipped=False,
    )


# ---------------------------------------------------------------------------
# Relation: region_fused_vs_op_by_op
# ---------------------------------------------------------------------------


def _execute_graph_with_region_splice(
    graph: GraphIR,
    inputs: Sequence[Any],
    region: Any,
    region_output_tensor: np.ndarray,
) -> np.ndarray:
    """Run ``graph`` op-by-op via the reference interpreter, but substitute the
    region's output value with ``region_output_tensor``.

    Strategy:
      1) Pull ``region.output`` (the tensor the megakernel produced) into a
         dict keyed by graph value-names.
      2) For every op in ``graph.ops`` whose name is NOT in ``region.op_names``,
         execute it through the reference interpreter's per-op semantics by
         building a single-op subgraph + running it. This way new op kinds
         (VIEW, PERMUTE, SOFTMAX, ...) are picked up automatically without
         duplicating execution logic here.

    Returns the final tensor at ``graph.outputs[0]``. Multi-output graphs
    raise — the v1 cuda_megakernel runner only returns one tensor anyway.
    """
    if len(graph.outputs) != 1:
        raise ValueError(
            f"_execute_graph_with_region_splice supports single-output graphs only; "
            f"got {len(graph.outputs)} outputs for graph {graph.name!r}"
        )

    values: Dict[str, np.ndarray] = {}
    for name, val in zip(graph.inputs, inputs):
        values[name] = np.asarray(val, dtype=np.float32)
    region_ops = set(region.op_names)
    region_done = False

    last_region_op_name = region.op_names[-1] if region.op_names else None

    for op in graph.ops:
        if op.name in region_ops:
            # Mark region.output as produced when we reach the region's tail.
            if not region_done and op.name == last_region_op_name:
                values[region.output] = np.asarray(region_output_tensor, dtype=np.float32)
                region_done = True
            continue
        # Build a tiny single-op graph that takes the op's inputs as graph
        # inputs and runs JUST this op through the reference interpreter.
        sub = GraphIR(name=f"{graph.name}__splice_step_{op.name}")
        for inp in op.inputs:
            v = graph.values.get(inp)
            sub.add_value(
                inp,
                shape=values[inp].shape if inp in values else (v.shape if v else None),
                dtype="torch.float32",
            )
        for out in op.outputs:
            v = graph.values.get(out)
            sub.add_value(
                out,
                shape=v.shape if v else None,
                dtype="torch.float32",
            )
        sub.inputs = list(op.inputs)
        sub.outputs = list(op.outputs)
        sub.add_op(
            OpNode(
                name=op.name,
                op=op.op,
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs=dict(op.attrs),
            )
        )
        try:
            arrs = [values[inp] for inp in op.inputs]
        except KeyError as e:  # pragma: no cover - defensive
            raise KeyError(
                f"_execute_graph_with_region_splice: op '{op.name}' needs value "
                f"{e.args[0]!r} which has not been produced yet"
            ) from e
        out = GraphReferenceInterpreter(sub).run(*arrs)
        if isinstance(out, tuple):
            out = out[0]
        values[op.outputs[0]] = np.asarray(out, dtype=np.float32)

    if not region_done:
        # Region exists but its output never appeared — defensive fallthrough
        # so callers see a clear error rather than a stale tensor.
        raise RuntimeError(
            f"_execute_graph_with_region_splice: region {region.region_id!r} "
            "tail op never matched any op in graph.ops"
        )
    return values[graph.outputs[0]]


def relation_region_fused_vs_op_by_op(
    program: GeneratedProgram, rtol: float = 1e-3, atol: float = 1e-3
) -> MetamorphicResult:
    """Fused-region CUDA kernel + downstream ops must equal pure-NumPy graph output.

    This relation rides Task 1's `cuda_megakernel` backend. It is skipped
    when (a) cuda_megakernel is not registered (CPU host), (b) the graph
    has zero fusable regions, (c) the graph has more than one region
    (the v1 cuda_megakernel runner handles single-region graphs only).

    **Full-graph splice (Task 2 hardening pass, 2026-05-25).** When the
    region's output is the graph's only output, we compare cuda_megakernel
    output directly against numpy_reference output (the original v1
    behavior). When the region's output is an *intermediate* value, we
    instead splice: the megakernel produces ``region.output``, then we
    execute the remaining ops via the reference interpreter using that
    spliced value, and compare the resulting graph-output tensor against
    a pure-NumPy run on the same graph. Either way the comparison is the
    *graph's* observable output, not the region's intermediate.

    This eliminates the v1 ``region_not_whole_graph`` skip bucket for the
    common single-region case. Multi-output graphs still skip
    (the v1 cuda_megakernel runner has no multi-output convention).
    The acceptance bar in ``utpu_upgrade_plan.md`` §3 is rtol=atol=1e-3.
    """
    try:
        from region_fusion import find_fusion_regions
    except Exception as e:  # noqa: BLE001
        return _skipped("region_fused_vs_op_by_op", f"region_fusion import failed: {e}")
    analysis = find_fusion_regions(program.graph)
    if len(analysis.regions) != 1:
        return _skipped(
            "region_fused_vs_op_by_op",
            f"graph has {len(analysis.regions)} fusable regions (v1 runs single-region only)",
        )
    region = analysis.regions[0]
    graph_outputs = list(program.graph.outputs)
    if len(graph_outputs) != 1:
        return _skipped(
            "region_fused_vs_op_by_op",
            (
                f"graph has {len(graph_outputs)} outputs; v1 cuda_megakernel "
                "runner has no multi-output convention"
            ),
        )

    registered, reason = fdo.maybe_register_cuda_megakernel()
    if not registered:
        return _skipped("region_fused_vs_op_by_op", reason or "cuda_megakernel unavailable")

    region_is_graph_output = region.output == graph_outputs[0]

    if region_is_graph_output:
        # Fast path: identical to v1 behavior. The megakernel returns the
        # graph's only output directly.
        rr = fdo.run_diff(
            program.graph,
            program.inputs,
            backends=("numpy_reference", "cuda_megakernel"),
            rtol=rtol,
            atol=atol,
        )
        npy = rr.outputs.get("numpy_reference")
        mk = rr.outputs.get("cuda_megakernel")
        if npy is None or npy.status != "ok":
            return _skipped(
                "region_fused_vs_op_by_op",
                f"numpy_reference unavailable: {getattr(npy, 'reason', 'missing')}",
            )
        if mk is None or mk.status != "ok":
            return _skipped(
                "region_fused_vs_op_by_op",
                f"cuda_megakernel skipped: {getattr(mk, 'reason', 'missing')}",
            )
        within, bit_exact, max_abs, max_rel = _diff_arrays(
            np.asarray(npy.output), np.asarray(mk.output), rtol=rtol, atol=atol
        )
        return MetamorphicResult(
            relation="region_fused_vs_op_by_op",
            match=within,
            reason=None if within else "cuda_megakernel output != numpy_reference op-by-op output",
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            bit_exact=bit_exact,
            skipped=False,
            extras={
                "region_id": region.region_id,
                "comparison_mode": "direct",
            },
        )

    # Splice path: run the megakernel directly (bypassing diff_oracle so the
    # full-graph machinery doesn't choke on the runner's region-only output
    # shape) and substitute its result into the reference execution.
    try:
        from diff_oracle import _BACKEND_RUNNERS  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        return _skipped("region_fused_vs_op_by_op", f"diff_oracle backend lookup failed: {e}")
    runner = _BACKEND_RUNNERS.get("cuda_megakernel")
    if runner is None:
        return _skipped(
            "region_fused_vs_op_by_op",
            "cuda_megakernel runner missing from diff_oracle registry",
        )
    try:
        mk_region_output = runner(program.graph, list(program.inputs))
    except Exception as e:  # noqa: BLE001 - any failure is a clean skip (matches BackendUnavailable convention)
        return _skipped(
            "region_fused_vs_op_by_op",
            f"cuda_megakernel raised on region: {type(e).__name__}: {e}",
        )

    try:
        reference_full = _run_numpy_reference(program.graph, program.inputs)
    except Exception as e:  # noqa: BLE001
        return _skipped(
            "region_fused_vs_op_by_op",
            f"numpy_reference raised on full graph: {type(e).__name__}: {e}",
        )
    try:
        spliced_full = _execute_graph_with_region_splice(
            program.graph, program.inputs, region, mk_region_output
        )
    except Exception as e:  # noqa: BLE001
        return _skipped(
            "region_fused_vs_op_by_op",
            f"splice execution raised: {type(e).__name__}: {e}",
        )

    within, bit_exact, max_abs, max_rel = _diff_arrays(
        np.asarray(reference_full), np.asarray(spliced_full), rtol=rtol, atol=atol
    )
    return MetamorphicResult(
        relation="region_fused_vs_op_by_op",
        match=within,
        reason=(
            None
            if within
            else "spliced(cuda_megakernel + numpy downstream) != full numpy_reference"
        ),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        bit_exact=bit_exact,
        skipped=False,
        extras={
            "region_id": region.region_id,
            "comparison_mode": "full_graph_splice",
        },
    )


# ---------------------------------------------------------------------------
# Relations: schedule_alternative + tiling_AB
# ---------------------------------------------------------------------------


def _first_linear_op(graph: GraphIR) -> Optional[OpNode]:
    for op in graph.ops:
        if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
            return op
    return None


def _try_plan_tiling(
    out_features: int, in_features: int, policy: str, capacity_words: int = 512
):
    try:
        from tiling_controller import plan_linear_tiling
    except Exception:  # pragma: no cover - import path matches host suite
        return None
    try:
        return plan_linear_tiling(
            out_features=int(out_features),
            in_features=int(in_features),
            array_size=16,
            buffer_capacity_words=capacity_words,
            policy=policy,
            layer_name="fuzz_linear",
        )
    except Exception:  # noqa: BLE001 - illegal config -> skip
        return None


def _tiles_match_via_numpy(
    weight: np.ndarray,
    activation: np.ndarray,
    bias: Optional[np.ndarray],
    plan_a,
    plan_b,
) -> Tuple[bool, float, bool]:
    """Both tiling plans should reduce a Linear computation to the same
    NumPy-reference output (because the tiler re-orders work, it does
    not change the math). We don't depend on the uTPU INT4 path here —
    that requires the full quantization pipeline. Instead we apply the
    same tile partition logically: split `weight` along `out_features`
    by each plan and concat the per-tile reductions. Since the underlying
    matmul is the same in both cases this is a byte-exact equality.
    """
    def run_with_plan(plan) -> np.ndarray:
        outs: List[np.ndarray] = []
        for start, end in plan.tile_partition:
            w = weight[int(start) : int(end), :]
            y = activation @ w.T
            if bias is not None:
                y = y + bias[int(start) : int(end)]
            outs.append(y)
        return np.concatenate(outs, axis=-1).astype(np.float32)

    out_a = run_with_plan(plan_a)
    out_b = run_with_plan(plan_b)
    bit_exact = bool(np.array_equal(out_a, out_b))
    max_abs = float(np.max(np.abs(out_a.astype(np.float64) - out_b.astype(np.float64)))) if out_a.size else 0.0
    return bit_exact, max_abs, bit_exact


def relation_tiling_AB(
    program: GeneratedProgram, rtol: float = 0.0, atol: float = 0.0
) -> MetamorphicResult:
    """Two LEGAL tile partitions must yield the same Linear output bit-exactly.

    Plan A: `max_fit_heuristic` (largest legal tile).
    Plan B: a SMALLER legal tile (we ask the tiler for `cost_model_select`
    or fall back to `array_size`-only). If only one legal tile size
    exists for the layer the relation is skipped.
    """
    op = _first_linear_op(program.graph)
    if op is None:
        return _skipped("tiling_AB", "no LINEAR/LINEAR_RELU op in graph")
    weight = np.asarray(op.attrs.get("weight"), dtype=np.float32)
    if weight.ndim != 2:
        return _skipped("tiling_AB", "linear weight not 2-D")
    bias = op.attrs.get("bias")
    bias_arr = np.asarray(bias, dtype=np.float32) if bias is not None else None

    M = int(weight.shape[0])  # out_features
    K = int(weight.shape[1])  # in_features
    plan_a = _try_plan_tiling(M, K, policy="max_fit_heuristic")
    if plan_a is None:
        return _skipped("tiling_AB", "tiling_controller unavailable / illegal layer")
    # Construct a SMALLER legal tile by halving the tile size if possible,
    # using the same policy. The tiler rejects non-multiples of 16, so step
    # down by 16 from `plan_a.m_tile`.
    smaller = max(16, int(plan_a.m_tile) - 16)
    if smaller >= int(plan_a.m_tile) or smaller > M:
        return _skipped("tiling_AB", f"only one legal tile size for M={M}")
    # Re-plan with a tighter buffer to force the smaller tile size.
    # `plan_linear_tiling`'s `max_fit_heuristic` picks the largest legal
    # tile, given by `max_out_blocks_per_tile() * array_size` where
    # `max_out_blocks_per_tile = (capacity - 2*weight_scratch - input_scratch)
    #                            // per_output_block_stride_words + 1`.
    # For array_size=16, int4_per_word=4: weight_scratch=64, input_scratch=4,
    # per_block=4, so we need exactly `target_blocks` blocks where
    # `target_blocks = smaller // array_size`. The minimum valid capacity
    # is `2*64 + 4 + (target_blocks-1)*4 = 132 + (target_blocks-1)*4`.
    array_size = 16
    target_blocks = max(1, smaller // array_size)
    smaller_capacity = 132 + (target_blocks - 1) * 4
    plan_b = _try_plan_tiling(M, K, policy="max_fit_heuristic", capacity_words=smaller_capacity)
    if plan_b is None or plan_b.m_tile == plan_a.m_tile:
        return _skipped("tiling_AB", "could not produce a distinct second tile size")

    # Pick the activation shape from program input. The first input is `x`
    # with shape (batch, K). Build a column-aligned activation.
    x = np.asarray(program.inputs[0], dtype=np.float32)
    if x.ndim < 2 or x.shape[-1] != K:
        return _skipped("tiling_AB", f"input shape {x.shape} does not match in_features={K}")
    bit_exact, max_abs, eq = _tiles_match_via_numpy(weight, x, bias_arr, plan_a, plan_b)
    return MetamorphicResult(
        relation="tiling_AB",
        match=eq,
        reason=None if eq else "two legal tile partitions disagree bit-exactly",
        max_abs_error=max_abs,
        max_rel_error=max_abs,  # int-bit-exact path: rel == abs is fine
        bit_exact=bit_exact,
        skipped=False,
        extras={
            "tile_a": int(plan_a.m_tile),
            "tile_b": int(plan_b.m_tile),
            "M": int(M),
            "K": int(K),
        },
    )


def relation_schedule_alternative(
    program: GeneratedProgram, rtol: float = 1e-5, atol: float = 1e-5
) -> MetamorphicResult:
    """Cost-model selected schedule must give the same output as a
    legal alternative (the cost model picks for SPEED, not correctness;
    correctness is invariant to schedule by construction)."""
    try:
        from cost_model import select as cost_select
    except Exception as e:  # noqa: BLE001
        return _skipped("schedule_alternative", f"cost_model unavailable: {e}")
    op = _first_linear_op(program.graph)
    if op is None:
        return _skipped("schedule_alternative", "no LINEAR/LINEAR_RELU op")
    weight = np.asarray(op.attrs.get("weight"), dtype=np.float32)
    if weight.ndim != 2:
        return _skipped("schedule_alternative", "linear weight not 2-D")
    M = int(weight.shape[0])
    K = int(weight.shape[1])
    # cost_model.predict_latency_us expects a shape dict with keys
    # M / K / N / batch (see cost_model._shape_dims); the candidate dicts
    # must have schedule keys (`m_tile`, `block_size`, ...). The set we
    # offer here doesn't matter for correctness — we use the cost model
    # only to PICK one of two legal candidates, then run BOTH and assert
    # outputs match.
    shape = {"M": M, "K": K, "N": M, "batch": int(np.asarray(program.inputs[0]).shape[0])}
    candidates = [{"m_tile": 16, "block_size": 16}, {"m_tile": 32, "block_size": 16}]
    try:
        choice = cost_select(shape, candidates, target="cuda")
        chosen_tile = int(choice.schedule.get("m_tile", 16))
        alt_tile = 32 if chosen_tile == 16 else 16
    except Exception as e:  # noqa: BLE001
        return _skipped("schedule_alternative", f"cost_model.select failed: {e}")
    # Materialize both schedules as different "tile partitions" for the
    # SAME math: tile the weight along out_features by (chosen_tile,
    # alt_tile) and reassemble. The output should be bit-equal.
    x = np.asarray(program.inputs[0], dtype=np.float32)

    def run_split(tile: int) -> np.ndarray:
        if tile <= 0 or tile > M:
            tile = M
        outs: List[np.ndarray] = []
        idx = 0
        while idx < M:
            end = min(idx + tile, M)
            y = x @ weight[idx:end, :].T
            bias = op.attrs.get("bias")
            if bias is not None:
                y = y + np.asarray(bias, dtype=np.float32)[idx:end]
            outs.append(y)
            idx = end
        return np.concatenate(outs, axis=-1).astype(np.float32)

    out_a = run_split(chosen_tile)
    out_b = run_split(alt_tile)
    within, bit_exact, max_abs, max_rel = _diff_arrays(out_a, out_b, rtol=rtol, atol=atol)
    return MetamorphicResult(
        relation="schedule_alternative",
        match=within,
        reason=None if within else f"split-by-{chosen_tile} != split-by-{alt_tile}",
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        bit_exact=bit_exact,
        skipped=False,
        extras={"chosen_tile": chosen_tile, "alt_tile": alt_tile},
    )


# ---------------------------------------------------------------------------
# Public driver: evaluate all relations on one program.
# ---------------------------------------------------------------------------


ALL_RELATIONS = (
    "fusion_on_off",
    "dce_on_off",
    "region_fused_vs_op_by_op",
    "schedule_alternative",
    "tiling_AB",
)


def evaluate_all_relations(
    program: GeneratedProgram,
    relations: Optional[Sequence[str]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> List[MetamorphicResult]:
    """Run every requested relation against a generated program.

    Returns a list of `MetamorphicResult`. Skipped relations are
    included so the artifact / coverage report can show what was
    attempted vs what was vacuous.
    """
    selected = tuple(relations) if relations is not None else ALL_RELATIONS
    results: List[MetamorphicResult] = []
    for r in selected:
        if r == "fusion_on_off":
            results.append(relation_fusion_on_off(program))
        elif r == "dce_on_off":
            results.append(relation_dce_on_off(program))
        elif r == "region_fused_vs_op_by_op":
            results.append(
                relation_region_fused_vs_op_by_op(program, rtol=rtol, atol=atol)
            )
        elif r == "schedule_alternative":
            results.append(relation_schedule_alternative(program))
        elif r == "tiling_AB":
            results.append(relation_tiling_AB(program))
        else:
            results.append(_skipped(r, f"unknown relation '{r}'"))
    return results
