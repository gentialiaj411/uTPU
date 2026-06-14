"""Region formation for fused CUDA region kernels (Task 1 of `utpu_upgrade_plan.md`).

**Honest scope (read first).**

This is the IR-level planning pass for v1 fused CUDA region kernels. It does
NOT — and intentionally does not — fuse multi-Linear MLPs into one persistent
kernel. CUDA has no in-kernel grid-wide synchronization without explicit
cooperative-groups dispatch, so naively chaining a second matmul after a
first inside one kernel would let layer-2 threads read layer-1 outputs before
those outputs are globally produced. That is the classic "fake megakernel"
trap and this module REJECTS it with `rejection_kind="global_sync_required"`.

**What v1 legally fuses (each is safe by construction — no inter-thread sync
across CTAs is required, and within a thread the result is the standard
in-thread matmul with an epilogue):**

1. `linear_with_epilogue` — one LINEAR or LINEAR_RELU root, optionally
   chained with single-consumer RELU / ADD (residual) / SCALE ops that
   fold into the same kernel as a per-output-element epilogue. The
   per-thread accumulator is finished before the epilogue runs, so the
   epilogue sees only `acc` and externally-produced residual values.
2. `elementwise_chain` — a chain of two or more RELU / ADD / SCALE ops,
   each single-consumer. No reduction; all values are per-element
   independent.

**What v1 explicitly rejects (so the artifact / docs can never claim it):**

- `LINEAR -> LINEAR` or `LINEAR -> LINEAR_RELU` (the multi-Linear trap):
  `rejection_kind="global_sync_required"` **by default**. When the caller opts
  in with `find_fusion_regions(..., allow_single_cta_multilayer=True)` AND the
  entire intermediate activation provably fits one CTA's shared-memory budget,
  the chain is instead fused as a `single_cta_bounded_multilayer` region (a
  block-level `__syncthreads()` barrier replaces grid-wide sync). The flag is
  **default-off**, so legacy callers are byte-identical and the global-sync
  trap still fires for them. If the legality proof fails (hidden layer too
  wide), the candidate is rejected with `rejection_kind="single_cta_exceeds_shared_mem"`.
- Multi-consumer intermediate value: removing the intermediate from the
  materialized op stream would change semantics for the other consumer.
  `rejection_kind="multi_consumer"`.
- Fused intermediate value that is also a graph output: we'd be eliding a
  value the caller explicitly wants. `rejection_kind="intermediate_is_graph_output"`.
- ADD whose residual operand is produced inside the same candidate region:
  reading that operand would require the producing thread to be done, which
  needs synchronization we don't have. `rejection_kind="residual_internal"`.

The module is pure Python; no CUDA / torch dependencies. All tests run on
the CPU host. The CUDA codegen (`cuda_megakernel_backend.py`) consumes the
`RegionPlan` objects this module produces; the diff-oracle bench
(`run_megakernel_benchmark.py`) uses `execute_region_numpy` as the
GPU-free correctness oracle the megakernel output is compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode


# Op kinds that may participate as the *root* (must perform the matmul).
_ROOT_OPS: FrozenSet[str] = frozenset({OpKind.LINEAR, OpKind.LINEAR_RELU})

# Op kinds that may participate as elementwise / epilogue ops.
_EPILOGUE_OPS: FrozenSet[str] = frozenset({OpKind.RELU, OpKind.ADD, OpKind.SCALE})

# Op kinds that, if encountered as the next op after any root or epilogue op,
# would require global synchronization (they consume an entire intermediate
# tensor across CTAs) — explicit trap list.
_GLOBAL_SYNC_TRIGGERS: FrozenSet[str] = frozenset({OpKind.LINEAR, OpKind.LINEAR_RELU})

# Default shared-memory budget (bytes) for the single-CTA legality proof.
# 48 KiB is the dynamic shared memory guaranteed available per block on every
# CUDA architecture since sm_20 without an opt-in carveout, so a 2-layer MLP
# whose entire intermediate activation fits here can be computed by ONE thread
# block using a `__syncthreads()` barrier instead of grid-wide synchronization.
DEFAULT_SHARED_MEM_BUDGET_BYTES = 48 * 1024


@dataclass(frozen=True)
class RegionPlan:
    """A fusable region. Always safe by v1 legality rules (no grid-wide sync needed).

    Fields:
      - `region_id` — stable string id (`region_<index>_<root_or_first_op>`).
      - `region_kind` — `"linear_with_epilogue"` or `"elementwise_chain"`.
      - `op_names` — ordered op names belonging to this region (execution order).
      - `root_op_name` — for `linear_with_epilogue` the LINEAR/LINEAR_RELU op;
        `None` for `elementwise_chain`.
      - `epilogue_op_names` — ops after the root that fold into the same kernel
        (empty for a pure elementwise chain — that case uses `op_names` directly).
      - `inputs_external` — SSA value names this region consumes from outside
        the region (graph inputs or earlier region outputs).
      - `output` — the SSA value name the region's last op produces (the
        single value the kernel writes).
      - `rationale` — human-readable why-this-region-fuses string used in
        the artifact and in test failure messages.
    """

    region_id: str
    region_kind: str
    op_names: Tuple[str, ...]
    root_op_name: Optional[str]
    epilogue_op_names: Tuple[str, ...]
    inputs_external: Tuple[str, ...]
    output: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_kind": self.region_kind,
            "op_names": list(self.op_names),
            "root_op_name": self.root_op_name,
            "epilogue_op_names": list(self.epilogue_op_names),
            "inputs_external": list(self.inputs_external),
            "output": self.output,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class RegionRejection:
    """A region candidate that was considered but ruled out. Recorded for
    transparency (the artifact lists every rejection + its `rejection_kind`)."""

    candidate_op_names: Tuple[str, ...]
    rejection_kind: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_op_names": list(self.candidate_op_names),
            "rejection_kind": self.rejection_kind,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RegionAnalysis:
    regions: Tuple[RegionPlan, ...]
    rejections: Tuple[RegionRejection, ...]
    ops_in_regions: FrozenSet[str] = field(default_factory=frozenset)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "rejections": [r.to_dict() for r in self.rejections],
            "ops_in_regions": sorted(self.ops_in_regions),
        }


def _consumer_count(graph: GraphIR, value_name: str) -> int:
    val = graph.values.get(value_name)
    if val is None:
        return 0
    return len(val.consumers)


def _next_consumer_op(graph: GraphIR, value_name: str) -> Optional[OpNode]:
    val = graph.values.get(value_name)
    if val is None or not val.consumers:
        return None
    consumer_name = val.consumers[0]
    for op in graph.ops:
        if op.name == consumer_name:
            return op
    return None


def _value_is_graph_output(graph: GraphIR, value_name: str) -> bool:
    return value_name in graph.outputs


def _external_inputs_for_region(
    graph: GraphIR,
    region_ops: Sequence[OpNode],
) -> Tuple[str, ...]:
    """SSA values consumed by ops in the region that are NOT produced inside it.

    Order is stable: appearance order across `region_ops` inputs.
    """
    produced_inside = {out for op in region_ops for out in op.outputs}
    seen: List[str] = []
    seen_set: set = set()
    for op in region_ops:
        for inp in op.inputs:
            if inp in produced_inside or inp in seen_set:
                continue
            seen.append(inp)
            seen_set.add(inp)
    return tuple(seen)


def _try_grow_epilogue_chain(
    graph: GraphIR,
    root: OpNode,
) -> Tuple[List[OpNode], List[RegionRejection]]:
    """Walk forward from a root op, accumulating safe epilogue ops.

    Returns the chosen epilogue list (possibly empty) and any rejections
    recorded along the way (e.g. the global-sync trap).
    """
    epilogue: List[OpNode] = []
    rejections: List[RegionRejection] = []
    if not root.outputs:
        return epilogue, rejections
    current_value = root.outputs[0]
    current_op = root

    while True:
        # If the current op's output has more than one consumer, the chain ends
        # here cleanly (no rejection — we just don't grow further; the existing
        # consumer is still served by the materialized value, since the kernel
        # writes it).
        if _consumer_count(graph, current_value) != 1:
            break
        # If the current value is also a graph output, fold it into the region
        # (the kernel still writes it); but the next op cannot fuse in, since
        # the consumer is "the graph output set", not "the next op". Stop.
        if _value_is_graph_output(graph, current_value):
            break

        nxt = _next_consumer_op(graph, current_value)
        if nxt is None:
            break

        # THE TRAP: a LINEAR or LINEAR_RELU as the next consumer would need
        # grid-wide synchronization to read the full producer output. v1
        # rejects this explicitly so we never quietly emit an incorrect kernel.
        if nxt.op in _GLOBAL_SYNC_TRIGGERS:
            rejections.append(
                RegionRejection(
                    candidate_op_names=tuple(op.name for op in [root, *epilogue, nxt]),
                    rejection_kind="global_sync_required",
                    reason=(
                        f"cannot fuse '{nxt.name}' ({nxt.op}) after "
                        f"'{current_op.name}' ({current_op.op}) — the second "
                        "matmul requires the first layer's entire output to be "
                        "globally visible, which needs grid-wide synchronization "
                        "(not available without cooperative-groups dispatch). v1 "
                        "rejects this; single-CTA bounded multi-layer is future work."
                    ),
                )
            )
            break

        # The next op must be elementwise / epilogue-eligible.
        if nxt.op not in _EPILOGUE_OPS:
            rejections.append(
                RegionRejection(
                    candidate_op_names=tuple(op.name for op in [root, *epilogue, nxt]),
                    rejection_kind="unsupported_op",
                    reason=(
                        f"cannot fuse '{nxt.name}' ({nxt.op}) into the region — "
                        f"v1 epilogue set is {sorted(_EPILOGUE_OPS)}"
                    ),
                )
            )
            break

        # For ADD, the residual operand (the input that isn't the chain output)
        # must come from OUTSIDE the candidate region. If it were produced
        # inside, we'd need to synchronize on that producer thread, which
        # arbitrary thread schedules can't guarantee.
        if nxt.op == OpKind.ADD:
            chain_outputs = {root.outputs[0], *(eop.outputs[0] for eop in epilogue)}
            inside_inputs = [inp for inp in nxt.inputs if inp in chain_outputs]
            outside_inputs = [inp for inp in nxt.inputs if inp not in chain_outputs]
            if len(inside_inputs) != 1 or len(outside_inputs) != 1:
                rejections.append(
                    RegionRejection(
                        candidate_op_names=tuple(op.name for op in [root, *epilogue, nxt]),
                        rejection_kind="residual_internal",
                        reason=(
                            f"ADD '{nxt.name}' has inputs {list(nxt.inputs)}; expected "
                            "exactly one input from the chain and one external residual. "
                            "Internal-only ADD would require in-kernel cross-thread sync."
                        ),
                    )
                )
                break

        epilogue.append(nxt)
        current_op = nxt
        current_value = nxt.outputs[0]

    return epilogue, rejections


def _try_grow_elementwise_chain(
    graph: GraphIR,
    start: OpNode,
    consumed: set,
) -> Tuple[List[OpNode], List[RegionRejection]]:
    """Walk forward from an elementwise op to accumulate a chain of >= 2 ops."""
    chain: List[OpNode] = [start]
    rejections: List[RegionRejection] = []
    current_op = start
    if not start.outputs:
        return chain, rejections
    current_value = start.outputs[0]

    while True:
        if _consumer_count(graph, current_value) != 1:
            break
        if _value_is_graph_output(graph, current_value):
            break
        nxt = _next_consumer_op(graph, current_value)
        if nxt is None or nxt.name in consumed:
            break

        if nxt.op in _GLOBAL_SYNC_TRIGGERS:
            # A linear consuming the elementwise chain is the same trap from
            # the opposite direction — the linear reads the full chain output.
            rejections.append(
                RegionRejection(
                    candidate_op_names=tuple(op.name for op in [*chain, nxt]),
                    rejection_kind="global_sync_required",
                    reason=(
                        f"cannot fuse '{nxt.name}' ({nxt.op}) after elementwise "
                        f"chain ending at '{current_op.name}' — the matmul requires "
                        "the chain's entire output globally."
                    ),
                )
            )
            break
        if nxt.op not in _EPILOGUE_OPS:
            rejections.append(
                RegionRejection(
                    candidate_op_names=tuple(op.name for op in [*chain, nxt]),
                    rejection_kind="unsupported_op",
                    reason=f"chain ended: '{nxt.name}' ({nxt.op}) outside elementwise set",
                )
            )
            break

        if nxt.op == OpKind.ADD:
            chain_outputs = {op.outputs[0] for op in chain}
            inside_inputs = [inp for inp in nxt.inputs if inp in chain_outputs]
            outside_inputs = [inp for inp in nxt.inputs if inp not in chain_outputs]
            if len(inside_inputs) != 1 or len(outside_inputs) != 1:
                rejections.append(
                    RegionRejection(
                        candidate_op_names=tuple(op.name for op in [*chain, nxt]),
                        rejection_kind="residual_internal",
                        reason=(
                            f"ADD '{nxt.name}' inputs {list(nxt.inputs)} are not one-chain + one-external"
                        ),
                    )
                )
                break

        chain.append(nxt)
        current_op = nxt
        current_value = nxt.outputs[0]

    return chain, rejections


def _hidden_dim_of(linear_op: OpNode) -> int:
    """Intermediate width (output features) of a LINEAR / LINEAR_RELU op."""
    return int(np.asarray(linear_op.attrs["weight"]).shape[0])


def _try_form_single_cta_multilayer(
    graph: GraphIR,
    fc1: OpNode,
    region_index: int,
    shared_mem_budget_bytes: int,
) -> Tuple[Optional[RegionPlan], Optional[RegionRejection]]:
    """Attempt to fuse a 2-Linear MLP chain into ONE single-CTA kernel.

    Accepted pattern (v1):
        fc1(LINEAR|LINEAR_RELU) -> [optional RELU] -> fc2(LINEAR|LINEAR_RELU)
    where the intermediate value is single-consumer and not a graph output.

    Legality proof (the whole point): a single thread block can compute layer 1
    into shared memory, `__syncthreads()`, then compute layer 2 — with NO
    grid-wide synchronization — IFF the entire intermediate activation fits one
    block's shared-memory budget. We prove `hidden_dim * 4 bytes <= budget`.

    Returns:
      - `(plan, None)`     when the region forms,
      - `(None, rejection)` when the pattern matched but the legality proof failed,
      - `(None, None)`     when the pattern did not match (caller falls back to
                            the normal epilogue / global-sync-trap path).
    """
    out1 = fc1.outputs[0]
    if _consumer_count(graph, out1) != 1 or _value_is_graph_output(graph, out1):
        return None, None
    nxt = _next_consumer_op(graph, out1)
    if nxt is None:
        return None, None

    mid_ops: List[OpNode] = []
    # Optional single RELU on the hidden activation.
    if nxt.op == OpKind.RELU:
        relu_out = nxt.outputs[0]
        if _consumer_count(graph, relu_out) != 1 or _value_is_graph_output(graph, relu_out):
            return None, None
        mid_ops.append(nxt)
        nxt = _next_consumer_op(graph, relu_out)
        if nxt is None:
            return None, None

    # The next op must be the second matmul; otherwise this is not a 2-layer MLP
    # and the normal path (epilogue growth / global-sync trap) should handle it.
    if nxt.op not in (OpKind.LINEAR, OpKind.LINEAR_RELU):
        return None, None
    fc2 = nxt

    region_ops = [fc1, *mid_ops, fc2]
    hidden_dim = _hidden_dim_of(fc1)
    intermediate_bytes = hidden_dim * 4  # float32 intermediate held in shared mem
    if intermediate_bytes > shared_mem_budget_bytes:
        return None, RegionRejection(
            candidate_op_names=tuple(o.name for o in region_ops),
            rejection_kind="single_cta_exceeds_shared_mem",
            reason=(
                f"2-layer MLP '{fc1.name}'->'{fc2.name}' cannot be single-CTA "
                f"fused: hidden_dim={hidden_dim} needs {intermediate_bytes} B of "
                f"shared memory for the intermediate activation, over the "
                f"{shared_mem_budget_bytes} B budget. A wider hidden layer would "
                "need grid-wide synchronization (cooperative-groups dispatch)."
            ),
        )

    rid = f"region_{region_index}_{fc1.name}"
    plan = RegionPlan(
        region_id=rid,
        region_kind="single_cta_bounded_multilayer",
        op_names=tuple(o.name for o in region_ops),
        root_op_name=fc1.name,
        epilogue_op_names=tuple(o.name for o in mid_ops),
        inputs_external=_external_inputs_for_region(graph, region_ops),
        output=fc2.outputs[0],
        rationale=(
            f"2-layer MLP {[o.op for o in region_ops]} fused into ONE CUDA "
            f"kernel under a single-CTA legality proof: the intermediate "
            f"activation (hidden_dim={hidden_dim}, {intermediate_bytes} B) fits "
            f"one block's {shared_mem_budget_bytes} B shared-memory budget, so a "
            "block-level __syncthreads() barrier replaces grid-wide sync."
        ),
    )
    return plan, None


def find_fusion_regions(
    graph: GraphIR,
    *,
    allow_single_cta_multilayer: bool = False,
    shared_mem_budget_bytes: int = DEFAULT_SHARED_MEM_BUDGET_BYTES,
) -> RegionAnalysis:
    """Greedy left-to-right region formation over the IR.

    Each op is visited once and either assigned to a region (and its
    consumers walked to grow the region) or left as a singleton (which
    is NOT a region — singletons are simply executed op-by-op by the
    existing runtime).

    The walk records every trap rejection it encounters (especially the
    global-sync trap) so the artifact + tests can prove the rule actively
    fires, not merely "happens to be unreached".
    """
    if not isinstance(graph, GraphIR):
        raise TypeError(f"graph must be a GraphIR, got {type(graph).__name__}")

    regions: List[RegionPlan] = []
    rejections: List[RegionRejection] = []
    consumed: set = set()

    region_counter = 0
    for op in graph.ops:
        if op.name in consumed:
            continue

        if op.op in _ROOT_OPS:
            # Opt-in: try to fuse a whole 2-layer MLP into one single-CTA kernel
            # before falling back to the (default) epilogue / global-sync path.
            if allow_single_cta_multilayer:
                ml_plan, ml_rej = _try_form_single_cta_multilayer(
                    graph, op, region_counter, shared_mem_budget_bytes
                )
                if ml_plan is not None:
                    regions.append(ml_plan)
                    region_counter += 1
                    consumed.update(ml_plan.op_names)
                    continue
                if ml_rej is not None:
                    # Legality proof failed (hidden too wide). Record it, then
                    # fall through: fc1 may still fuse a linear_with_epilogue
                    # region (e.g. fc1+relu), leaving fc2 a singleton — the same
                    # tail behavior as the default path.
                    rejections.append(ml_rej)
            epilogue, root_rejections = _try_grow_epilogue_chain(graph, op)
            rejections.extend(root_rejections)
            region_ops = [op, *epilogue]
            # A bare LINEAR with no epilogue is NOT a region — the existing
            # blocked-FC kernel already handles it. Only count regions that
            # actually merge two or more ops, OR that fold an in-kernel
            # ReLU+residual+scale set.
            if len(region_ops) >= 2:
                rid = f"region_{region_counter}_{op.name}"
                region_counter += 1
                plan = RegionPlan(
                    region_id=rid,
                    region_kind="linear_with_epilogue",
                    op_names=tuple(o.name for o in region_ops),
                    root_op_name=op.name,
                    epilogue_op_names=tuple(eop.name for eop in epilogue),
                    inputs_external=_external_inputs_for_region(graph, region_ops),
                    output=region_ops[-1].outputs[0],
                    rationale=(
                        f"matmul root '{op.name}' ({op.op}) with epilogue "
                        f"{[e.op for e in epilogue]} folded into one CUDA "
                        "kernel; per-thread accumulator finishes before "
                        "the elementwise tail, so no cross-CTA sync is needed."
                    ),
                )
                regions.append(plan)
                consumed.update(plan.op_names)
            continue

        if op.op in _EPILOGUE_OPS:
            chain, chain_rejections = _try_grow_elementwise_chain(graph, op, consumed)
            rejections.extend(chain_rejections)
            if len(chain) >= 2:
                rid = f"region_{region_counter}_{op.name}"
                region_counter += 1
                plan = RegionPlan(
                    region_id=rid,
                    region_kind="elementwise_chain",
                    op_names=tuple(o.name for o in chain),
                    root_op_name=None,
                    epilogue_op_names=tuple(),
                    inputs_external=_external_inputs_for_region(graph, chain),
                    output=chain[-1].outputs[0],
                    rationale=(
                        f"elementwise chain {[o.op for o in chain]} fused "
                        "into one CUDA kernel; per-element independent, no "
                        "reduction, no cross-thread sync needed."
                    ),
                )
                regions.append(plan)
                consumed.update(plan.op_names)
            # If chain length is 1, leave the op as a singleton — the
            # existing executor handles it; we don't emit a degenerate "region".

    return RegionAnalysis(
        regions=tuple(regions),
        rejections=tuple(rejections),
        ops_in_regions=frozenset(consumed),
    )


# -------------------------------------------------------------------------
# CPU-side reference execution of a region. This is the GPU-free correctness
# oracle the CUDA megakernel output is compared against in tests + the
# benchmark artifact. It MUST produce the same numbers as the existing
# reference interpreter would running the ops op-by-op — if it doesn't,
# the region planner is broken.
# -------------------------------------------------------------------------

def _as_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _exec_op_numpy(op: OpNode, values: Dict[str, np.ndarray]) -> np.ndarray:
    if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
        x = _as_float32(values[op.inputs[0]])
        w = _as_float32(op.attrs["weight"])
        bias_arr = op.attrs.get("bias")
        b = _as_float32(bias_arr) if bias_arr is not None else None
        y = np.matmul(x, w.T)
        if b is not None:
            y = y + b
        if op.op == OpKind.LINEAR_RELU:
            y = np.maximum(y, 0.0)
        return y.astype(np.float32, copy=False)
    if op.op == OpKind.RELU:
        return np.maximum(_as_float32(values[op.inputs[0]]), 0.0).astype(np.float32, copy=False)
    if op.op == OpKind.ADD:
        lhs = _as_float32(values[op.inputs[0]])
        rhs = _as_float32(values[op.inputs[1]])
        return (lhs + rhs).astype(np.float32, copy=False)
    if op.op == OpKind.SCALE:
        x = _as_float32(values[op.inputs[0]])
        s = float(op.attrs.get("scale", 1.0))
        return (x * s).astype(np.float32, copy=False)
    raise ValueError(f"region executor does not support op kind '{op.op}' (v1 regions are linear-with-epilogue / elementwise-chain only)")


def execute_region_numpy(
    region: RegionPlan,
    graph: GraphIR,
    external_values: Dict[str, np.ndarray],
) -> np.ndarray:
    """Run a region op-by-op in NumPy. Returns the region's output ndarray.

    `external_values` must provide every name in `region.inputs_external`.
    This function does NOT change shapes or dtypes vs. the reference
    interpreter — it is byte-identical for the in-region ops to the
    `GraphReferenceInterpreter` path, by construction.
    """
    missing = [name for name in region.inputs_external if name not in external_values]
    if missing:
        raise KeyError(
            f"execute_region_numpy: missing external inputs {missing} for region {region.region_id}"
        )
    values: Dict[str, np.ndarray] = {name: _as_float32(external_values[name]) for name in region.inputs_external}
    op_by_name = {op.name: op for op in graph.ops}
    for op_name in region.op_names:
        op = op_by_name[op_name]
        out = _exec_op_numpy(op, values)
        values[op.outputs[0]] = out
    return values[region.output]
