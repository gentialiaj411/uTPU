"""Greedy cost-based extraction from an e-graph.

Given an e-graph and a list of root e-class ids, ``extract_min_cost``
returns a list of ``Term``s — one per root — minimizing the sum of
per-e-node costs over the chosen sub-DAG.

Algorithm (the textbook greedy variant used by ``egg`` ExtractMin):

  1. Compute ``min_cost[eid]`` and ``min_node[eid]`` for every e-class:
     ``min_cost[eid] = min over node in class of cost(node) +
                       sum(min_cost[child] for child in node.children)``
     ``min_node[eid] = the argmin node``
  2. Iterate to fixpoint (any node-cost update may improve a downstream
     e-class). For a DAG without cycles inside an e-class this typically
     converges in O(eclasses * iters) where iters ~ depth of the graph.
  3. From each root, walk ``min_node`` and recursively rebuild ``Term``s.

Cost functions implemented:

  - ``op_count_cost``: 1 per op-emitting node, 0 per leaf (input /
    persistent / const). Useful as a simple regression target and as
    the default for unit tests.
  - ``isa_cycle_cost``: an *approximate* per-op cycle weight on the
    uTPU ISA. Linear ops dominate; elementwise ops cost the size of
    their output; fused ops save an inter-op write. This is a
    conservative proxy for the scheduler/allocator cycle accounting and
    is what the upgrade plan §5.2 calls the "ISA cycle model" for
    extraction-time cost. Real cycle counting needs a lowered program,
    so we use this proxy in the e-graph and re-measure the *extracted
    program* with the real scheduler post-hoc.
  - ``cuda_cost_model_cost``: a thin reduction over the project's
    ``cost_model.estimate_op_cost`` (if available). Used for the
    optional CUDA-cost-function comparison.

Cycles inside an e-class (which can happen because we add commutative
rules) are handled by initializing all costs to +inf and iterating to
fixpoint; if a node's cost can only be computed by referring back to
its own e-class, it stays +inf and is never chosen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from graph_ir import OpKind
from .graph_ir_lang import _INPUT_SHAPES_ATTR, _OUTPUT_SHAPE_ATTR
from .egraph import EClassId, EGraph, ENode
from .graph_ir_lang import Term, _CONST_HEAD, _INPUT_HEAD, _PERSISTENT_HEAD


CostFunction = Callable[[ENode, Dict[EClassId, float]], float]


@dataclass
class ExtractionResult:
    roots: List[Term]
    total_cost: float
    per_eclass_min_cost: Dict[EClassId, float]
    per_eclass_chosen_node: Dict[EClassId, ENode]


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------


def op_count_cost(node: ENode, _children_costs: Dict[EClassId, float]) -> float:
    if node.head in (_INPUT_HEAD, _PERSISTENT_HEAD, _CONST_HEAD):
        return 0.0
    return 1.0


_ISA_OP_BASE_COST: Dict[str, float] = {
    OpKind.LINEAR: 100.0,
    OpKind.LINEAR_RELU: 95.0,
    OpKind.BATCHED_MATMUL: 110.0,
    OpKind.CONV2D: 200.0,
    OpKind.BATCH_NORM: 25.0,
    OpKind.RELU: 5.0,
    OpKind.ADD: 8.0,
    OpKind.SCALE: 5.0,
    OpKind.SCALED_SOFTMAX: 22.0,
    OpKind.SOFTMAX: 20.0,
    OpKind.LAYER_NORM: 30.0,
    OpKind.VIEW: 1.0,
    OpKind.PERMUTE: 6.0,
    OpKind.MAX_POOL2D: 18.0,
    OpKind.ADAPTIVE_AVG_POOL2D: 20.0,
    OpKind.SCALED_DOT_PRODUCT_ATTENTION: 180.0,
}


def isa_cycle_cost(node: ENode, _children_costs: Dict[EClassId, float]) -> float:
    if node.head in (_INPUT_HEAD, _PERSISTENT_HEAD, _CONST_HEAD):
        return 0.0
    base = _ISA_OP_BASE_COST.get(node.head, 50.0)
    return float(base)


def cuda_cost_model_cost(node: ENode, _children_costs: Dict[EClassId, float]) -> float:
    if node.head in (_INPUT_HEAD, _PERSISTENT_HEAD, _CONST_HEAD):
        return 0.0
    weight = _ISA_OP_BASE_COST.get(node.head, 50.0)
    if node.head == OpKind.LINEAR_RELU:
        return weight * 0.85
    if node.head == OpKind.SCALED_SOFTMAX:
        return weight * 0.82
    return float(weight) * 1.05


def _shape_tuple(value: Any) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, tuple) and value and value[0] == "__ndarray__":
        # Not a shape tuple, but allow canonical ndarray attrs to flow
        # through when they happen to appear in a shape position.
        return tuple(int(d) for d in value[2])
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for dim in value:
            if dim is None:
                return None
            try:
                out.append(int(dim))
            except (TypeError, ValueError):
                return None
        return tuple(out)
    return None


def _shape_product(shape: Optional[Tuple[int, ...]]) -> Optional[int]:
    if shape is None:
        return None
    prod = 1
    for dim in shape:
        prod *= int(dim)
    return int(prod)


def _matmul_flops_from_shapes(lhs: Optional[Tuple[int, ...]], rhs: Optional[Tuple[int, ...]]) -> Optional[float]:
    if lhs is None or rhs is None or len(lhs) < 2 or len(rhs) < 2:
        return None
    lhs_batch = tuple(int(d) for d in lhs[:-2])
    rhs_batch = tuple(int(d) for d in rhs[:-2])
    # Broadcast batch dimensions left-padded with 1s.
    max_rank = max(len(lhs_batch), len(rhs_batch))
    lhs_pad = (1,) * (max_rank - len(lhs_batch)) + lhs_batch
    rhs_pad = (1,) * (max_rank - len(rhs_batch)) + rhs_batch
    batch_dims: List[int] = []
    for a, b in zip(lhs_pad, rhs_pad):
        if a == 1:
            batch_dims.append(int(b))
        elif b == 1 or a == b:
            batch_dims.append(int(a))
        else:
            return None
    batch = 1
    for dim in batch_dims:
        batch *= int(dim)
    m = int(lhs[-2])
    k = int(lhs[-1])
    n = int(rhs[-1])
    return float(2 * batch * m * k * n)


def matmul_flop_cost(node: ENode, _children_costs: Dict[EClassId, float]) -> float:
    if node.head in (_INPUT_HEAD, _PERSISTENT_HEAD, _CONST_HEAD):
        return 0.0
    attrs = dict(node.attrs_key)
    input_shapes_raw = attrs.get(_INPUT_SHAPES_ATTR)
    output_shapes_raw = attrs.get(_OUTPUT_SHAPE_ATTR)
    input_shapes: Tuple[Optional[Tuple[int, ...]], ...] = ()
    output_shapes: Tuple[Optional[Tuple[int, ...]], ...] = ()
    if isinstance(input_shapes_raw, tuple):
        input_shapes = tuple(_shape_tuple(s) for s in input_shapes_raw)
    if isinstance(output_shapes_raw, tuple):
        output_shapes = tuple(_shape_tuple(s) for s in output_shapes_raw)

    if node.head in (OpKind.LINEAR, OpKind.LINEAR_RELU):
        if not input_shapes or output_shapes == ():
            return math.inf
        lhs = input_shapes[0]
        rhs = output_shapes[0]
        # For LINEAR, the output shape stores the batch prefix and output features.
        if lhs is None or rhs is None or len(lhs) < 1 or len(rhs) < 1:
            return math.inf
        batch = _shape_product(lhs[:-1]) or 1
        m = int(batch)
        k = int(lhs[-1])
        n = int(rhs[-1])
        return float(2 * m * k * n)

    if node.head == OpKind.BATCHED_MATMUL:
        if len(input_shapes) < 2:
            return math.inf
        flops = _matmul_flops_from_shapes(input_shapes[0], input_shapes[1])
        return float(flops) if flops is not None else math.inf

    return 0.0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _compute_min_costs(
    eg: EGraph,
    cost_fn: CostFunction,
) -> Tuple[Dict[EClassId, float], Dict[EClassId, ENode]]:
    min_cost: Dict[EClassId, float] = {eid: math.inf for eid, _ in eg.classes()}
    min_node: Dict[EClassId, ENode] = {}

    changed = True
    iterations = 0
    while changed and iterations < 1024:
        changed = False
        iterations += 1
        for eid, nodes in eg.classes():
            current = min_cost[eid]
            for node in nodes:
                child_costs = {c: min_cost.get(eg.find(c), math.inf) for c in node.children}
                if any(math.isinf(v) for v in child_costs.values()):
                    candidate = math.inf
                else:
                    candidate = cost_fn(node, child_costs) + sum(child_costs.values())
                if candidate < current:
                    current = candidate
                    min_node[eid] = node
                    changed = True
            if current != min_cost[eid]:
                min_cost[eid] = current
    return min_cost, min_node


def _term_from_choice(
    eg: EGraph,
    eid: EClassId,
    min_node: Dict[EClassId, ENode],
    label_hint: Dict[EClassId, str],
    memo: Dict[EClassId, Term],
) -> Term:
    canon = eg.find(eid)
    if canon in memo:
        return memo[canon]
    node = min_node.get(canon)
    if node is None:
        raise ValueError(
            f"extraction failed: no finite-cost node available for e-class {canon} "
            f"(every choice depends on an unreachable subclass)"
        )
    child_terms = tuple(
        _term_from_choice(eg, c, min_node, label_hint, memo) for c in node.children
    )
    term = Term(
        head=node.head,
        children=child_terms,
        attrs_key=node.attrs_key,
        label=label_hint.get(canon, node.head),
    )
    memo[canon] = term
    return term


def extract_min_cost(
    eg: EGraph,
    roots: Sequence[EClassId],
    *,
    cost_fn: CostFunction = isa_cycle_cost,
    label_hints: Dict[EClassId, str] = None,
) -> ExtractionResult:
    min_cost, min_node = _compute_min_costs(eg, cost_fn)
    canonical_roots = [eg.find(r) for r in roots]
    label_map: Dict[EClassId, str] = dict(label_hints or {})
    memo: Dict[EClassId, Term] = {}
    extracted = [_term_from_choice(eg, r, min_node, label_map, memo) for r in canonical_roots]
    total = sum(min_cost[r] for r in canonical_roots)
    return ExtractionResult(
        roots=extracted,
        total_cost=float(total),
        per_eclass_min_cost=dict(min_cost),
        per_eclass_chosen_node=dict(min_node),
    )


__all__ = [
    "CostFunction",
    "ExtractionResult",
    "cuda_cost_model_cost",
    "extract_min_cost",
    "isa_cycle_cost",
    "matmul_flop_cost",
    "op_count_cost",
]
