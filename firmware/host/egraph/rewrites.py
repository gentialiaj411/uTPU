"""Semantics-preserving rewrite rules over the e-graph.

Each ``Rewrite`` is a (matcher, builder) pair. ``matcher(eg, eid)`` returns
a list of *match bindings* (dicts of named e-class ids) for a given seed
e-class. ``builder(eg, binding)`` constructs the RHS by adding fresh
e-nodes to the e-graph and returns the e-class id of the rewritten root.
The saturation driver then calls ``eg.merge(seed, rewritten_root)`` so
both sides live in the same e-class.

We deliberately keep the rule set small + auditable. Each rule has a
short docstring explaining the semantic justification. All rules are
**directionally one-way** at the AST level (LHS -> RHS) but become
bidirectional in the e-graph because of the merge.

Current rules (all semantics-preserving by inspection):

  - ``LINEAR_RELU_FUSION``: ``relu(linear(x, w, b)) == linear_relu(x, w, b)``
    (the project already has this as a fixed-pipeline pass; we just
    re-express it as an e-graph rule so it can compose with peers).

  - ``SCALE_SOFTMAX_FUSION``: ``softmax(scale(x, s), dim) ==``
    ``scaled_softmax(x, s, dim)`` — softmax is scale-equivariant only
    if the scale is incorporated into the kernel; this is the existing
    project fusion expressed as a rule.

  - ``SCALE_REASSOCIATION``: ``scale(scale(x, a), b) == scale(x, a*b)``
    — two consecutive scalar scales fuse into one. This is the
    *phase-ordering* rule: in the fixed pipeline, DCE / fusion run
    before this fold, so a graph with two redundant scales never gets
    the second-level fusion (e.g. ``scale_softmax``) it would unlock.

  - ``SCALE_ZERO_ELIMINATION``: ``scale(x, 1.0) == x`` — multiplicative
    identity. Always a strict improvement when it fires.

  - ``ADD_COMMUTATIVITY``: ``add(a, b) == add(b, a)`` — gives the
    extractor a chance to pick whichever ordering enables further
    rewrites. Safe because numpy / CUDA / our ISA all commute add.

  - ``PERMUTE_INVOLUTION``: ``permute(permute(x, p), p^{-1}) == x``
    when ``p ∘ p^{-1} == identity``. Strict cost reduction when fires.

Note on safety: the diff-oracle gate downstream means *any* rule that
breaks semantics will be caught at extraction time and the extraction
rejected. The rules above are documented as semantics-preserving so a
reader can audit them, NOT because correctness depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from graph_ir import OpKind
from .graph_ir_lang import (
    _INPUT_SHAPES_ATTR,
    _OUTPUT_SHAPE_ATTR,
    _canonicalize_attr_value,
    _decanonicalize_attr_value,
)
from .egraph import EClassId, EGraph, ENode


MatchBinding = Dict[str, Any]
Matcher = Callable[[EGraph, EClassId], List[MatchBinding]]
Builder = Callable[[EGraph, MatchBinding], EClassId]


@dataclass(frozen=True)
class Rewrite:
    name: str
    matcher: Matcher
    builder: Builder
    description: str = ""


def _enodes_of(eg: EGraph, eid: EClassId) -> List[ENode]:
    return eg.class_nodes(eid)


def _attrs_dict(node: ENode) -> Dict[str, Any]:
    return dict(node.attrs_key)


def _shape_tuple(value: Any) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, tuple) and value and value[0] == "__ndarray__":
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


def _shape_list(raw: Any) -> Tuple[Optional[Tuple[int, ...]], ...]:
    if not isinstance(raw, tuple):
        return ()
    return tuple(_shape_tuple(v) for v in raw)


def _shape_meta(attrs: Dict[str, Any]) -> Tuple[Tuple[Optional[Tuple[int, ...]], ...], Tuple[Optional[Tuple[int, ...]], ...]]:
    return _shape_list(attrs.get(_INPUT_SHAPES_ATTR)), _shape_list(attrs.get(_OUTPUT_SHAPE_ATTR))


def _shape_meta_items(
    input_shapes: Tuple[Optional[Tuple[int, ...]], ...],
    output_shapes: Tuple[Optional[Tuple[int, ...]], ...],
) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if input_shapes:
        items.append((_INPUT_SHAPES_ATTR, input_shapes))
    if output_shapes:
        items.append((_OUTPUT_SHAPE_ATTR, output_shapes))
    return items


def _bmm_output_shape(lhs: Optional[Tuple[int, ...]], rhs: Optional[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    if lhs is None or rhs is None or len(lhs) < 2 or len(rhs) < 2:
        return None
    lhs_batch = tuple(int(d) for d in lhs[:-2])
    rhs_batch = tuple(int(d) for d in rhs[:-2])
    max_rank = max(len(lhs_batch), len(rhs_batch))
    lhs_pad = (1,) * (max_rank - len(lhs_batch)) + lhs_batch
    rhs_pad = (1,) * (max_rank - len(rhs_batch)) + rhs_batch
    batch: List[int] = []
    for a, b in zip(lhs_pad, rhs_pad):
        if a == 1:
            batch.append(int(b))
        elif b == 1 or a == b:
            batch.append(int(a))
        else:
            return None
    return tuple(batch + [int(lhs[-2]), int(rhs[-1])])


def _maybe_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    dec = _decanonicalize_attr_value(value)
    if dec is None:
        return None
    return np.asarray(dec, dtype=np.float32)


def _match_linear_relu_fusion(eg: EGraph, seed: EClassId) -> List[MatchBinding]:
    """``relu(linear(x, w, b)) -> linear_relu(x, w, b)``"""
    out: List[MatchBinding] = []
    for node in _enodes_of(eg, seed):
        if node.head != OpKind.RELU or len(node.children) != 1:
            continue
        inner = node.children[0]
        for inner_node in _enodes_of(eg, inner):
            if inner_node.head != OpKind.LINEAR:
                continue
            out.append({
                "x": inner_node.children[0],
                "w": inner_node.children[1] if len(inner_node.children) >= 2 else None,
                "b": inner_node.children[2] if len(inner_node.children) >= 3 else None,
                "linear_attrs": inner_node.attrs_key,
            })
    return out


def _build_linear_relu_fusion(eg: EGraph, binding: MatchBinding) -> EClassId:
    children = [binding["x"]]
    if binding.get("w") is not None:
        children.append(binding["w"])
    if binding.get("b") is not None:
        children.append(binding["b"])
    return eg.add(ENode(
        head=OpKind.LINEAR_RELU,
        children=tuple(children),
        attrs_key=binding.get("linear_attrs", ()),
    ))


LINEAR_RELU_FUSION = Rewrite(
    name="linear_relu_fusion",
    matcher=_match_linear_relu_fusion,
    builder=_build_linear_relu_fusion,
    description="relu(linear(x,w,b)) == linear_relu(x,w,b)",
)


def _match_linear_fusion(eg: EGraph, seed: EClassId) -> List[MatchBinding]:
    """``linear(linear(x, W1, b1), W2, b2) -> linear(x, W2 @ W1, b')``."""
    out: List[MatchBinding] = []
    for node in _enodes_of(eg, seed):
        if node.head != OpKind.LINEAR or len(node.children) != 1:
            continue
        outer_attrs = _attrs_dict(node)
        if "weight" not in outer_attrs:
            continue
        if outer_attrs.get("dtype_quant") is not None:
            continue
        outer_input_shapes, outer_output_shapes = _shape_meta(outer_attrs)
        inner = node.children[0]
        for inner_node in _enodes_of(eg, inner):
            if inner_node.head != OpKind.LINEAR or len(inner_node.children) != 1:
                continue
            inner_attrs = _attrs_dict(inner_node)
            if "weight" not in inner_attrs:
                continue
            if inner_attrs.get("dtype_quant") is not None:
                continue
            inner_input_shapes, inner_output_shapes = _shape_meta(inner_attrs)
            out.append({
                "x": inner_node.children[0],
                "inner_weight": inner_attrs["weight"],
                "inner_bias": inner_attrs.get("bias"),
                "outer_weight": outer_attrs["weight"],
                "outer_bias": outer_attrs.get("bias"),
                "input_shapes": inner_input_shapes,
                "output_shapes": outer_output_shapes,
            })
    return out


def _build_linear_fusion(eg: EGraph, binding: MatchBinding) -> EClassId:
    inner_w = np.asarray(_decanonicalize_attr_value(binding["inner_weight"]), dtype=np.float32)
    outer_w = np.asarray(_decanonicalize_attr_value(binding["outer_weight"]), dtype=np.float32)
    fused_w = np.matmul(outer_w, inner_w).astype(np.float32, copy=False)

    inner_b = _maybe_array(binding.get("inner_bias"))
    outer_b = _maybe_array(binding.get("outer_bias"))
    fused_b: Optional[np.ndarray] = None
    if inner_b is not None:
        b_term = np.matmul(inner_b.astype(np.float32, copy=False), outer_w.T.astype(np.float32, copy=False))
        fused_b = b_term.astype(np.float32, copy=False)
        if outer_b is not None:
            fused_b = (fused_b + outer_b.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    elif outer_b is not None:
        fused_b = outer_b.astype(np.float32, copy=False)

    attrs: List[Tuple[str, Any]] = [
        ("weight", fused_w),
        ("in_features", int(fused_w.shape[1])),
        ("out_features", int(fused_w.shape[0])),
    ]
    if fused_b is not None:
        attrs.append(("bias", fused_b))
    attrs.extend(_shape_meta_items(binding.get("input_shapes", ()), binding.get("output_shapes", ())))
    return eg.add(ENode(
        head=OpKind.LINEAR,
        children=(binding["x"],),
        attrs_key=tuple((str(k), _canonicalize_attr_value(v)) for k, v in attrs),
    ))


LINEAR_FUSION = Rewrite(
    name="linear_fusion",
    matcher=_match_linear_fusion,
    builder=_build_linear_fusion,
    description="linear(linear(x,W1,b1),W2,b2) == linear(x,W2@W1,b')",
)


def _match_batched_matmul_association(eg: EGraph, seed: EClassId) -> List[MatchBinding]:
    """``matmul(matmul(A,B),C) == matmul(A,matmul(B,C))``."""
    out: List[MatchBinding] = []
    for node in _enodes_of(eg, seed):
        if node.head != OpKind.BATCHED_MATMUL or len(node.children) != 2:
            continue
        outer_attrs = _attrs_dict(node)
        outer_inputs, outer_outputs = _shape_meta(outer_attrs)
        left = node.children[0]
        right = node.children[1]
        for inner_node in _enodes_of(eg, left):
            if inner_node.head != OpKind.BATCHED_MATMUL or len(inner_node.children) != 2:
                continue
            inner_attrs = _attrs_dict(inner_node)
            inner_inputs, inner_outputs = _shape_meta(inner_attrs)
            a_shape = inner_inputs[0] if len(inner_inputs) >= 1 else None
            b_shape = inner_inputs[1] if len(inner_inputs) >= 2 else None
            c_shape = outer_inputs[1] if len(outer_inputs) >= 2 else None
            if a_shape is None or b_shape is None or c_shape is None:
                continue
            bc_shape = _bmm_output_shape(b_shape, c_shape)
            if bc_shape is None:
                continue
            rhs_shape = outer_outputs[0] if len(outer_outputs) >= 1 else None
            if rhs_shape is None:
                continue
            out.append({
                "a": inner_node.children[0],
                "b": inner_node.children[1],
                "c": right,
                "left_input_shapes": inner_inputs,
                "inner_output_shape": inner_outputs[0] if len(inner_outputs) >= 1 else None,
                "rhs_input_shapes": outer_inputs,
                "output_shapes": outer_outputs,
            })
    return out


def _build_batched_matmul_association(eg: EGraph, binding: MatchBinding) -> EClassId:
    inner_inputs = binding.get("left_input_shapes", ())
    outer_outputs = binding.get("output_shapes", ())
    b_shape = inner_inputs[1] if len(inner_inputs) >= 2 else None
    c_shape = binding.get("rhs_input_shapes", ())
    c_rhs = c_shape[1] if len(c_shape) >= 2 else None
    inner_out = _bmm_output_shape(b_shape, c_rhs)
    inner_attrs: List[Tuple[str, Any]] = _shape_meta_items(
        (b_shape, c_rhs),
        (inner_out,) if inner_out is not None else (),
    )
    inner_eid = eg.add(ENode(
        head=OpKind.BATCHED_MATMUL,
        children=(binding["b"], binding["c"]),
        attrs_key=tuple((str(k), _canonicalize_attr_value(v)) for k, v in inner_attrs),
    ))
    outer_attrs: List[Tuple[str, Any]] = _shape_meta_items(
        (binding.get("left_input_shapes", ())[0] if binding.get("left_input_shapes") else None, inner_out),
        outer_outputs,
    )
    return eg.add(ENode(
        head=OpKind.BATCHED_MATMUL,
        children=(binding["a"], inner_eid),
        attrs_key=tuple((str(k), _canonicalize_attr_value(v)) for k, v in outer_attrs),
    ))


BATCHED_MATMUL_ASSOCIATION = Rewrite(
    name="batched_matmul_association",
    matcher=_match_batched_matmul_association,
    builder=_build_batched_matmul_association,
    description="matmul(matmul(A,B),C) == matmul(A,matmul(B,C))",
)


def _match_scale_softmax_fusion(eg: EGraph, seed: EClassId) -> List[MatchBinding]:
    """``softmax(scale(x, s), dim) -> scaled_softmax(x, s, dim)``"""
    out: List[MatchBinding] = []
    for node in _enodes_of(eg, seed):
        if node.head != OpKind.SOFTMAX or len(node.children) != 1:
            continue
        sm_attrs = _attrs_dict(node)
        inner = node.children[0]
        for inner_node in _enodes_of(eg, inner):
            if inner_node.head != OpKind.SCALE or len(inner_node.children) != 1:
                continue
            scale_attrs = _attrs_dict(inner_node)
            if "scale" not in scale_attrs:
                continue
            out.append({
                "x": inner_node.children[0],
                "scale": scale_attrs["scale"],
                "softmax_attrs": sm_attrs,
            })
    return out


def _build_scale_softmax_fusion(eg: EGraph, binding: MatchBinding) -> EClassId:
    new_attrs: List[Tuple[str, Any]] = [("scale", binding["scale"])]
    sm_attrs = binding.get("softmax_attrs", {})
    if "dim" in sm_attrs:
        new_attrs.append(("dim", sm_attrs["dim"]))
    if "causal_mask" in sm_attrs:
        new_attrs.append(("causal_mask", sm_attrs["causal_mask"]))
    return eg.add(ENode(
        head=OpKind.SCALED_SOFTMAX,
        children=(binding["x"],),
        attrs_key=tuple(new_attrs),
    ))


SCALE_SOFTMAX_FUSION = Rewrite(
    name="scale_softmax_fusion",
    matcher=_match_scale_softmax_fusion,
    builder=_build_scale_softmax_fusion,
    description="softmax(scale(x,s),dim) == scaled_softmax(x,s,dim)",
)


def _match_scale_reassociation(eg: EGraph, seed: EClassId) -> List[MatchBinding]:
    """``scale(scale(x, a), b) -> scale(x, a*b)``  (THE phase-ordering rule)"""
    out: List[MatchBinding] = []
    for node in _enodes_of(eg, seed):
        if node.head != OpKind.SCALE or len(node.children) != 1:
            continue
        outer_attrs = _attrs_dict(node)
        if "scale" not in outer_attrs:
            continue
        b = outer_attrs["scale"]
        inner = node.children[0]
        for inner_node in _enodes_of(eg, inner):
            if inner_node.head != OpKind.SCALE or len(inner_node.children) != 1:
                continue
            inner_attrs = _attrs_dict(inner_node)
            if "scale" not in inner_attrs:
                continue
            a = inner_attrs["scale"]
            try:
                combined = float(a) * float(b)
            except (TypeError, ValueError):
                continue
            out.append({
                "x": inner_node.children[0],
                "combined_scale": combined,
            })
    return out


def _build_scale_reassociation(eg: EGraph, binding: MatchBinding) -> EClassId:
    return eg.add(ENode(
        head=OpKind.SCALE,
        children=(binding["x"],),
        attrs_key=(("scale", float(binding["combined_scale"])),),
    ))


SCALE_REASSOCIATION = Rewrite(
    name="scale_reassociation",
    matcher=_match_scale_reassociation,
    builder=_build_scale_reassociation,
    description="scale(scale(x,a),b) == scale(x,a*b)  -- phase-ordering rule",
)


def _match_scale_identity(eg: EGraph, seed: EClassId) -> List[MatchBinding]:
    """``scale(x, 1.0) -> x``  (strict cost reduction)"""
    out: List[MatchBinding] = []
    for node in _enodes_of(eg, seed):
        if node.head != OpKind.SCALE or len(node.children) != 1:
            continue
        attrs = _attrs_dict(node)
        if "scale" not in attrs:
            continue
        try:
            if float(attrs["scale"]) == 1.0:
                out.append({"x": node.children[0]})
        except (TypeError, ValueError):
            continue
    return out


def _build_scale_identity(eg: EGraph, binding: MatchBinding) -> EClassId:
    return binding["x"]


SCALE_IDENTITY = Rewrite(
    name="scale_identity",
    matcher=_match_scale_identity,
    builder=_build_scale_identity,
    description="scale(x,1.0) == x",
)


def _match_add_commutativity(eg: EGraph, seed: EClassId) -> List[MatchBinding]:
    """``add(a, b) -> add(b, a)``  (no-op cost-wise, but enables matching)"""
    out: List[MatchBinding] = []
    for node in _enodes_of(eg, seed):
        if node.head != OpKind.ADD or len(node.children) != 2:
            continue
        out.append({"a": node.children[0], "b": node.children[1]})
    return out


def _build_add_commutativity(eg: EGraph, binding: MatchBinding) -> EClassId:
    return eg.add(ENode(
        head=OpKind.ADD,
        children=(binding["b"], binding["a"]),
        attrs_key=(),
    ))


ADD_COMMUTATIVITY = Rewrite(
    name="add_commutativity",
    matcher=_match_add_commutativity,
    builder=_build_add_commutativity,
    description="add(a,b) == add(b,a)",
)


def _match_permute_involution(eg: EGraph, seed: EClassId) -> List[MatchBinding]:
    """``permute(permute(x, p), q) -> x`` when ``q == p^{-1}``"""
    out: List[MatchBinding] = []
    for node in _enodes_of(eg, seed):
        if node.head != OpKind.PERMUTE or len(node.children) != 1:
            continue
        outer_attrs = _attrs_dict(node)
        outer_p = outer_attrs.get("args")
        if not isinstance(outer_p, tuple):
            continue
        try:
            outer_perm = tuple(int(i) for i in outer_p)
        except (TypeError, ValueError):
            continue
        inner = node.children[0]
        for inner_node in _enodes_of(eg, inner):
            if inner_node.head != OpKind.PERMUTE or len(inner_node.children) != 1:
                continue
            inner_attrs = _attrs_dict(inner_node)
            inner_p = inner_attrs.get("args")
            if not isinstance(inner_p, tuple):
                continue
            try:
                inner_perm = tuple(int(i) for i in inner_p)
            except (TypeError, ValueError):
                continue
            if len(inner_perm) != len(outer_perm):
                continue
            try:
                composed = tuple(inner_perm[i] for i in outer_perm)
            except IndexError:
                continue
            if composed == tuple(range(len(composed))):
                out.append({"x": inner_node.children[0]})
    return out


def _build_permute_involution(eg: EGraph, binding: MatchBinding) -> EClassId:
    return binding["x"]


PERMUTE_INVOLUTION = Rewrite(
    name="permute_involution",
    matcher=_match_permute_involution,
    builder=_build_permute_involution,
    description="permute(permute(x,p),q) == x  when p ∘ q == identity",
)


DEFAULT_REWRITES: Tuple[Rewrite, ...] = (
    LINEAR_RELU_FUSION,
    LINEAR_FUSION,
    BATCHED_MATMUL_ASSOCIATION,
    SCALE_SOFTMAX_FUSION,
    SCALE_REASSOCIATION,
    SCALE_IDENTITY,
    PERMUTE_INVOLUTION,
    ADD_COMMUTATIVITY,
)


def apply_rewrite(eg: EGraph, rule: Rewrite, seed_eid: EClassId) -> int:
    """Apply ``rule`` once at ``seed_eid``: find all matches, build each
    RHS, merge with the seed. Returns the number of *new* merges
    triggered by this call. A "new merge" means seed and RHS were in
    different e-classes BEFORE the merge; if they were already congruent,
    no merge is counted (so saturation termination is detectable by
    "no merges fired in a full pass"). Builder side-effects (adding fresh
    e-nodes that hashcons-dedup into existing classes) are NOT counted
    as merges."""
    bindings = rule.matcher(eg, seed_eid)
    merges = 0
    for binding in bindings:
        rhs_eid = rule.builder(eg, binding)
        seed_root = eg.find(seed_eid)
        rhs_root = eg.find(rhs_eid)
        if seed_root == rhs_root:
            continue
        eg.merge(seed_eid, rhs_eid)
        merges += 1
    return merges


__all__ = [
    "ADD_COMMUTATIVITY",
    "Builder",
    "DEFAULT_REWRITES",
    "BATCHED_MATMUL_ASSOCIATION",
    "LINEAR_RELU_FUSION",
    "LINEAR_FUSION",
    "MatchBinding",
    "Matcher",
    "PERMUTE_INVOLUTION",
    "Rewrite",
    "SCALE_IDENTITY",
    "SCALE_REASSOCIATION",
    "SCALE_SOFTMAX_FUSION",
    "apply_rewrite",
]
