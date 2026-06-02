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

from graph_ir import OpKind
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
    "LINEAR_RELU_FUSION",
    "MatchBinding",
    "Matcher",
    "PERMUTE_INVOLUTION",
    "Rewrite",
    "SCALE_IDENTITY",
    "SCALE_REASSOCIATION",
    "SCALE_SOFTMAX_FUSION",
    "apply_rewrite",
]
