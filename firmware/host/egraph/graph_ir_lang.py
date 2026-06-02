"""Bridge between ``GraphIR`` and the e-graph term language.

GraphIR is a stateful, named, ordered op list with attrs dicts. The e-graph
operates on a flat ``Term`` (s-expression) form: each term is a head symbol +
ordered child terms + an opaque hashable ``attrs_key``.

The bridge does three things:

  1. ``lift_graph_ir(graph)`` walks the graph from each declared output
     backward through producers, building one ``Term`` per output and
     interning input tensors as ``Term`` leaves with head ``"input"``.

  2. ``Term`` is convertible to / from ``ENode`` via ``term_to_enode`` and
     the recursive ``insert_term_into_egraph`` driver. Children of an e-node
     are e-class ids; ``Term`` carries the children directly so we can
     recursively insert them and refer to their canonical ids.

  3. ``lower_term_to_graph_ir(roots, ...)`` reconstructs a fresh
     ``GraphIR`` from a list of extracted root ``Term``s, generating fresh
     value names and re-running shape inference. Two extracted terms that
     share a child subterm by *Python object identity* are coalesced into a
     single ``OpNode`` in the lowered graph (so common subexpressions
     stay shared).

We deliberately encode the **semantically relevant** attrs (scale value,
permute dims, view target shape, softmax/layer_norm flags) into
``attrs_key``. Attrs that are purely informational (``"name"``, ``"target"``
debug strings) are NOT in the key, since two e-nodes that differ only in
debug metadata are semantically equal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode, TensorValue
from .egraph import EClassId, EGraph, ENode


_INPUT_HEAD = "__input__"
_PERSISTENT_HEAD = "__persistent__"
_CONST_HEAD = "__const__"


_ATTR_KEY_DENYLIST: Tuple[str, ...] = (
    "cuda_lowering_available",
    "memory_plan",
    "schedule",
    "cuda_schedule",
    "tile_plan",
    "comment",
    "target",
    "source",
    "lowering_origin",
    "debug",
)
_NDARRAY_TAG = "__ndarray__"


@dataclass(frozen=True)
class Term:
    """A term in the e-graph language. ``head`` is the op kind (or
    ``__input__`` / ``__persistent__`` / ``__const__`` for leaves).
    ``children`` is the tuple of child terms (positional, order-sensitive).
    ``attrs_key`` carries the semantically-relevant attrs as a hashable.
    ``label`` is informational only (preserves a name hint for debugging
    and for ``lower_term_to_graph_ir`` to produce friendly value names).
    ``label`` is NOT part of equality; equality is structural over
    ``head`` + ``children`` + ``attrs_key``.
    """

    head: str
    children: Tuple["Term", ...]
    attrs_key: Tuple[Any, ...]
    label: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Term):
            return False
        return (
            self.head == other.head
            and self.attrs_key == other.attrs_key
            and self.children == other.children
        )

    def __hash__(self) -> int:
        return hash((self.head, self.attrs_key, self.children))


def _canonicalize_attr_value(value: Any) -> Any:
    """Convert ``value`` into a stable, hashable canonical form.

    Lists/tuples become tuples (recursively); dicts become a sorted tuple
    of ``(key, value)`` pairs (recursively); ``np.ndarray`` becomes a
    sentinel-tagged tuple of ``(__ndarray__, dtype, shape, raw bytes)`` so
    bit-identical weight tensors deduplicate but distinct weights stay
    distinct."""
    if isinstance(value, np.ndarray):
        return (_NDARRAY_TAG, value.dtype.str, tuple(int(d) for d in value.shape), value.tobytes())
    if isinstance(value, list):
        return tuple(_canonicalize_attr_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_canonicalize_attr_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted(
            (str(k), _canonicalize_attr_value(v)) for k, v in value.items()
        ))
    return value


def _attrs_key_for_op(op: OpNode) -> Tuple[Any, ...]:
    """Extract a stable hashable summary of ``op.attrs`` over a denylist
    that strips debug/scheduling fields irrelevant to semantics. Weight
    tensors (``op.attrs['weight']``, ``'bias'``, ``'running_mean'``,
    etc.) are kept and canonicalized via the ``__ndarray__`` tag so two
    ops with bit-identical weights congruence-merge."""
    out: List[Tuple[str, Any]] = []
    for k in sorted(op.attrs.keys()):
        if k in _ATTR_KEY_DENYLIST:
            continue
        out.append((str(k), _canonicalize_attr_value(op.attrs[k])))
    return tuple(out)


def lift_graph_ir(graph: GraphIR) -> List[Term]:
    """Convert ``graph`` into a list of ``Term`` roots (one per graph output).

    Common subexpressions in the GraphIR (values produced by the same op
    and consumed by multiple downstream ops) become Python-object-identical
    sub-``Term``s in the result, so downstream lowering can coalesce them
    back into a single ``OpNode``.

    Raises:
        ValueError: if the graph contains a value with no producer that is
            not declared in ``graph.inputs`` (i.e. an undefined free
            reference).
    """
    value_to_term: Dict[str, Term] = {}
    for input_name in graph.inputs:
        tv = graph.values.get(input_name)
        shape = tv.shape if tv else None
        dtype = tv.dtype if tv else None
        value_to_term[input_name] = Term(
            head=_INPUT_HEAD,
            children=(),
            attrs_key=(("name", input_name), ("shape", _canonicalize_attr_value(shape)), ("dtype", dtype)),
            label=input_name,
        )

    for name, tv in graph.values.items():
        if tv.persistent and name not in value_to_term:
            value_to_term[name] = Term(
                head=_PERSISTENT_HEAD,
                children=(),
                attrs_key=(("name", name), ("shape", _canonicalize_attr_value(tv.shape)), ("dtype", tv.dtype)),
                label=name,
            )

    op_by_output: Dict[str, OpNode] = {}
    for op in graph.ops:
        for out in op.outputs:
            op_by_output[out] = op

    def build(value_name: str, stack: List[str]) -> Term:
        if value_name in value_to_term:
            return value_to_term[value_name]
        if value_name in stack:
            raise ValueError(
                f"cycle detected during e-graph lift at value '{value_name}'; "
                f"stack={stack!r}"
            )
        op = op_by_output.get(value_name)
        if op is None:
            tv = graph.values.get(value_name)
            if tv is not None and tv.persistent:
                term = Term(
                    head=_PERSISTENT_HEAD,
                    children=(),
                    attrs_key=(("name", value_name), ("shape", _canonicalize_attr_value(tv.shape)), ("dtype", tv.dtype)),
                    label=value_name,
                )
                value_to_term[value_name] = term
                return term
            raise ValueError(
                f"value '{value_name}' has no producer and is not declared as a graph input"
            )
        stack.append(value_name)
        child_terms = tuple(build(inp, stack) for inp in op.inputs)
        stack.pop()
        term = Term(
            head=op.op,
            children=child_terms,
            attrs_key=_attrs_key_for_op(op),
            label=value_name,
        )
        for out in op.outputs:
            value_to_term[out] = term
        return term

    roots: List[Term] = []
    for out_name in graph.outputs:
        roots.append(build(out_name, []))
    return roots


def insert_term_into_egraph(eg: EGraph, term: Term, memo: Optional[Dict[int, EClassId]] = None) -> EClassId:
    """Insert ``term`` (and all its subterms) into ``eg``; return the
    canonical e-class id of the root.

    ``memo`` (optional) maps Python object id of seen terms to e-class
    ids so structurally-shared subterms in the input also become shared
    e-classes in the e-graph. If omitted, a fresh memo is created."""
    if memo is None:
        memo = {}
    tid = id(term)
    if tid in memo:
        return eg.find(memo[tid])
    child_ids = tuple(insert_term_into_egraph(eg, c, memo) for c in term.children)
    enode = ENode(head=term.head, children=child_ids, attrs_key=term.attrs_key)
    eid = eg.add(enode)
    memo[tid] = eid
    return eid


def term_to_enode(term: Term, child_ids: Sequence[EClassId]) -> ENode:
    return ENode(
        head=term.head,
        children=tuple(child_ids),
        attrs_key=term.attrs_key,
    )


def lower_term_to_graph_ir(
    roots: Sequence[Term],
    *,
    source_graph: GraphIR,
    name_suffix: str = ".egraph",
) -> GraphIR:
    """Reconstruct a fresh ``GraphIR`` from extracted ``Term`` roots.

    ``source_graph`` provides the input declarations, the persistent
    tensor metadata (shape/dtype/value), and the graph name. Common
    subterms (Python-identical) are coalesced into a single ``OpNode``
    in the lowered graph. Value names are regenerated with a stable
    counter so they can't collide with the source graph.

    Re-runs nothing automatically; the caller should invoke
    ``shape_inference_pass`` (or the full pipeline) if shape annotations
    or downstream pass behaviour matter.
    """
    new_graph = GraphIR(
        name=source_graph.name + name_suffix,
        inputs=list(source_graph.inputs),
    )

    for input_name in source_graph.inputs:
        src_tv = source_graph.values.get(input_name)
        new_graph.add_value(
            input_name,
            shape=src_tv.shape if src_tv else None,
            dtype=src_tv.dtype if src_tv else None,
            persistent=False,
        )

    for name, tv in source_graph.values.items():
        if tv.persistent:
            new_graph.add_value(
                name,
                shape=tv.shape,
                dtype=tv.dtype,
                persistent=True,
            )

    op_counter = {"i": 0}
    val_counter = {"i": 0}
    term_to_value_name: Dict[int, str] = {}

    def fresh_op_name(prefix: str) -> str:
        op_counter["i"] += 1
        return f"{prefix}_eg{op_counter['i']:04d}"

    def fresh_val_name(prefix: str) -> str:
        val_counter["i"] += 1
        return f"{prefix}_v{val_counter['i']:04d}"

    def emit(term: Term) -> str:
        """Return the value name produced by ``term`` in the new graph,
        emitting op nodes as needed."""
        tid = id(term)
        if tid in term_to_value_name:
            return term_to_value_name[tid]
        if term.head == _INPUT_HEAD:
            attrs = dict(term.attrs_key)
            name = attrs.get("name", term.label)
            term_to_value_name[tid] = name
            return name
        if term.head == _PERSISTENT_HEAD:
            attrs = dict(term.attrs_key)
            name = attrs.get("name", term.label)
            term_to_value_name[tid] = name
            return name
        child_value_names = [emit(c) for c in term.children]
        prefix = term.label.replace(".", "_") or term.head
        op_name = fresh_op_name(prefix)
        out_name = fresh_val_name(prefix)
        op_attrs: Dict[str, Any] = {}
        for k, v in term.attrs_key:
            op_attrs[k] = _decanonicalize_attr_value(v)
        new_graph.add_op(OpNode(
            name=op_name,
            op=term.head,
            inputs=list(child_value_names),
            outputs=[out_name],
            attrs=op_attrs,
        ))
        term_to_value_name[tid] = out_name
        return out_name

    output_names: List[str] = []
    for root in roots:
        output_names.append(emit(root))
    new_graph.outputs = output_names
    return new_graph


def _decanonicalize_attr_value(value: Any) -> Any:
    if isinstance(value, tuple):
        if (
            len(value) == 4
            and value[0] == _NDARRAY_TAG
            and isinstance(value[1], str)
            and isinstance(value[2], tuple)
            and isinstance(value[3], (bytes, bytearray))
        ):
            return np.frombuffer(value[3], dtype=np.dtype(value[1])).reshape(value[2]).copy()
        if value and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in value):
            return {k: _decanonicalize_attr_value(v) for k, v in value}
        return tuple(_decanonicalize_attr_value(v) for v in value)
    return value


__all__ = [
    "Term",
    "lift_graph_ir",
    "insert_term_into_egraph",
    "term_to_enode",
    "lower_term_to_graph_ir",
]
