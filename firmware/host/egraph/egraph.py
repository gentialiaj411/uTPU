"""Core e-graph: union-find + hashcons + congruence closure.

An e-graph stores equivalence classes (``EClass``) of e-nodes (``ENode``).
An e-node is a head symbol applied to a tuple of e-class IDs (children).
Two e-nodes are *congruent* if they have the same head and their child
e-class IDs are equal under the current union-find. Congruence closure
ensures that whenever children become equivalent, their parents do too.

Public surface (small + auditable):
  - ``EGraph.add(node)`` : insert an e-node, return its canonical e-class id.
  - ``EGraph.merge(a, b)`` : union two e-class ids; return canonical id.
  - ``EGraph.rebuild()`` : restore congruence after a batch of merges.
  - ``EGraph.find(eid)`` : canonical id for ``eid``.
  - ``EGraph.classes()`` : iterable of canonical (eid, [enodes]) pairs.

Design notes:
  - We use a *deferred rebuild* style (similar to ``egg``'s rebuild loop):
    merges populate a worklist, ``rebuild`` re-canonicalizes parent nodes,
    detects new congruences, and re-merges to fixpoint.
  - Hashcons key is ``(head, tuple_of_canonical_child_ids, attrs_key)`` where
    ``attrs_key`` is an opaque hashable produced by the term language. This
    lets two ``scale`` ops with the same ``scale=0.5`` attr congruence-merge,
    but two ``scale`` ops with different scales stay distinct.
  - All public methods are pure-Python; no third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

EClassId = int


@dataclass(frozen=True)
class ENode:
    """An e-node: a head symbol applied to a tuple of e-class IDs.

    ``attrs_key`` is an arbitrary hashable carrying op-specific payload
    (e.g. ``("scale", 0.5)`` for a ``SCALE`` op, ``("permute", (0, 2, 1))``
    for a permute). It must be canonical for semantic equivalence:
    equivalent ops must produce equal ``attrs_key`` and inequivalent ops
    must produce different ``attrs_key``.
    """

    head: str
    children: Tuple[EClassId, ...] = ()
    attrs_key: Tuple[Any, ...] = ()

    def canonical(self, find: "callable") -> "ENode":
        """Return a copy with all child e-class ids canonicalized."""
        return ENode(
            head=self.head,
            children=tuple(find(c) for c in self.children),
            attrs_key=self.attrs_key,
        )


@dataclass
class EClass:
    eid: EClassId
    nodes: List[ENode] = field(default_factory=list)
    parents: List[Tuple[ENode, EClassId]] = field(default_factory=list)


class EGraph:
    def __init__(self) -> None:
        self._uf_parent: Dict[EClassId, EClassId] = {}
        self._uf_rank: Dict[EClassId, int] = {}
        self._classes: Dict[EClassId, EClass] = {}
        self._hashcons: Dict[ENode, EClassId] = {}
        self._next_id: EClassId = 0
        self._worklist: List[EClassId] = []
        self._size_high_watermark: int = 0
        self._enode_count_high_watermark: int = 0

    def add(self, node: ENode) -> EClassId:
        """Insert ``node``; return canonical e-class id. Idempotent
        (re-adding a congruent node returns the existing e-class)."""
        canon = node.canonical(self.find)
        existing = self._hashcons.get(canon)
        if existing is not None:
            return self.find(existing)
        new_id = self._next_id
        self._next_id += 1
        self._uf_parent[new_id] = new_id
        self._uf_rank[new_id] = 0
        self._classes[new_id] = EClass(eid=new_id, nodes=[canon])
        self._hashcons[canon] = new_id
        for child in canon.children:
            self._classes[child].parents.append((canon, new_id))
        self._update_size_watermark()
        return new_id

    def find(self, eid: EClassId) -> EClassId:
        root = eid
        while self._uf_parent[root] != root:
            root = self._uf_parent[root]
        cursor = eid
        while self._uf_parent[cursor] != root:
            nxt = self._uf_parent[cursor]
            self._uf_parent[cursor] = root
            cursor = nxt
        return root

    def merge(self, a: EClassId, b: EClassId) -> EClassId:
        """Union the e-classes containing ``a`` and ``b``. Returns the
        surviving canonical id. Caller must invoke ``rebuild()`` (or
        ``saturate`` will do it) to restore congruence."""
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self._uf_rank[ra] < self._uf_rank[rb]:
            ra, rb = rb, ra
        self._uf_parent[rb] = ra
        if self._uf_rank[ra] == self._uf_rank[rb]:
            self._uf_rank[ra] += 1
        cls_a = self._classes[ra]
        cls_b = self._classes[rb]
        cls_a.nodes.extend(cls_b.nodes)
        cls_a.parents.extend(cls_b.parents)
        del self._classes[rb]
        self._worklist.append(ra)
        return ra

    def rebuild(self) -> int:
        """Process the merge worklist to restore congruence closure.

        Returns the number of new congruence-driven merges that fired
        (useful for saturation telemetry)."""
        new_merges = 0
        while self._worklist:
            todo = list({self.find(eid) for eid in self._worklist})
            self._worklist = []
            for eid in todo:
                cls = self._classes.get(eid)
                if cls is None:
                    continue
                new_parents: List[Tuple[ENode, EClassId]] = []
                seen_canon: Dict[ENode, EClassId] = {}
                for (old_node, parent_eid) in cls.parents:
                    canon = old_node.canonical(self.find)
                    parent_canon = self.find(parent_eid)
                    prior = seen_canon.get(canon)
                    if prior is None:
                        seen_canon[canon] = parent_canon
                        new_parents.append((canon, parent_canon))
                    else:
                        merged = self.merge(prior, parent_canon)
                        new_merges += 1
                        if merged not in seen_canon.values():
                            seen_canon[canon] = merged
                cls.parents = new_parents
        self._recanonicalize_hashcons()
        self._update_size_watermark()
        return new_merges

    def _recanonicalize_hashcons(self) -> None:
        """Recompute hashcons from current canonical state. Called only
        from rebuild() so external callers don't see a stale hashcons."""
        new_hashcons: Dict[ENode, EClassId] = {}
        for eid, cls in list(self._classes.items()):
            unique_nodes: Dict[ENode, None] = {}
            for node in cls.nodes:
                canon_node = node.canonical(self.find)
                unique_nodes.setdefault(canon_node, None)
                new_hashcons[canon_node] = self.find(eid)
            cls.nodes = list(unique_nodes.keys())
        self._hashcons = new_hashcons

    def classes(self) -> Iterable[Tuple[EClassId, List[ENode]]]:
        """Yield (canonical_eid, [enodes]) for every live e-class."""
        for eid, cls in self._classes.items():
            yield self.find(eid), list(cls.nodes)

    def class_nodes(self, eid: EClassId) -> List[ENode]:
        """Nodes for the canonical class of ``eid``."""
        return list(self._classes[self.find(eid)].nodes)

    def num_eclasses(self) -> int:
        return len(self._classes)

    def num_enodes(self) -> int:
        return sum(len(cls.nodes) for cls in self._classes.values())

    def size_watermarks(self) -> Tuple[int, int]:
        return self._size_high_watermark, self._enode_count_high_watermark

    def _update_size_watermark(self) -> None:
        ec = len(self._classes)
        en = sum(len(cls.nodes) for cls in self._classes.values())
        if ec > self._size_high_watermark:
            self._size_high_watermark = ec
        if en > self._enode_count_high_watermark:
            self._enode_count_high_watermark = en

    def lookup(self, node: ENode) -> Optional[EClassId]:
        """Return the e-class id for ``node`` if congruent, else None.
        Useful for rewrite-rule matching without triggering an insert."""
        canon = node.canonical(self.find)
        existing = self._hashcons.get(canon)
        return None if existing is None else self.find(existing)
