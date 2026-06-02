"""Bounded equality-saturation driver.

Repeatedly applies every rewrite to every existing e-class until a
fixpoint or one of the safety caps fires:

  - ``max_iterations``: outer iteration count cap.
  - ``max_eclasses``: hard cap on the number of live e-classes.
  - ``max_enodes``: hard cap on the total e-node count.
  - ``timeout_s``: wall-clock cap (per-saturate call).

Saturation is deliberately deferred-rebuild: each outer pass scans the
current e-class set, queues all rewrites that fire, performs the merges,
then calls ``EGraph.rebuild()`` once to restore congruence. This is the
``egg`` rebuild-loop pattern adapted for plain Python.

``SaturationStats`` reports the cap that fired (if any), so the comparison
harness can record whether the result is a true saturation or a bounded
truncation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .egraph import EClassId, EGraph
from .rewrites import Rewrite, apply_rewrite


@dataclass(frozen=True)
class SaturationConfig:
    max_iterations: int = 32
    max_eclasses: int = 10_000
    max_enodes: int = 50_000
    timeout_s: float = 10.0


@dataclass
class SaturationStats:
    iterations: int = 0
    merges_total: int = 0
    rule_fire_counts: Dict[str, int] = field(default_factory=dict)
    terminated_reason: str = "saturated"
    max_eclasses_observed: int = 0
    max_enodes_observed: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "iterations": self.iterations,
            "merges_total": self.merges_total,
            "rule_fire_counts": dict(self.rule_fire_counts),
            "terminated_reason": self.terminated_reason,
            "max_eclasses_observed": self.max_eclasses_observed,
            "max_enodes_observed": self.max_enodes_observed,
            "elapsed_s": round(self.elapsed_s, 6),
        }


def saturate(
    eg: EGraph,
    rewrites: Iterable[Rewrite],
    *,
    config: Optional[SaturationConfig] = None,
) -> SaturationStats:
    cfg = config or SaturationConfig()
    rules = list(rewrites)
    stats = SaturationStats(rule_fire_counts={rule.name: 0 for rule in rules})
    started = time.monotonic()

    for it in range(cfg.max_iterations):
        stats.iterations = it + 1
        iter_merges = 0
        eclass_snapshot: List[EClassId] = sorted({eid for eid, _ in eg.classes()})
        for rule in rules:
            for eid in eclass_snapshot:
                canon = eg.find(eid)
                merges = apply_rewrite(eg, rule, canon)
                if merges:
                    stats.rule_fire_counts[rule.name] = stats.rule_fire_counts.get(rule.name, 0) + merges
                    iter_merges += merges
        congr_merges = eg.rebuild()
        iter_merges += congr_merges
        stats.merges_total += iter_merges

        nc = eg.num_eclasses()
        nn = eg.num_enodes()
        stats.max_eclasses_observed = max(stats.max_eclasses_observed, nc)
        stats.max_enodes_observed = max(stats.max_enodes_observed, nn)

        elapsed = time.monotonic() - started
        if nc > cfg.max_eclasses:
            stats.terminated_reason = "max_eclasses_exceeded"
            stats.elapsed_s = elapsed
            return stats
        if nn > cfg.max_enodes:
            stats.terminated_reason = "max_enodes_exceeded"
            stats.elapsed_s = elapsed
            return stats
        if elapsed > cfg.timeout_s:
            stats.terminated_reason = "timeout"
            stats.elapsed_s = elapsed
            return stats
        if iter_merges == 0:
            stats.terminated_reason = "saturated"
            stats.elapsed_s = elapsed
            return stats

    stats.terminated_reason = "max_iterations_exceeded"
    stats.elapsed_s = time.monotonic() - started
    return stats


__all__ = ["SaturationConfig", "SaturationStats", "saturate"]
