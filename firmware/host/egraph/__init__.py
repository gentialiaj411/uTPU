"""E-graph + equality saturation + cost-based extraction (Task 3).

Public surface intentionally small. The headline entry points are:
  - ``EGraph`` (egraph.py): the union-find + hashcons + congruence-closure core.
  - ``saturate`` (saturate.py): bounded rewrite-rule application.
  - ``extract_min_cost`` (extract.py): greedy min-cost program extraction.
  - ``lift_graph_ir`` / ``lower_term_to_graph_ir`` (graph_ir_lang.py): the
    bridge between the GraphIR used elsewhere in the project and the flat
    term language the e-graph operates on.

Honesty contract: rule correctness is NOT load-bearing. Every extracted
program is differential-verified against its source via
``firmware/host/fuzz/differential_oracle.diff_two_graphs`` before being
accepted; mismatches reject the extraction and the source is kept.
"""

from .egraph import ENode, EClassId, EGraph
from .extract import (
    CostFunction,
    ExtractionResult,
    extract_min_cost,
    isa_cycle_cost,
    cuda_cost_model_cost,
    op_count_cost,
)
from .graph_ir_lang import (
    Term,
    lift_graph_ir,
    lower_term_to_graph_ir,
)
from .rewrites import (
    Rewrite,
    DEFAULT_REWRITES,
    apply_rewrite,
)
from .saturate import (
    SaturationConfig,
    SaturationStats,
    saturate,
)

__all__ = [
    "CostFunction",
    "DEFAULT_REWRITES",
    "EClassId",
    "EGraph",
    "ENode",
    "ExtractionResult",
    "Rewrite",
    "SaturationConfig",
    "SaturationStats",
    "Term",
    "apply_rewrite",
    "cuda_cost_model_cost",
    "extract_min_cost",
    "isa_cycle_cost",
    "lift_graph_ir",
    "lower_term_to_graph_ir",
    "op_count_cost",
    "saturate",
]
