"""Superoptimizer comparison harness (Task 3 §5.6).

Pipeline-vs-egraph comparison on a corpus of input graphs:

  source_graph  ──►  GraphPassManager.run(source)             ─►  pipeline_graph  ─►  cost_pipeline
                 ╰─► e-graph lift → saturate → extract → lower ─►  egraph_graph    ─►  cost_egraph
                                                                  ↓
                                                          diff_two_graphs(source, egraph_graph)
                                                                  ↓
                                                          extracted_equiv_verified ∈ {True, False}

If ``extracted_equiv_verified`` is False, the extracted graph is REJECTED
and we record ``egraph_cost_for_comparison = pipeline_cost`` so the
aggregate cost reduction never counts an unverified rewrite as a win.
This is the safety net required by the upgrade plan §5.2 step 5.

Outputs ``bench/results/superopt_payoff.json`` with the schema in
upgrade plan §5.4. Defaults to the *ISA cycle* cost function (exact, ties
to the RTL-corroborated scheduler model). Pass ``--cost-function
cuda_cost_model`` for the CUDA-cost-model variant.

Corpus comes from ``firmware/host/fuzz/graph_generator.generate_program``
(the same generator the metamorphic fuzzer uses) plus a small set of
*planted phase-ordering* graphs (see ``_planted_phase_ordering_corpus``)
that are deliberately constructed to require a reassociation the fixed
pipeline doesn't do.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from egraph import (
    DEFAULT_REWRITES,
    EGraph,
    SaturationConfig,
    cuda_cost_model_cost,
    extract_min_cost,
    isa_cycle_cost,
    op_count_cost,
    saturate,
)
from egraph.extract import _ISA_OP_BASE_COST
from egraph.graph_ir_lang import (
    insert_term_into_egraph,
    lift_graph_ir,
    lower_term_to_graph_ir,
)
from fuzz.differential_oracle import diff_two_graphs
from fuzz.graph_generator import generate_program
from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import GraphPassManager


SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Cost over an extracted/produced GraphIR
# ---------------------------------------------------------------------------


_CUDA_COST_OVERRIDES: Dict[str, float] = {
    OpKind.LINEAR_RELU: _ISA_OP_BASE_COST[OpKind.LINEAR_RELU] * 0.85,
    OpKind.SCALED_SOFTMAX: _ISA_OP_BASE_COST[OpKind.SCALED_SOFTMAX] * 0.82,
}


def graph_cost_isa_cycles(graph: GraphIR) -> float:
    return float(sum(_ISA_OP_BASE_COST.get(op.op, 50.0) for op in graph.ops))


def graph_cost_cuda_model(graph: GraphIR) -> float:
    total = 0.0
    for op in graph.ops:
        base = _CUDA_COST_OVERRIDES.get(op.op, _ISA_OP_BASE_COST.get(op.op, 50.0) * 1.05)
        total += float(base)
    return total


def graph_cost_op_count(graph: GraphIR) -> float:
    return float(len(graph.ops))


_GRAPH_COST_FNS: Dict[str, Callable[[GraphIR], float]] = {
    "isa_cycle_model": graph_cost_isa_cycles,
    "cuda_cost_model": graph_cost_cuda_model,
    "op_count": graph_cost_op_count,
}


_EGRAPH_COST_FNS = {
    "isa_cycle_model": isa_cycle_cost,
    "cuda_cost_model": cuda_cost_model_cost,
    "op_count": op_count_cost,
}


# ---------------------------------------------------------------------------
# Planted phase-ordering corpus (proof-of-teeth)
# ---------------------------------------------------------------------------


def _planted_phase_ordering_corpus() -> List[Tuple[str, GraphIR, List[np.ndarray]]]:
    """Return a list of (name, graph, sample_inputs) tuples representing
    graphs deliberately constructed so the fixed pipeline misses an
    optimization that the e-graph can find by composing a reassociation
    with an existing rule."""
    out: List[Tuple[str, GraphIR, List[np.ndarray]]] = []

    g1 = GraphIR(name="planted_double_scale_collapse", inputs=["x"])
    g1.add_value("x", shape=(8,), dtype="float32")
    g1.add_op(OpNode(name="s1", op=OpKind.SCALE, inputs=["x"], outputs=["t1"],
                     attrs={"scale": 0.5}))
    g1.add_op(OpNode(name="s2", op=OpKind.SCALE, inputs=["t1"], outputs=["t2"],
                     attrs={"scale": 2.0}))
    g1.add_op(OpNode(name="r", op=OpKind.RELU, inputs=["t2"], outputs=["y"]))
    g1.outputs = ["y"]
    out.append((g1.name, g1, [np.random.RandomState(0).randn(8).astype(np.float32)]))

    g2 = GraphIR(name="planted_scale_softmax_via_reassoc", inputs=["x"])
    g2.add_value("x", shape=(4, 8), dtype="float32")
    g2.add_op(OpNode(name="s1", op=OpKind.SCALE, inputs=["x"], outputs=["t1"],
                     attrs={"scale": 0.25}))
    g2.add_op(OpNode(name="s2", op=OpKind.SCALE, inputs=["t1"], outputs=["t2"],
                     attrs={"scale": 4.0}))
    g2.add_op(OpNode(name="sm", op=OpKind.SOFTMAX, inputs=["t2"], outputs=["y"],
                     attrs={"dim": -1}))
    g2.outputs = ["y"]
    out.append((g2.name, g2, [np.random.RandomState(1).randn(4, 8).astype(np.float32)]))

    g3 = GraphIR(name="planted_redundant_permute_pair", inputs=["x"])
    g3.add_value("x", shape=(2, 3, 4), dtype="float32")
    g3.add_op(OpNode(name="p1", op=OpKind.PERMUTE, inputs=["x"], outputs=["t1"],
                     attrs={"args": (0, 2, 1)}))
    g3.add_op(OpNode(name="p2", op=OpKind.PERMUTE, inputs=["t1"], outputs=["t2"],
                     attrs={"args": (0, 2, 1)}))
    g3.add_op(OpNode(name="r", op=OpKind.RELU, inputs=["t2"], outputs=["y"]))
    g3.outputs = ["y"]
    out.append((g3.name, g3, [np.random.RandomState(2).randn(2, 3, 4).astype(np.float32)]))

    g4 = GraphIR(name="planted_identity_scale", inputs=["x"])
    g4.add_value("x", shape=(16,), dtype="float32")
    g4.add_op(OpNode(name="s_one", op=OpKind.SCALE, inputs=["x"], outputs=["t1"],
                     attrs={"scale": 1.0}))
    g4.add_op(OpNode(name="r", op=OpKind.RELU, inputs=["t1"], outputs=["y"]))
    g4.outputs = ["y"]
    out.append((g4.name, g4, [np.random.RandomState(3).randn(16).astype(np.float32)]))

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class PerGraphResult:
    graph_name: str
    seed: Optional[int]
    pipeline_cost: float
    egraph_cost: float
    egraph_cost_for_comparison: float
    cost_reduction_pct: float
    pipeline_op_count: int
    egraph_op_count: int
    phase_ordering_win: bool
    extracted_equiv_verified: bool
    equiv_check_max_abs_error: float
    equiv_check_reason: str
    saturation_terminated_reason: str
    saturation_iterations: int
    saturation_merges_total: int
    rule_fire_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateResult:
    cost_reduction_pct_median: float
    cost_reduction_pct_max: float
    num_phase_ordering_wins: int
    num_extractions_rejected_by_equiv_check: int
    num_graphs_evaluated: int
    pct_graphs_with_any_win: float
    median_pipeline_cost: float
    median_egraph_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_HERE,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _evaluate_one(
    name: str,
    seed: Optional[int],
    source: GraphIR,
    inputs: Sequence[np.ndarray],
    *,
    cost_function: str,
    graph_cost_fn: Callable[[GraphIR], float],
    egraph_cost_fn,
    saturation_cfg: SaturationConfig,
    target_backend: str,
    rtol: float,
    atol: float,
) -> PerGraphResult:
    pm = GraphPassManager(target_backend=target_backend)
    pipeline_graph = pm.run(source).graph
    pipeline_cost = graph_cost_fn(pipeline_graph)
    pipeline_op_count = len(pipeline_graph.ops)

    eg = EGraph()
    terms = lift_graph_ir(source)
    root_ids = [insert_term_into_egraph(eg, t, {}) for t in terms]
    sat_stats = saturate(eg, DEFAULT_REWRITES, config=saturation_cfg)
    extraction = extract_min_cost(eg, root_ids, cost_fn=egraph_cost_fn)
    egraph_graph = lower_term_to_graph_ir(extraction.roots, source_graph=source)
    egraph_cost = graph_cost_fn(egraph_graph)
    egraph_op_count = len(egraph_graph.ops)

    equiv = diff_two_graphs(source, egraph_graph, list(inputs), rtol=rtol, atol=atol)
    extracted_equiv_verified = bool(equiv.get("match", False))
    equiv_max_abs = float(equiv.get("max_abs_error", 0.0))
    equiv_reason = str(equiv.get("reason", "ok") if not extracted_equiv_verified else "ok")

    if extracted_equiv_verified:
        egraph_cost_for_comparison = egraph_cost
    else:
        egraph_cost_for_comparison = pipeline_cost

    if pipeline_cost > 0.0:
        reduction_pct = 100.0 * (pipeline_cost - egraph_cost_for_comparison) / pipeline_cost
    else:
        reduction_pct = 0.0

    phase_ordering_win = (
        extracted_equiv_verified
        and egraph_cost_for_comparison < pipeline_cost
    )

    return PerGraphResult(
        graph_name=name,
        seed=seed,
        pipeline_cost=float(pipeline_cost),
        egraph_cost=float(egraph_cost),
        egraph_cost_for_comparison=float(egraph_cost_for_comparison),
        cost_reduction_pct=float(reduction_pct),
        pipeline_op_count=int(pipeline_op_count),
        egraph_op_count=int(egraph_op_count),
        phase_ordering_win=bool(phase_ordering_win),
        extracted_equiv_verified=bool(extracted_equiv_verified),
        equiv_check_max_abs_error=equiv_max_abs,
        equiv_check_reason=equiv_reason,
        saturation_terminated_reason=sat_stats.terminated_reason,
        saturation_iterations=int(sat_stats.iterations),
        saturation_merges_total=int(sat_stats.merges_total),
        rule_fire_counts=dict(sat_stats.rule_fire_counts),
    )


def run_superopt_benchmark(
    *,
    output_path: str,
    seed_start: int = 0,
    num_random_graphs: int = 64,
    cost_function: str = "isa_cycle_model",
    target_backend: str = "cuda",
    saturation_cfg: Optional[SaturationConfig] = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    include_planted: bool = True,
) -> Dict[str, Any]:
    if cost_function not in _GRAPH_COST_FNS:
        raise ValueError(
            f"unknown cost_function {cost_function!r}; valid: {sorted(_GRAPH_COST_FNS)}"
        )
    graph_cost_fn = _GRAPH_COST_FNS[cost_function]
    egraph_cost_fn = _EGRAPH_COST_FNS[cost_function]
    cfg = saturation_cfg or SaturationConfig()

    results: List[PerGraphResult] = []
    planted_wins: List[Dict[str, Any]] = []

    if include_planted:
        for name, source, inputs in _planted_phase_ordering_corpus():
            r = _evaluate_one(
                name=name,
                seed=None,
                source=source,
                inputs=inputs,
                cost_function=cost_function,
                graph_cost_fn=graph_cost_fn,
                egraph_cost_fn=egraph_cost_fn,
                saturation_cfg=cfg,
                target_backend=target_backend,
                rtol=rtol,
                atol=atol,
            )
            results.append(r)
            if r.phase_ordering_win:
                planted_wins.append({
                    "graph_name": r.graph_name,
                    "pipeline_op_count": r.pipeline_op_count,
                    "egraph_op_count": r.egraph_op_count,
                    "cost_reduction_pct": r.cost_reduction_pct,
                })

    natural_wins: List[Dict[str, Any]] = []
    for i in range(num_random_graphs):
        seed = int(seed_start + i)
        try:
            prog = generate_program(seed)
        except Exception:
            continue
        try:
            r = _evaluate_one(
                name=f"random_seed_{seed}",
                seed=seed,
                source=prog.graph,
                inputs=prog.inputs,
                cost_function=cost_function,
                graph_cost_fn=graph_cost_fn,
                egraph_cost_fn=egraph_cost_fn,
                saturation_cfg=cfg,
                target_backend=target_backend,
                rtol=rtol,
                atol=atol,
            )
        except Exception as exc:
            results.append(PerGraphResult(
                graph_name=f"random_seed_{seed}",
                seed=seed,
                pipeline_cost=0.0,
                egraph_cost=0.0,
                egraph_cost_for_comparison=0.0,
                cost_reduction_pct=0.0,
                pipeline_op_count=0,
                egraph_op_count=0,
                phase_ordering_win=False,
                extracted_equiv_verified=False,
                equiv_check_max_abs_error=0.0,
                equiv_check_reason=f"harness_exception: {type(exc).__name__}: {exc}",
                saturation_terminated_reason="harness_exception",
                saturation_iterations=0,
                saturation_merges_total=0,
                rule_fire_counts={},
            ))
            continue
        results.append(r)
        if r.phase_ordering_win:
            natural_wins.append({
                "graph_name": r.graph_name,
                "seed": seed,
                "pipeline_op_count": r.pipeline_op_count,
                "egraph_op_count": r.egraph_op_count,
                "cost_reduction_pct": r.cost_reduction_pct,
            })

    reductions = [r.cost_reduction_pct for r in results if r.equiv_check_reason != "harness_exception"]
    pipe_costs = [r.pipeline_cost for r in results if r.pipeline_cost > 0]
    eg_costs = [r.egraph_cost_for_comparison for r in results if r.pipeline_cost > 0]
    num_wins = sum(1 for r in results if r.phase_ordering_win)
    num_rejected = sum(1 for r in results if not r.extracted_equiv_verified and r.equiv_check_reason != "harness_exception")
    num_evaluated = sum(1 for r in results if r.equiv_check_reason != "harness_exception")
    pct_any_win = (100.0 * num_wins / num_evaluated) if num_evaluated else 0.0

    aggregate = AggregateResult(
        cost_reduction_pct_median=float(np.median(reductions)) if reductions else 0.0,
        cost_reduction_pct_max=float(max(reductions)) if reductions else 0.0,
        num_phase_ordering_wins=int(num_wins),
        num_extractions_rejected_by_equiv_check=int(num_rejected),
        num_graphs_evaluated=int(num_evaluated),
        pct_graphs_with_any_win=float(pct_any_win),
        median_pipeline_cost=float(np.median(pipe_costs)) if pipe_costs else 0.0,
        median_egraph_cost=float(np.median(eg_costs)) if eg_costs else 0.0,
    )

    artifact: Dict[str, Any] = {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "cost_function": cost_function,
        "target_backend": target_backend,
        "graphs_evaluated": num_evaluated,
        "results": [r.to_dict() for r in results],
        "aggregate": aggregate.to_dict(),
        "planted_phase_ordering_wins": planted_wins,
        "natural_phase_ordering_wins": natural_wins,
        "saturation_config": {
            "max_iterations": cfg.max_iterations,
            "max_eclasses": cfg.max_eclasses,
            "max_enodes": cfg.max_enodes,
            "timeout_s": cfg.timeout_s,
        },
        "rewrite_rules_registered": [rule.name for rule in DEFAULT_REWRITES],
        "equivalence_check": {
            "rtol": rtol,
            "atol": atol,
            "policy": (
                "every extracted graph differential-verified against source via "
                "fuzz.differential_oracle.diff_two_graphs; mismatches REJECT the "
                "extraction and fall back to pipeline cost (no unverified rewrites "
                "are counted as wins)"
            ),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        },
        "git_sha": _git_sha(),
        "generated_at_unix": time.time(),
    }
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    return artifact


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Equality-saturation superoptimizer benchmark (Task 3)")
    default_out = os.path.normpath(os.path.join(_HERE, "..", "..", "bench", "results", "superopt_payoff.json"))
    ap.add_argument("--output", default=default_out,
                    help="output artifact path (default: bench/results/superopt_payoff.json)")
    ap.add_argument("--num-random-graphs", type=int, default=64,
                    help="number of generator-produced graphs to evaluate")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--cost-function", default="isa_cycle_model",
                    choices=sorted(_GRAPH_COST_FNS))
    ap.add_argument("--target-backend", default="cuda",
                    help="backend for GraphPassManager (does not run kernels)")
    ap.add_argument("--max-iterations", type=int, default=32)
    ap.add_argument("--max-eclasses", type=int, default=10000)
    ap.add_argument("--max-enodes", type=int, default=50000)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--no-planted", action="store_true",
                    help="omit the planted phase-ordering corpus")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = SaturationConfig(
        max_iterations=args.max_iterations,
        max_eclasses=args.max_eclasses,
        max_enodes=args.max_enodes,
        timeout_s=args.timeout_s,
    )
    artifact = run_superopt_benchmark(
        output_path=args.output,
        seed_start=args.seed_start,
        num_random_graphs=args.num_random_graphs,
        cost_function=args.cost_function,
        target_backend=args.target_backend,
        saturation_cfg=cfg,
        rtol=args.rtol,
        atol=args.atol,
        include_planted=not args.no_planted,
    )
    agg = artifact["aggregate"]
    print(json.dumps({
        "status": artifact["status"],
        "cost_function": artifact["cost_function"],
        "graphs_evaluated": artifact["graphs_evaluated"],
        "aggregate": agg,
        "rule_fire_counts_total": _aggregate_rule_fires(artifact["results"]),
        "output_path": args.output,
    }, indent=2))
    return 0


def _aggregate_rule_fires(results: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in results:
        for k, v in (r.get("rule_fire_counts") or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


if __name__ == "__main__":
    sys.exit(main())
