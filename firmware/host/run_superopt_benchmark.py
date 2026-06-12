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
    matmul_flop_cost,
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
from graph_passes import GraphPassManager, shape_inference_pass


SCHEMA_VERSION = 2


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


def _shape_tuple(shape: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if shape is None:
        return None
    try:
        return tuple(int(d) for d in shape)
    except TypeError:
        return None


def _batch_matmul_flops(lhs: Optional[Tuple[int, ...]], rhs: Optional[Tuple[int, ...]]) -> float:
    if lhs is None or rhs is None or len(lhs) < 2 or len(rhs) < 2:
        return 0.0
    lhs_batch = tuple(int(d) for d in lhs[:-2])
    rhs_batch = tuple(int(d) for d in rhs[:-2])
    max_rank = max(len(lhs_batch), len(rhs_batch))
    lhs_pad = (1,) * (max_rank - len(lhs_batch)) + lhs_batch
    rhs_pad = (1,) * (max_rank - len(rhs_batch)) + rhs_batch
    batch = 1
    for a, b in zip(lhs_pad, rhs_pad):
        if a == 1:
            batch *= int(b)
        elif b == 1 or a == b:
            batch *= int(a)
        else:
            return 0.0
    return float(2 * batch * int(lhs[-2]) * int(lhs[-1]) * int(rhs[-1]))


def graph_cost_matmul_flops(graph: GraphIR) -> float:
    total = 0.0
    for op in graph.ops:
        if op.op in (OpKind.LINEAR, OpKind.LINEAR_RELU):
            inp = graph.values.get(op.inputs[0])
            out = graph.values.get(op.outputs[0])
            in_shape = _shape_tuple(inp.shape if inp else None)
            out_shape = _shape_tuple(out.shape if out else None)
            if in_shape is None:
                in_features = int(op.attrs.get("in_features", 0))
                if in_features <= 0:
                    continue
                batch = 1
                if out_shape is not None and len(out_shape) >= 1:
                    batch = 1
                n = int(op.attrs.get("out_features", out_shape[-1] if out_shape else 0))
                if n <= 0:
                    continue
                if out_shape is not None and len(out_shape) > 1:
                    batch = int(np.prod(out_shape[:-1]))
                total += float(2 * batch * in_features * n)
                continue
            if len(in_shape) < 1:
                continue
            batch = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
            k = int(in_shape[-1])
            if out_shape is not None and len(out_shape) >= 1:
                n = int(out_shape[-1])
            else:
                n = int(op.attrs.get("out_features", 0))
            if n <= 0:
                continue
            total += float(2 * batch * k * n)
            continue
        if op.op == OpKind.BATCHED_MATMUL:
            lhs = graph.values.get(op.inputs[0])
            rhs = graph.values.get(op.inputs[1])
            lhs_shape = _shape_tuple(lhs.shape if lhs else None)
            rhs_shape = _shape_tuple(rhs.shape if rhs else None)
            total += _batch_matmul_flops(lhs_shape, rhs_shape)
    return float(total)


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


def _realistic_attention_corpus() -> List[Tuple[str, GraphIR, List[np.ndarray], str]]:
    """Real-model-shaped chained matmul corpus for FLOP reduction."""
    out: List[Tuple[str, GraphIR, List[np.ndarray], str]] = []
    specs = [
        ("attention_cross_t32_s128_d16_v8", 1, 2, 32, 128, 16, 8),
        ("attention_cross_t48_s192_d32_v16", 1, 4, 48, 192, 32, 16),
        ("attention_cross_t24_s96_d16_v32", 2, 2, 24, 96, 16, 32),
        ("attention_cross_t64_s128_d8_v8", 1, 4, 64, 128, 8, 8),
    ]
    for idx, (name, b, h, tq, tk, dk, dv) in enumerate(specs):
        g = GraphIR(name=name, inputs=["q", "k_t", "v"])
        g.add_value("q", shape=(b, h, tq, dk), dtype="float32")
        g.add_value("k_t", shape=(b, h, dk, tk), dtype="float32")
        g.add_value("v", shape=(b, h, tk, dv), dtype="float32")
        g.add_op(OpNode(name=f"{name}_m1", op=OpKind.BATCHED_MATMUL, inputs=["q", "k_t"], outputs=["scores"], attrs={}))
        g.add_value("scores", shape=(b, h, tq, tk), dtype="float32")
        g.add_op(OpNode(name=f"{name}_m2", op=OpKind.BATCHED_MATMUL, inputs=["scores", "v"], outputs=["out"], attrs={}))
        g.add_value("out", shape=(b, h, tq, dv), dtype="float32")
        g.outputs = ["out"]
        rs = np.random.RandomState(1000 + idx)
        out.append((
            name,
            g,
            [
                rs.randn(b, h, tq, dk).astype(np.float32),
                rs.randn(b, h, dk, tk).astype(np.float32),
                rs.randn(b, h, tk, dv).astype(np.float32),
            ],
            "attention_cross",
        ))
    return out


def _realistic_stacked_mlp_corpus() -> List[Tuple[str, GraphIR, List[np.ndarray], str]]:
    """Stacked linear chains that look like MLP projection stacks."""
    out: List[Tuple[str, GraphIR, List[np.ndarray], str]] = []
    specs = [
        ("stacked_mlp_64_128_128_64", 8, 64, 128, 128, 64),
        ("stacked_mlp_96_192_192_96", 4, 96, 192, 192, 96),
        ("stacked_mlp_32_128_64_64", 16, 32, 128, 64, 64),
        ("stacked_mlp_128_256_256_128", 4, 128, 256, 256, 128),
    ]
    for idx, (name, m, k, h1, h2, n) in enumerate(specs):
        g = GraphIR(name=name, inputs=["x"])
        g.add_value("x", shape=(m, k), dtype="float32")
        rs = np.random.RandomState(2000 + idx)
        w1 = rs.randn(h1, k).astype(np.float32)
        w2 = rs.randn(h2, h1).astype(np.float32)
        w3 = rs.randn(n, h2).astype(np.float32)
        g.add_op(OpNode(name=f"{name}_l1", op=OpKind.LINEAR, inputs=["x"], outputs=["h1"], attrs={"weight": w1, "in_features": k, "out_features": h1}))
        g.add_value("h1", shape=(m, h1), dtype="float32")
        g.add_op(OpNode(name=f"{name}_l2", op=OpKind.LINEAR, inputs=["h1"], outputs=["h2"], attrs={"weight": w2, "in_features": h1, "out_features": h2}))
        g.add_value("h2", shape=(m, h2), dtype="float32")
        g.add_op(OpNode(name=f"{name}_l3", op=OpKind.LINEAR, inputs=["h2"], outputs=["y"], attrs={"weight": w3, "in_features": h2, "out_features": n}))
        g.add_value("y", shape=(m, n), dtype="float32")
        g.outputs = ["y"]
        out.append((
            name,
            g,
            [rs.randn(m, k).astype(np.float32)],
            "stacked_mlp",
        ))
    return out


def _realistic_corpus() -> List[Tuple[str, GraphIR, List[np.ndarray], str]]:
    return _realistic_attention_corpus() + _realistic_stacked_mlp_corpus()


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


@dataclass
class MeasuredGraphResult:
    graph_name: str
    corpus_class: str
    seed: Optional[int]
    pipeline_flops: float
    egraph_flops: float
    egraph_flops_for_comparison: float
    flop_reduction_pct: float
    pipeline_isa_cycles: float
    egraph_isa_cycles: float
    egraph_isa_cycles_for_comparison: float
    isa_cycle_reduction_pct: float
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
class MeasuredAggregateResult:
    win_rate_pct: float
    flop_reduction_pct_median_on_wins: float
    flop_reduction_pct_max_on_wins: float
    isa_cycle_reduction_pct_median_on_wins: float
    isa_cycle_reduction_pct_max_on_wins: float
    num_phase_ordering_wins: int
    num_extractions_rejected_by_equiv_check: int
    num_graphs_evaluated: int
    median_pipeline_flops: float
    median_egraph_flops: float
    median_pipeline_isa_cycles: float
    median_egraph_isa_cycles: float

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


def _summarize_measured_results(results: List[MeasuredGraphResult]) -> MeasuredAggregateResult:
    evaluated = [r for r in results if r.equiv_check_reason != "harness_exception"]
    wins = [r for r in evaluated if r.phase_ordering_win]
    flop_reductions = [r.flop_reduction_pct for r in wins]
    isa_reductions = [r.isa_cycle_reduction_pct for r in wins]
    pipeline_flops = [r.pipeline_flops for r in evaluated if r.pipeline_flops > 0]
    egraph_flops = [r.egraph_flops_for_comparison for r in evaluated if r.pipeline_flops > 0]
    pipeline_isa = [r.pipeline_isa_cycles for r in evaluated if r.pipeline_isa_cycles > 0]
    egraph_isa = [r.egraph_isa_cycles_for_comparison for r in evaluated if r.pipeline_isa_cycles > 0]
    num_rejected = sum(1 for r in evaluated if not r.extracted_equiv_verified)
    num_wins = len(wins)
    num_eval = len(evaluated)
    win_rate = (100.0 * num_wins / num_eval) if num_eval else 0.0
    return MeasuredAggregateResult(
        win_rate_pct=float(win_rate),
        flop_reduction_pct_median_on_wins=float(np.median(flop_reductions)) if flop_reductions else 0.0,
        flop_reduction_pct_max_on_wins=float(max(flop_reductions)) if flop_reductions else 0.0,
        isa_cycle_reduction_pct_median_on_wins=float(np.median(isa_reductions)) if isa_reductions else 0.0,
        isa_cycle_reduction_pct_max_on_wins=float(max(isa_reductions)) if isa_reductions else 0.0,
        num_phase_ordering_wins=int(num_wins),
        num_extractions_rejected_by_equiv_check=int(num_rejected),
        num_graphs_evaluated=int(num_eval),
        median_pipeline_flops=float(np.median(pipeline_flops)) if pipeline_flops else 0.0,
        median_egraph_flops=float(np.median(egraph_flops)) if egraph_flops else 0.0,
        median_pipeline_isa_cycles=float(np.median(pipeline_isa)) if pipeline_isa else 0.0,
        median_egraph_isa_cycles=float(np.median(egraph_isa)) if egraph_isa else 0.0,
    )


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


def _evaluate_measured_one(
    name: str,
    corpus_class: str,
    seed: Optional[int],
    source: GraphIR,
    inputs: Sequence[np.ndarray],
    *,
    saturation_cfg: SaturationConfig,
    target_backend: str,
    rtol: float,
    atol: float,
) -> MeasuredGraphResult:
    pm = GraphPassManager(target_backend=target_backend)
    pipeline_graph = pm.run(source).graph
    shape_inference_pass(pipeline_graph)
    pipeline_flops = graph_cost_matmul_flops(pipeline_graph)
    pipeline_isa = graph_cost_isa_cycles(pipeline_graph)
    pipeline_op_count = len(pipeline_graph.ops)

    eg = EGraph()
    terms = lift_graph_ir(source)
    root_ids = [insert_term_into_egraph(eg, t, {}) for t in terms]
    sat_stats = saturate(eg, DEFAULT_REWRITES, config=saturation_cfg)
    extraction = extract_min_cost(eg, root_ids, cost_fn=matmul_flop_cost)
    egraph_graph = lower_term_to_graph_ir(extraction.roots, source_graph=source)
    shape_inference_pass(egraph_graph)
    egraph_flops = graph_cost_matmul_flops(egraph_graph)
    egraph_isa = graph_cost_isa_cycles(egraph_graph)
    egraph_op_count = len(egraph_graph.ops)

    equiv = diff_two_graphs(source, egraph_graph, list(inputs), rtol=rtol, atol=atol)
    extracted_equiv_verified = bool(equiv.get("match", False))
    equiv_max_abs = float(equiv.get("max_abs_error", 0.0))
    equiv_reason = str(equiv.get("reason", "ok") if not extracted_equiv_verified else "ok")

    if extracted_equiv_verified:
        egraph_flops_for_comparison = egraph_flops
        egraph_isa_for_comparison = egraph_isa
    else:
        egraph_flops_for_comparison = pipeline_flops
        egraph_isa_for_comparison = pipeline_isa

    flop_reduction_pct = 0.0
    if pipeline_flops > 0.0:
        flop_reduction_pct = 100.0 * (pipeline_flops - egraph_flops_for_comparison) / pipeline_flops
    isa_cycle_reduction_pct = 0.0
    if pipeline_isa > 0.0:
        isa_cycle_reduction_pct = 100.0 * (pipeline_isa - egraph_isa_for_comparison) / pipeline_isa

    phase_ordering_win = extracted_equiv_verified and egraph_flops_for_comparison < pipeline_flops

    return MeasuredGraphResult(
        graph_name=name,
        corpus_class=corpus_class,
        seed=seed,
        pipeline_flops=float(pipeline_flops),
        egraph_flops=float(egraph_flops),
        egraph_flops_for_comparison=float(egraph_flops_for_comparison),
        flop_reduction_pct=float(flop_reduction_pct),
        pipeline_isa_cycles=float(pipeline_isa),
        egraph_isa_cycles=float(egraph_isa),
        egraph_isa_cycles_for_comparison=float(egraph_isa_for_comparison),
        isa_cycle_reduction_pct=float(isa_cycle_reduction_pct),
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
    include_realistic: bool = True,
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

    realistic_results: List[MeasuredGraphResult] = []
    realistic_corpus_payload: Optional[Dict[str, Any]] = None
    if include_realistic:
        for idx, (name, source, inputs, corpus_class) in enumerate(_realistic_corpus()):
            realistic_results.append(_evaluate_measured_one(
                name=name,
                corpus_class=corpus_class,
                seed=1000 + idx,
                source=source,
                inputs=inputs,
                saturation_cfg=cfg,
                target_backend=target_backend,
                rtol=rtol,
                atol=atol,
            ))
        class_breakdown: Dict[str, Dict[str, Any]] = {}
        for corpus_class in sorted({r.corpus_class for r in realistic_results}):
            class_results = [r for r in realistic_results if r.corpus_class == corpus_class]
            class_breakdown[corpus_class] = _summarize_measured_results(class_results).to_dict()
        realistic_aggregate = _summarize_measured_results(realistic_results)
        realistic_corpus_payload = {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "corpus_name": "realistic",
            "cost_function": "matmul_flop_model",
            "target_backend": target_backend,
            "graphs_evaluated": realistic_aggregate.num_graphs_evaluated,
            "results": [r.to_dict() for r in realistic_results],
            "aggregate": realistic_aggregate.to_dict(),
            "class_breakdown": class_breakdown,
            "rewrites_rejected_by_equivalence_gate": realistic_aggregate.num_extractions_rejected_by_equiv_check,
            "rewrite_rules_registered": [rule.name for rule in DEFAULT_REWRITES],
            "equivalence_check": {
                "rtol": rtol,
                "atol": atol,
                "policy": (
                    "every extracted graph differential-verified against source via "
                    "fuzz.differential_oracle.diff_two_graphs; mismatches REJECT the "
                    "extraction and fall back to pipeline FLOP/cycle cost"
                ),
            },
            "generated_at_unix": time.time(),
        }

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
    if realistic_corpus_payload is not None:
        artifact["realistic_corpus"] = realistic_corpus_payload
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
    ap.add_argument("--no-realistic", action="store_true",
                    help="omit the realistic FLOP-focused corpus")
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
        include_realistic=not args.no_realistic,
    )
    agg = artifact["aggregate"]
    payload = {
        "status": artifact["status"],
        "cost_function": artifact["cost_function"],
        "graphs_evaluated": artifact["graphs_evaluated"],
        "aggregate": agg,
        "rule_fire_counts_total": _aggregate_rule_fires(artifact["results"]),
        "output_path": args.output,
    }
    if "realistic_corpus" in artifact:
        payload["realistic_corpus"] = {
            "graphs_evaluated": artifact["realistic_corpus"]["graphs_evaluated"],
            "aggregate": artifact["realistic_corpus"]["aggregate"],
            "class_breakdown": artifact["realistic_corpus"]["class_breakdown"],
        }
    print(json.dumps(payload, indent=2))
    return 0


def _aggregate_rule_fires(results: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in results:
        for k, v in (r.get("rule_fire_counts") or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


if __name__ == "__main__":
    sys.exit(main())
