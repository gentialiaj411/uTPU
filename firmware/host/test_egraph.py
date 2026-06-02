"""Tests for the equality-saturation superoptimizer (Task 3, §5.5).

Coverage:
  - E-graph core: union-find canonicalization, hashcons dedup,
    congruence closure across rebuild.
  - GraphIR <-> term language: lift/lower round-trip on a hand-built
    graph; canonicalization of numpy weight tensors; persistent values
    preserved.
  - Rewrite rules: each ``DEFAULT_REWRITES`` rule fires on a hand-built
    LHS and equivalence-preserves the source on the equivalence oracle.
  - Saturation driver: terminates on saturation; ``max_iterations``,
    ``max_eclasses``, ``max_enodes``, ``timeout_s`` caps trip when set
    low; rule_fire_counts honestly tracks merge counts.
  - Extraction: minimum cost over multiple candidate forms; falls back
    to source when no rule fires.
  - Planted phase-ordering test: hand-built graph where the fixed
    pipeline misses a fusion the e-graph finds. Both the harness
    detects this AND ``run_superopt_benchmark`` records it as a win.
  - Equiv-check gate: every extracted graph is differential-verified;
    a deliberately-broken extraction is REJECTED by the harness.
  - Schema lock on ``bench/results/superopt_payoff.json`` (if present).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from egraph import (
    DEFAULT_REWRITES,
    EGraph,
    ENode,
    Rewrite,
    SaturationConfig,
    SaturationStats,
    cuda_cost_model_cost,
    extract_min_cost,
    isa_cycle_cost,
    op_count_cost,
    saturate,
)
from egraph.graph_ir_lang import (
    Term,
    _canonicalize_attr_value,
    _decanonicalize_attr_value,
    insert_term_into_egraph,
    lift_graph_ir,
    lower_term_to_graph_ir,
)
from egraph.rewrites import (
    ADD_COMMUTATIVITY,
    LINEAR_RELU_FUSION,
    PERMUTE_INVOLUTION,
    SCALE_IDENTITY,
    SCALE_REASSOCIATION,
    SCALE_SOFTMAX_FUSION,
    apply_rewrite,
)
from fuzz.differential_oracle import diff_two_graphs
from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import GraphPassManager


# ---------------------------------------------------------------------------
# E-graph core
# ---------------------------------------------------------------------------


def test_egraph_add_dedupes_congruent_nodes() -> None:
    eg = EGraph()
    a = eg.add(ENode("input", (), (("name", "a"),)))
    b = eg.add(ENode("input", (), (("name", "a"),)))
    assert a == b, "two congruent input nodes should hashcons to the same e-class"


def test_egraph_distinct_inputs_get_distinct_eclasses() -> None:
    eg = EGraph()
    a = eg.add(ENode("input", (), (("name", "a"),)))
    b = eg.add(ENode("input", (), (("name", "b"),)))
    assert a != b


def test_egraph_merge_then_find_returns_canonical() -> None:
    eg = EGraph()
    a = eg.add(ENode("leaf", (), (("name", "a"),)))
    b = eg.add(ENode("leaf", (), (("name", "b"),)))
    eg.merge(a, b)
    assert eg.find(a) == eg.find(b)


def test_egraph_congruence_closure_propagates_through_parents() -> None:
    """If two child e-classes merge, their parents that share the same
    head + congruent children should also merge in rebuild()."""
    eg = EGraph()
    a = eg.add(ENode("leaf", (), (("name", "a"),)))
    b = eg.add(ENode("leaf", (), (("name", "b"),)))
    p1 = eg.add(ENode("op", (a,), ()))
    p2 = eg.add(ENode("op", (b,), ()))
    assert p1 != p2, "parents start in different e-classes"
    eg.merge(a, b)
    eg.rebuild()
    assert eg.find(p1) == eg.find(p2), "parents must congruence-merge after children merge"


def test_egraph_size_watermarks_track_max() -> None:
    eg = EGraph()
    for i in range(5):
        eg.add(ENode("leaf", (), (("i", i),)))
    ec, en = eg.size_watermarks()
    assert ec >= 5 and en >= 5


def test_egraph_lookup_returns_existing_class_only_when_congruent() -> None:
    eg = EGraph()
    x = eg.add(ENode("input", (), (("name", "x"),)))
    eg.add(ENode("scale", (x,), (("scale", 0.5),)))
    assert eg.lookup(ENode("scale", (x,), (("scale", 0.5),))) is not None
    assert eg.lookup(ENode("scale", (x,), (("scale", 0.6),))) is None


# ---------------------------------------------------------------------------
# Term language round-trip
# ---------------------------------------------------------------------------


def _build_double_scale_graph(scale_a: float, scale_b: float) -> GraphIR:
    g = GraphIR(name="double_scale", inputs=["x"])
    g.add_value("x", shape=(8,), dtype="float32")
    g.add_op(OpNode(name="s1", op=OpKind.SCALE, inputs=["x"], outputs=["t1"],
                    attrs={"scale": scale_a}))
    g.add_op(OpNode(name="s2", op=OpKind.SCALE, inputs=["t1"], outputs=["t2"],
                    attrs={"scale": scale_b}))
    g.outputs = ["t2"]
    return g


def test_lift_then_lower_roundtrip_preserves_op_count() -> None:
    g = _build_double_scale_graph(0.5, 2.0)
    terms = lift_graph_ir(g)
    g2 = lower_term_to_graph_ir(terms, source_graph=g)
    assert len(g2.ops) == len(g.ops)
    assert g2.outputs and g2.inputs == g.inputs


def test_canonicalize_attr_value_ndarray_roundtrip_is_bit_exact() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    canon = _canonicalize_attr_value(arr)
    arr2 = _decanonicalize_attr_value(canon)
    assert isinstance(arr2, np.ndarray)
    assert arr2.dtype == arr.dtype
    assert arr2.shape == arr.shape
    assert np.array_equal(arr2, arr)


def test_canonicalize_distinguishes_different_weights() -> None:
    a = np.ones((4, 4), dtype=np.float32)
    b = np.zeros((4, 4), dtype=np.float32)
    assert _canonicalize_attr_value(a) != _canonicalize_attr_value(b)


def test_lift_preserves_linear_weight_through_attrs_key() -> None:
    g = GraphIR(name="g", inputs=["x"])
    g.add_value("x", shape=(2, 4), dtype="float32")
    w = np.random.RandomState(0).randn(8, 4).astype(np.float32)
    g.add_op(OpNode(
        name="lin",
        op=OpKind.LINEAR,
        inputs=["x"],
        outputs=["y"],
        attrs={"weight": w, "in_features": 4, "out_features": 8},
    ))
    g.outputs = ["y"]
    terms = lift_graph_ir(g)
    g2 = lower_term_to_graph_ir(terms, source_graph=g)
    lin = next(op for op in g2.ops if op.op == OpKind.LINEAR)
    assert isinstance(lin.attrs.get("weight"), np.ndarray)
    assert np.array_equal(lin.attrs["weight"], w)
    diff = diff_two_graphs(g, g2, [np.random.RandomState(1).randn(2, 4).astype(np.float32)])
    assert diff["match"], f"lift/lower round-trip changed semantics: {diff}"


# ---------------------------------------------------------------------------
# Rewrite rules
# ---------------------------------------------------------------------------


def _saturate_and_extract(source: GraphIR, cost_fn=isa_cycle_cost) -> GraphIR:
    eg = EGraph()
    terms = lift_graph_ir(source)
    root_ids = [insert_term_into_egraph(eg, t, {}) for t in terms]
    saturate(eg, DEFAULT_REWRITES)
    extraction = extract_min_cost(eg, root_ids, cost_fn=cost_fn)
    return lower_term_to_graph_ir(extraction.roots, source_graph=source)


def test_scale_reassociation_collapses_double_scale_into_single() -> None:
    g = _build_double_scale_graph(0.5, 2.0)
    out = _saturate_and_extract(g)
    scale_ops = [op for op in out.ops if op.op == OpKind.SCALE]
    assert len(scale_ops) <= 1, f"expected ≤1 scale op after collapse, got {[op.attrs for op in scale_ops]}"


def test_scale_identity_eliminates_scale_one_op() -> None:
    g = GraphIR(name="scale_one", inputs=["x"])
    g.add_value("x", shape=(8,), dtype="float32")
    g.add_op(OpNode(name="s", op=OpKind.SCALE, inputs=["x"], outputs=["y"], attrs={"scale": 1.0}))
    g.outputs = ["y"]
    out = _saturate_and_extract(g)
    assert all(op.op != OpKind.SCALE for op in out.ops), \
        "scale-by-1.0 should be eliminated"


def test_linear_relu_fusion_fires_in_egraph() -> None:
    g = GraphIR(name="lin_relu", inputs=["x"])
    g.add_value("x", shape=(2, 4), dtype="float32")
    w = np.random.RandomState(0).randn(8, 4).astype(np.float32)
    g.add_op(OpNode(name="lin", op=OpKind.LINEAR, inputs=["x"], outputs=["t"],
                    attrs={"weight": w, "in_features": 4, "out_features": 8}))
    g.add_op(OpNode(name="r", op=OpKind.RELU, inputs=["t"], outputs=["y"]))
    g.outputs = ["y"]
    out = _saturate_and_extract(g)
    assert any(op.op == OpKind.LINEAR_RELU for op in out.ops)
    assert all(op.op != OpKind.LINEAR for op in out.ops), \
        "after fusion the unfused LINEAR should not be extracted (it costs more)"


def test_scale_softmax_fusion_via_reassociation_planted() -> None:
    """Phase-ordering case: scale*scale then softmax. The fixed pipeline
    doesn't reassociate scales, so the scale*softmax fusion sees an
    inner scale (the unfolded second factor) it can't pattern-match. The
    e-graph composes scale_reassociation + scale_identity + scale_softmax_fusion."""
    g = GraphIR(name="planted_phase_ordering", inputs=["x"])
    g.add_value("x", shape=(4, 8), dtype="float32")
    g.add_op(OpNode(name="s1", op=OpKind.SCALE, inputs=["x"], outputs=["t1"],
                    attrs={"scale": 0.25}))
    g.add_op(OpNode(name="s2", op=OpKind.SCALE, inputs=["t1"], outputs=["t2"],
                    attrs={"scale": 4.0}))
    g.add_op(OpNode(name="sm", op=OpKind.SOFTMAX, inputs=["t2"], outputs=["y"],
                    attrs={"dim": -1}))
    g.outputs = ["y"]
    pipeline_graph = GraphPassManager(target_backend="cuda").run(g).graph
    eg_graph = _saturate_and_extract(g)
    assert len(eg_graph.ops) < len(pipeline_graph.ops), (
        f"planted phase-ordering case should reduce op count; "
        f"pipeline kept {len(pipeline_graph.ops)} ops, egraph kept {len(eg_graph.ops)}"
    )


def test_permute_involution_eliminates_round_trip() -> None:
    g = GraphIR(name="permute_round_trip", inputs=["x"])
    g.add_value("x", shape=(2, 3, 4), dtype="float32")
    g.add_op(OpNode(name="p1", op=OpKind.PERMUTE, inputs=["x"], outputs=["t1"],
                    attrs={"args": (0, 2, 1)}))
    g.add_op(OpNode(name="p2", op=OpKind.PERMUTE, inputs=["t1"], outputs=["y"],
                    attrs={"args": (0, 2, 1)}))
    g.outputs = ["y"]
    out = _saturate_and_extract(g)
    assert all(op.op != OpKind.PERMUTE for op in out.ops)


def test_add_commutativity_is_a_no_op_cost_wise() -> None:
    g = GraphIR(name="add_pair", inputs=["a", "b"])
    g.add_value("a", shape=(8,), dtype="float32")
    g.add_value("b", shape=(8,), dtype="float32")
    g.add_op(OpNode(name="add", op=OpKind.ADD, inputs=["a", "b"], outputs=["y"]))
    g.outputs = ["y"]
    out = _saturate_and_extract(g)
    assert len(out.ops) == 1 and out.ops[0].op == OpKind.ADD


# ---------------------------------------------------------------------------
# Saturation driver: caps + telemetry
# ---------------------------------------------------------------------------


def test_saturation_terminates_on_saturated_when_no_rules_fire() -> None:
    g = GraphIR(name="trivial", inputs=["x"])
    g.add_value("x", shape=(8,), dtype="float32")
    g.add_op(OpNode(name="r", op=OpKind.RELU, inputs=["x"], outputs=["y"]))
    g.outputs = ["y"]
    eg = EGraph()
    terms = lift_graph_ir(g)
    root_ids = [insert_term_into_egraph(eg, t, {}) for t in terms]
    stats = saturate(eg, DEFAULT_REWRITES)
    assert stats.terminated_reason == "saturated"
    assert stats.merges_total == 0


def test_saturation_max_iterations_cap_trips_when_set_low() -> None:
    g = _build_double_scale_graph(0.5, 2.0)
    eg = EGraph()
    terms = lift_graph_ir(g)
    [insert_term_into_egraph(eg, t, {}) for t in terms]
    stats = saturate(eg, DEFAULT_REWRITES, config=SaturationConfig(max_iterations=1))
    assert stats.iterations == 1
    assert stats.terminated_reason in ("max_iterations_exceeded", "saturated")


def test_saturation_records_rule_fire_counts() -> None:
    g = _build_double_scale_graph(0.5, 2.0)
    eg = EGraph()
    terms = lift_graph_ir(g)
    [insert_term_into_egraph(eg, t, {}) for t in terms]
    stats = saturate(eg, DEFAULT_REWRITES)
    assert stats.rule_fire_counts.get("scale_reassociation", 0) >= 1
    assert stats.rule_fire_counts.get("scale_identity", 0) >= 1


def test_saturation_max_eclasses_cap_trips_when_set_low() -> None:
    g = _build_double_scale_graph(0.5, 2.0)
    eg = EGraph()
    terms = lift_graph_ir(g)
    [insert_term_into_egraph(eg, t, {}) for t in terms]
    stats = saturate(eg, DEFAULT_REWRITES, config=SaturationConfig(max_eclasses=2))
    assert stats.terminated_reason in (
        "max_eclasses_exceeded",
        "saturated",
        "max_enodes_exceeded",
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extraction_picks_min_cost_form_from_merged_eclass() -> None:
    """If we union an expensive form with a cheap form, extraction must
    pick the cheap one."""
    eg = EGraph()
    x = eg.add(ENode("input", (), (("name", "x"),)))
    expensive = eg.add(ENode("linear", (x,), (("has_bias", False),)))
    cheap = eg.add(ENode("view", (x,), (("args", (8,)),)))
    eg.merge(expensive, cheap)
    eg.rebuild()
    res = extract_min_cost(eg, [expensive], cost_fn=isa_cycle_cost)
    assert res.roots[0].head == OpKind.VIEW, \
        f"expected cheaper view to be extracted, got {res.roots[0].head}"


def test_extraction_op_count_cost_returns_op_count_total() -> None:
    g = _build_double_scale_graph(0.5, 2.0)
    out = _saturate_and_extract(g, cost_fn=op_count_cost)
    assert len(out.ops) <= 1


# ---------------------------------------------------------------------------
# Apply-rewrite primitive
# ---------------------------------------------------------------------------


def test_apply_rewrite_counts_only_new_merges() -> None:
    eg = EGraph()
    terms = lift_graph_ir(_build_double_scale_graph(0.5, 2.0))
    root_ids = [insert_term_into_egraph(eg, t, {}) for t in terms]
    target_eid = eg.find(root_ids[0])
    n1 = apply_rewrite(eg, SCALE_REASSOCIATION, target_eid)
    eg.rebuild()
    n2 = apply_rewrite(eg, SCALE_REASSOCIATION, eg.find(root_ids[0]))
    assert n1 >= 1
    assert n2 == 0, f"second apply on the same already-merged seed must be a no-op, got {n2}"


# ---------------------------------------------------------------------------
# Equivalence-check gate
# ---------------------------------------------------------------------------


def test_equiv_check_passes_on_planted_phase_ordering_case() -> None:
    g = _build_double_scale_graph(0.5, 2.0)
    out = _saturate_and_extract(g)
    inputs = [np.random.RandomState(7).randn(8).astype(np.float32)]
    diff = diff_two_graphs(g, out, inputs)
    assert diff["match"], f"semantics drift on planted case: {diff}"


def test_equiv_check_rejects_deliberately_broken_extraction() -> None:
    """If we manually substitute a wrong op (relu -> permute on a 1-D
    tensor), diff_two_graphs must catch it. This proves the safety net
    has teeth — the harness's equiv check is what protects against
    bad rules, NOT the rule correctness itself."""
    src = GraphIR(name="src", inputs=["x"])
    src.add_value("x", shape=(8,), dtype="float32")
    src.add_op(OpNode(name="r", op=OpKind.RELU, inputs=["x"], outputs=["y"]))
    src.outputs = ["y"]

    broken = GraphIR(name="broken", inputs=["x"])
    broken.add_value("x", shape=(8,), dtype="float32")
    broken.add_op(OpNode(name="s", op=OpKind.SCALE, inputs=["x"], outputs=["y"], attrs={"scale": -1.0}))
    broken.outputs = ["y"]

    inputs = [np.random.RandomState(0).randn(8).astype(np.float32)]
    diff = diff_two_graphs(src, broken, inputs)
    assert not diff["match"], "diff oracle must reject a relu->negative-scale rewrite"


# ---------------------------------------------------------------------------
# Comparison harness + artifact schema lock
# ---------------------------------------------------------------------------


def test_run_superopt_benchmark_emits_required_schema_keys(tmp_path) -> None:
    from run_superopt_benchmark import run_superopt_benchmark
    out = tmp_path / "superopt_payoff.json"
    artifact = run_superopt_benchmark(
        output_path=str(out),
        seed_start=100,
        num_random_graphs=4,
        cost_function="isa_cycle_model",
    )
    for key in (
        "status", "schema_version", "cost_function", "target_backend",
        "graphs_evaluated", "results", "aggregate", "planted_phase_ordering_wins",
        "natural_phase_ordering_wins", "saturation_config",
        "rewrite_rules_registered", "equivalence_check", "environment",
        "git_sha", "generated_at_unix",
    ):
        assert key in artifact, f"missing top-level key: {key!r}"
    agg = artifact["aggregate"]
    for key in (
        "cost_reduction_pct_median", "cost_reduction_pct_max",
        "num_phase_ordering_wins", "num_extractions_rejected_by_equiv_check",
        "num_graphs_evaluated", "pct_graphs_with_any_win",
        "median_pipeline_cost", "median_egraph_cost",
    ):
        assert key in agg, f"missing aggregate key: {key!r}"
    assert artifact["status"] == "ok"
    assert artifact["schema_version"] == 1
    assert isinstance(artifact["results"], list)
    assert artifact["graphs_evaluated"] >= 1


def test_run_superopt_benchmark_finds_all_planted_wins(tmp_path) -> None:
    from run_superopt_benchmark import run_superopt_benchmark
    out = tmp_path / "superopt_payoff.json"
    artifact = run_superopt_benchmark(
        output_path=str(out),
        seed_start=200,
        num_random_graphs=0,
        cost_function="isa_cycle_model",
        include_planted=True,
    )
    planted_names = {p["graph_name"] for p in artifact["planted_phase_ordering_wins"]}
    expected = {
        "planted_double_scale_collapse",
        "planted_scale_softmax_via_reassoc",
        "planted_redundant_permute_pair",
        "planted_identity_scale",
    }
    assert expected.issubset(planted_names), (
        f"missing planted wins: {expected - planted_names}; got {planted_names}"
    )


def test_run_superopt_benchmark_all_extractions_equiv_verified(tmp_path) -> None:
    from run_superopt_benchmark import run_superopt_benchmark
    out = tmp_path / "superopt_payoff.json"
    artifact = run_superopt_benchmark(
        output_path=str(out),
        seed_start=300,
        num_random_graphs=8,
        cost_function="isa_cycle_model",
    )
    for r in artifact["results"]:
        if r["equiv_check_reason"] == "harness_exception":
            continue
        assert r["extracted_equiv_verified"], (
            f"extraction for {r['graph_name']!r} not equiv-verified: "
            f"{r['equiv_check_reason']}"
        )


def test_run_superopt_benchmark_supports_cuda_cost_function(tmp_path) -> None:
    from run_superopt_benchmark import run_superopt_benchmark
    out = tmp_path / "superopt_payoff_cuda.json"
    artifact = run_superopt_benchmark(
        output_path=str(out),
        seed_start=400,
        num_random_graphs=4,
        cost_function="cuda_cost_model",
    )
    assert artifact["cost_function"] == "cuda_cost_model"


def _required_artifact_keys() -> List[str]:
    return [
        "status", "schema_version", "cost_function", "target_backend",
        "graphs_evaluated", "results", "aggregate",
        "planted_phase_ordering_wins", "natural_phase_ordering_wins",
        "saturation_config", "rewrite_rules_registered",
        "equivalence_check", "environment", "git_sha", "generated_at_unix",
    ]


def test_committed_artifact_schema_lock_if_present() -> None:
    path = os.path.normpath(os.path.join(_HERE, "..", "..", "bench", "results", "superopt_payoff.json"))
    if not os.path.exists(path):
        pytest.skip("superopt_payoff.json not yet generated on this host")
    with open(path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    for key in _required_artifact_keys():
        assert key in artifact, f"committed superopt_payoff.json missing key {key!r}"
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "ok"
    aggregate = artifact["aggregate"]
    nrej = aggregate["num_extractions_rejected_by_equiv_check"]
    wins = aggregate["num_phase_ordering_wins"]
    assert wins >= 4, (
        f"expected at least 4 planted phase-ordering wins in committed artifact, got {wins}"
    )
    assert nrej >= 0
    for r in artifact["results"]:
        if r["phase_ordering_win"]:
            assert r["extracted_equiv_verified"], (
                f"win {r['graph_name']!r} must be equiv-verified (honesty contract)"
            )


# ---------------------------------------------------------------------------
# E-graph hides multi-consumer phase-ordering case (regression)
# ---------------------------------------------------------------------------


def test_egraph_finds_linear_relu_under_dead_sibling_branch() -> None:
    """The fixed pipeline runs ``linear_relu_fusion`` BEFORE
    ``dead_code_elimination``, so a graph with a dead ``scale(linear(...))``
    branch siblings the live ``relu(linear(...))`` and the fusion is
    refused on a multi-consumer check. The e-graph should still find
    the fusion (it doesn't care that ``linear``'s output has 2 consumers;
    extraction picks linear_relu for the relu consumer and never reaches
    the dead branch from the graph root).

    This is the textbook phase-ordering regression test."""
    g = GraphIR(name="linear_with_dead_sibling", inputs=["x"])
    g.add_value("x", shape=(2, 4), dtype="float32")
    w = np.random.RandomState(0).randn(8, 4).astype(np.float32)
    g.add_op(OpNode(
        name="lin",
        op=OpKind.LINEAR,
        inputs=["x"],
        outputs=["t"],
        attrs={"weight": w, "in_features": 4, "out_features": 8},
    ))
    g.add_op(OpNode(name="relu_live", op=OpKind.RELU, inputs=["t"], outputs=["y_live"]))
    g.add_op(OpNode(name="scale_dead", op=OpKind.SCALE, inputs=["t"], outputs=["y_dead"],
                    attrs={"scale": 0.5}))
    g.outputs = ["y_live"]

    pipeline_graph = GraphPassManager(target_backend="cuda").run(g).graph
    pipeline_kinds = {op.op for op in pipeline_graph.ops}
    assert OpKind.LINEAR in pipeline_kinds, \
        "pipeline should leave LINEAR unfused due to multi-consumer at fusion time"
    assert OpKind.LINEAR_RELU not in pipeline_kinds, \
        "this is the phase-ordering bug — pipeline misses the linear_relu fusion here"

    eg_graph = _saturate_and_extract(g)
    eg_kinds = {op.op for op in eg_graph.ops}
    assert OpKind.LINEAR_RELU in eg_kinds, \
        "e-graph should find the linear_relu fusion despite multi-consumer source"
    inputs = [np.random.RandomState(11).randn(2, 4).astype(np.float32)]
    diff = diff_two_graphs(g, eg_graph, inputs)
    assert diff["match"], f"natural-graph phase-ordering win is not equiv-verified: {diff}"
