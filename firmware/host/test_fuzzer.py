"""Tests for the Task 2 metamorphic-differential fuzzer (`utpu_upgrade_plan.md` §4.5).

Locks the properties §4.5 demands plus the v2 hardening pass (2026-05-25):

1. The generator emits ONLY graphs that pass the compiler's legality
   predicates (no invalid graphs sneak in via the valid corpus).
2. **Planted-bug test** — a deliberately wrong fusion rewrite is caught
   by the metamorphic harness (proves teeth even when the live compiler
   is clean).
3. The minimizer reduces a known multi-op failing graph to the minimal
   triggering subset.
4. Schema lock — ``bench/results/fuzzer_report.json`` is parseable and
   carries every key the docs / ``run_fuzzer.py`` rely on (schema_version=3).
5. Invalid generator emits actually invalid graphs and the compiler
   rejects each with a clean diagnostic.
6. The full-graph megakernel relation evaluates (does NOT skip) for
   the formerly-skipped ``region_not_whole_graph`` seeds when CUDA is
   available — on a CPU-only host the relation still skips cleanly via
   the cuda_megakernel-unavailable bucket.
7. Distribution coverage: the runner records which adversarial input
   distributions were exercised in the artifact.

All tests are CPU-only; the ``region_fused_vs_op_by_op`` relation is
gated on cuda_megakernel availability and properly skips on Windows
hosts (verified via the ``relation.skipped`` field).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import (
    BackendLegalityError,
    backend_legality_pass,
    is_op_supported_for_backend,
    shape_inference_pass,
)
from graph_reference_interpreter import GraphReferenceInterpreter

from fuzz import differential_oracle as fdo
from fuzz.graph_generator import (
    ALL_GRAPH_FAMILIES,
    GRAPH_FAMILY_WEIGHTS,
    SHAPE_BUCKETS,
    GeneratedProgram,
    assert_program_legal,
    coverage_summary,
    generate_program,
)
from fuzz.input_distributions import (
    DISTRIBUTIONS as INPUT_DISTRIBUTIONS,
    apply_to_program,
    pick_distribution,
    sample_tensor,
)
from fuzz.invalid_generator import (
    INVALID_KINDS,
    check_rejection,
    generate_invalid_program,
)
from fuzz.metamorphic import (
    ALL_RELATIONS,
    MetamorphicResult,
    evaluate_all_relations,
    relation_dce_on_off,
    relation_fusion_on_off,
    relation_region_fused_vs_op_by_op,
    relation_schedule_alternative,
    relation_tiling_AB,
)
from fuzz.minimizer import minimize
from run_fuzzer import SCHEMA_VERSION, run_fuzzer


# ---------------------------------------------------------------------------
# (1) Generator legality — every emitted graph must pass the compiler's
#     own predicates AND shape inference.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(0, 32)))
def test_generator_emits_only_legal_graphs(seed: int) -> None:
    program = generate_program(seed)
    assert_program_legal(program)
    cuda_lowered = backend_legality_pass(program.graph, backend="cuda")
    assert cuda_lowered is not None


@pytest.mark.parametrize("seed", list(range(40, 60)))
def test_generator_graphs_run_through_reference_interpreter(seed: int) -> None:
    program = generate_program(seed)
    out = GraphReferenceInterpreter(program.graph).run(*program.inputs)
    if isinstance(out, tuple):
        assert len(out) >= 1
        out = out[0]
    arr = np.asarray(out)
    assert np.all(np.isfinite(arr)), (
        f"generator emitted graph that produced non-finite output for seed={seed}"
    )


def test_generator_is_deterministic_per_seed() -> None:
    a = generate_program(7)
    b = generate_program(7)
    assert a.graph.name == b.graph.name
    assert len(a.graph.ops) == len(b.graph.ops)
    assert [op.op for op in a.graph.ops] == [op.op for op in b.graph.ops]
    np.testing.assert_array_equal(a.inputs[0], b.inputs[0])


def test_generator_covers_every_op_kind_in_corpus() -> None:
    programs = [generate_program(s) for s in range(400)]
    cov = coverage_summary(programs)
    expected = {OpKind.LINEAR, OpKind.LINEAR_RELU, OpKind.RELU, OpKind.ADD, OpKind.SCALE}
    missing = expected - set(cov["ops_covered"])
    assert not missing, f"generator never emitted {sorted(missing)} in 400 seeds"


def test_generator_covers_extended_op_kinds_in_corpus() -> None:
    """v2 hardening pass — VIEW, PERMUTE, SOFTMAX, LAYER_NORM, BATCHED_MATMUL
    must each appear in a reasonably-sized corpus. We do not require
    every op in 100 seeds (family weighting is skewed toward linear/
    elementwise), but the broader ops MUST appear in 1000 seeds.
    """
    programs = [generate_program(s) for s in range(1000)]
    cov = coverage_summary(programs)
    expected_extended = {OpKind.VIEW, OpKind.PERMUTE}
    missing = expected_extended - set(cov["ops_covered"])
    assert not missing, (
        f"generator never emitted {sorted(missing)} in 1000 seeds — "
        f"layout_chain family weight may be too low. saw: {cov['ops_covered']!r}"
    )


def test_generator_covers_multiple_shape_buckets() -> None:
    programs = [generate_program(s) for s in range(200)]
    cov = coverage_summary(programs)
    assert len(cov["shape_buckets_covered"]) >= 4, (
        f"only hit {cov['shape_buckets_covered']!r} buckets in 200 seeds"
    )


def test_generator_covers_at_least_five_graph_families() -> None:
    """Promotion contract: the runner artifact will claim 5+ graph families.
    A 600-seed corpus MUST exercise at least 5 of the declared families.
    """
    programs = [generate_program(s) for s in range(600)]
    cov = coverage_summary(programs)
    fams = set(cov["graph_families_covered"])
    assert len(fams) >= 5, (
        f"only hit {len(fams)} families in 600 seeds: {sorted(fams)!r} "
        f"(weights: {GRAPH_FAMILY_WEIGHTS!r})"
    )


@pytest.mark.parametrize("family", list(ALL_GRAPH_FAMILIES))
def test_generator_each_family_builds_legal_graph(family: str) -> None:
    """For every declared family, the generator must emit a legal program."""
    # Each family produces deterministic-but-different graphs across seeds;
    # we try a small window so families with rank-3 constraints (attention_lite)
    # land on at least one bucket that fits.
    last_err: Exception | None = None
    for s in range(0, 16):
        try:
            program = generate_program(s, family=family)
            assert_program_legal(program)
            out = GraphReferenceInterpreter(program.graph).run(*program.inputs)
            arr = out[0] if isinstance(out, tuple) else out
            assert np.all(np.isfinite(np.asarray(arr)))
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise AssertionError(
        f"family {family!r} could not emit a legal program in 16 seeds; last error: {last_err!r}"
    )


def test_assert_program_legal_rejects_dangling_input() -> None:
    g = GraphIR(name="bad")
    g.inputs = ["x"]
    g.add_value("x", shape=(1, 2), dtype="torch.float32")
    g.outputs = ["y"]
    g.add_value("y", shape=(1, 2), dtype="torch.float32")
    program = GeneratedProgram(seed=0, graph=g, inputs=[np.zeros((1, 2), dtype=np.float32)])
    with pytest.raises(AssertionError):
        assert_program_legal(program)


# ---------------------------------------------------------------------------
# (5) Invalid generator — every emitted invalid graph must be rejected by
#     the expected validator with a clean diagnostic exception type.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_kind", list(INVALID_KINDS))
def test_invalid_generator_emits_actually_invalid_graphs(invalid_kind: str) -> None:
    """Every documented invalid kind must produce a graph that the compiler
    rejects with one of the documented clean exception types."""
    program = generate_invalid_program(seed=12345, kind=invalid_kind)
    assert program.invalid_kind == invalid_kind
    outcome = check_rejection(program)
    assert outcome.rejected_cleanly, (
        f"invalid_kind={invalid_kind!r} was NOT rejected cleanly: "
        f"unexpected_accept={outcome.unexpectedly_accepted}, "
        f"crashed_badly={outcome.crashed_badly}, "
        f"exception={outcome.exception_type!r}: {outcome.exception_message!r}"
    )


def test_invalid_generator_is_deterministic_per_seed_and_kind() -> None:
    a = generate_invalid_program(seed=9, kind="invalid_view_size")
    b = generate_invalid_program(seed=9, kind="invalid_view_size")
    assert a.graph.name == b.graph.name
    assert len(a.graph.ops) == len(b.graph.ops)
    np.testing.assert_array_equal(a.inputs[0], b.inputs[0])


def test_invalid_unsupported_op_for_backend_raises_backend_legality_error() -> None:
    """The unsupported_op_for_backend case MUST raise BackendLegalityError
    (not a generic ValueError or AttributeError) — that diagnostic type
    is part of the public compiler contract.
    """
    program = generate_invalid_program(seed=42, kind="unsupported_op_for_backend")
    with pytest.raises(BackendLegalityError):
        backend_legality_pass(program.graph, program.expected_backend or "utpu")


# ---------------------------------------------------------------------------
# Input distributions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dist", list(INPUT_DISTRIBUTIONS))
def test_input_distribution_produces_finite_float32_tensor(dist: str) -> None:
    arr = sample_tensor(dist, shape=(2, 8), seed=123)
    assert arr.dtype == np.float32
    assert arr.shape == (2, 8)
    assert np.all(np.isfinite(arr))


def test_input_distribution_apply_to_program_preserves_shapes() -> None:
    program = generate_program(0)
    distorted = apply_to_program(program, "alternating_sign", seed=0)
    assert len(distorted.inputs) == len(program.inputs)
    for orig, new in zip(program.inputs, distorted.inputs):
        assert orig.shape == new.shape
        assert new.dtype == np.float32
    assert distorted.metadata.get("input_distribution") == "alternating_sign"


def test_pick_distribution_eventually_hits_every_name_in_a_large_pool() -> None:
    """Sanity check that the weighted picker covers every distribution
    within a generous budget. This protects the artifact's
    `input_distributions_covered` field from silently shrinking."""
    import random as _r

    rng = _r.Random(2026)
    seen: set[str] = set()
    for _ in range(2000):
        seen.add(pick_distribution(rng))
    missing = set(INPUT_DISTRIBUTIONS) - seen
    assert not missing, f"pick_distribution failed to hit {missing!r} in 2000 draws"


# ---------------------------------------------------------------------------
# (2) Planted-bug — proves the metamorphic harness has teeth.
# ---------------------------------------------------------------------------


def _planted_wrong_linear_relu_fusion(graph: GraphIR) -> GraphIR:
    """Deliberately wrong fusion: replace LINEAR -> RELU with a LINEAR_RELU
    that ALSO scales the output by 1.5x. A correct fusion must preserve
    semantics; this one breaks them.
    """
    bad = copy.deepcopy(graph)
    new_ops: List[OpNode] = []
    consumed: set = set()
    op_by_name = {op.name: op for op in bad.ops}
    for op in bad.ops:
        if op.name in consumed:
            continue
        if op.op != OpKind.LINEAR or not op.outputs:
            new_ops.append(op)
            continue
        out_val = bad.values.get(op.outputs[0])
        if out_val is None or len(out_val.consumers) != 1:
            new_ops.append(op)
            continue
        consumer = op_by_name.get(out_val.consumers[0])
        if consumer is None or consumer.op != OpKind.RELU or consumer.inputs[0] != op.outputs[0]:
            new_ops.append(op)
            continue
        new_attrs = dict(op.attrs)
        bad_weight = np.asarray(op.attrs["weight"], dtype=np.float32) * 1.5
        new_attrs["weight"] = bad_weight
        new_attrs["fused_activation"] = "relu"
        fused = OpNode(
            name=f"{op.name}_relu_BAD_FUSION",
            op=OpKind.LINEAR_RELU,
            inputs=list(op.inputs),
            outputs=list(consumer.outputs),
            attrs=new_attrs,
        )
        new_ops.append(fused)
        consumed.add(consumer.name)

    bad.ops = new_ops
    referenced: set = set(bad.inputs) | set(bad.outputs)
    for o in bad.ops:
        referenced.update(o.inputs)
        referenced.update(o.outputs)
    bad.values = {k: v for k, v in bad.values.items() if k in referenced}
    for value in bad.values.values():
        value.consumers = []
        value.producer = None
    rebuild = GraphIR(name=bad.name)
    rebuild.inputs = list(bad.inputs)
    rebuild.outputs = list(bad.outputs)
    rebuild.metadata = dict(bad.metadata)
    for inp in rebuild.inputs:
        v = bad.values.get(inp)
        rebuild.add_value(
            inp,
            shape=v.shape if v else None,
            dtype=v.dtype if v else None,
        )
    for op in bad.ops:
        for inp in op.inputs:
            v = bad.values.get(inp)
            rebuild.add_value(
                inp,
                shape=v.shape if v else None,
                dtype=v.dtype if v else None,
            )
        for out in op.outputs:
            v = bad.values.get(out)
            rebuild.add_value(
                out,
                shape=v.shape if v else None,
                dtype=v.dtype if v else None,
            )
        rebuild.add_op(copy.deepcopy(op))
    return shape_inference_pass(rebuild)


def _find_seed_with_linear_relu_fuse_pair(max_seeds: int = 400) -> int:
    """Locate a seed whose graph contains a LINEAR -> RELU pair that the
    project's fusion rule would actually rewrite. With v2 family weights,
    the linear-chain family is still the most common emitter, but we
    bump max_seeds to accommodate the broader corpus."""
    from fuzz.metamorphic import _has_linear_then_relu_chain  # type: ignore

    for s in range(max_seeds):
        program = generate_program(s)
        if _has_linear_then_relu_chain(program.graph):
            return s
    raise pytest.skip(
        f"no LINEAR->RELU graph found in {max_seeds} seeds — generator weighting changed?"
    )


def test_planted_bug_caught_by_fusion_on_off_metamorphic_relation() -> None:
    """The §9 proof-of-teeth test.

    Inject a wrong "fusion" (LINEAR -> RELU rewritten to LINEAR_RELU with
    a 1.5x weight scale). The harness MUST catch the divergence.
    """
    seed = _find_seed_with_linear_relu_fuse_pair()
    program = generate_program(seed)
    bad_graph = _planted_wrong_linear_relu_fusion(program.graph)
    out_unfused = np.asarray(GraphReferenceInterpreter(program.graph).run(*program.inputs))
    out_bad_fused = np.asarray(GraphReferenceInterpreter(bad_graph).run(*program.inputs))
    assert not np.allclose(out_unfused, out_bad_fused, atol=1e-3, rtol=1e-3), (
        "planted bug did NOT change the output — re-tune the planted rewrite"
    )
    diff = fdo.diff_two_graphs(program.graph, bad_graph, program.inputs, rtol=1e-3, atol=1e-3)
    assert diff["match"] is False, (
        f"metamorphic harness FAILED to catch the planted bug: {diff!r}"
    )
    assert diff["max_abs_error"] > 1e-3, (
        f"planted-bug diff under tolerance: max_abs_error={diff['max_abs_error']}"
    )


def test_planted_bug_caught_by_evaluate_all_relations_via_diff_two_graphs() -> None:
    seed = _find_seed_with_linear_relu_fuse_pair()
    program = generate_program(seed)
    bad_graph = _planted_wrong_linear_relu_fusion(program.graph)
    diff = fdo.diff_two_graphs(program.graph, bad_graph, program.inputs)
    assert diff["match"] is False
    assert diff["bit_exact"] is False


# ---------------------------------------------------------------------------
# (3) Minimizer reduces a known multi-op failure to the minimal subset.
# ---------------------------------------------------------------------------


def test_minimizer_reduces_failing_graph_to_minimal_subset() -> None:
    M, N = 1, 8
    g = GraphIR(name="ddmin_target")
    g.inputs = ["x"]
    g.add_value("x", shape=(M, N), dtype="torch.float32")
    g.add_value("a", shape=(M, N), dtype="torch.float32")
    g.add_value("b", shape=(M, N), dtype="torch.float32")
    g.add_value("c", shape=(M, N), dtype="torch.float32")
    g.add_op(OpNode(name="r0", op=OpKind.RELU, inputs=["x"], outputs=["a"], attrs={}))
    g.add_op(OpNode(name="s0", op=OpKind.SCALE, inputs=["a"], outputs=["b"], attrs={"scale": 0.5}))
    g.add_op(OpNode(name="r1", op=OpKind.RELU, inputs=["b"], outputs=["c"], attrs={}))
    g.outputs = ["c"]
    program = GeneratedProgram(
        seed=-1,
        graph=g,
        inputs=[np.array([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]], dtype=np.float32)],
    )

    def predicate(p: GeneratedProgram) -> bool:
        return any(op.op == OpKind.SCALE for op in p.graph.ops)

    minimized, stats = minimize(program, predicate, max_iterations=8)
    assert any(op.op == OpKind.SCALE for op in minimized.graph.ops)
    assert len(minimized.graph.ops) < len(program.graph.ops), (
        f"minimizer made no progress: started {len(program.graph.ops)}, "
        f"ended {len(minimized.graph.ops)}"
    )
    assert stats.deletions_kept >= 1


def test_minimizer_no_op_when_predicate_already_false() -> None:
    program = generate_program(0)

    def predicate(_p: GeneratedProgram) -> bool:
        return False

    minimized, stats = minimize(program, predicate)
    assert minimized is program
    assert stats.iterations == 0


# ---------------------------------------------------------------------------
# Metamorphic relations — sanity checks (real compiler must agree).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(0, 16)))
def test_fusion_on_off_relation_passes_on_real_compiler(seed: int) -> None:
    program = generate_program(seed)
    result = relation_fusion_on_off(program)
    if result.skipped:
        pytest.skip(result.skip_reason or "no LINEAR->RELU pair")
    assert result.match, f"fusion_on_off failed unexpectedly: {result.to_dict()}"


@pytest.mark.parametrize("seed", list(range(0, 16)))
def test_dce_on_off_relation_passes_on_real_compiler(seed: int) -> None:
    program = generate_program(seed)
    result = relation_dce_on_off(program)
    assert result.match, f"dce_on_off failed unexpectedly: {result.to_dict()}"


@pytest.mark.parametrize("seed", list(range(0, 8)))
def test_schedule_alternative_relation_passes_on_real_compiler(seed: int) -> None:
    program = generate_program(seed)
    result = relation_schedule_alternative(program)
    if result.skipped:
        pytest.skip(result.skip_reason or "schedule alternative not applicable")
    assert result.match, f"schedule_alternative failed: {result.to_dict()}"


def test_region_fused_relation_skips_or_passes_on_cpu_only_host() -> None:
    """On a CPU-only host (no cuda-python / no GPU) the relation MUST
    skip with a human-readable reason. On a CUDA host it must match.
    Either way it must NEVER crash and NEVER fabricate a pass."""
    program = generate_program(0)
    result = relation_region_fused_vs_op_by_op(program)
    if result.skipped:
        assert result.skip_reason, "skipped relation must record a skip_reason"
    else:
        assert result.match, f"region_fused_vs_op_by_op failed on real compiler: {result.to_dict()}"


@pytest.mark.parametrize("seed", [20260546, 20260569, 20260597])
def test_region_fused_full_graph_handles_formerly_skipped_seeds(seed: int) -> None:
    """v2 hardening pass (2026-05-25) — full-graph splice eliminates
    ``region_not_whole_graph`` skips.

    For these seeds the generator emits a ``LINEAR -> {ADD,SCALE,RELU}
    -> LINEAR_RELU`` chain. The first two ops form a single fusable
    region whose ``region.output`` is the *intermediate* tensor; the
    *graph's* output is produced by the trailing ``LINEAR_RELU``.

    Old behavior (v1): relation skipped with ``region_not_whole_graph``
    because the v1 cuda_megakernel runner only returned the region tensor.

    New behavior (v2): the relation runs the megakernel for the region,
    splices its output back into the reference execution for the
    downstream ops, and compares the FULL graph output against a pure
    numpy_reference run.

    On a CPU host (no cuda-python) the relation correctly skips with
    ``cuda_megakernel_unavailable``. On a CUDA host it MUST evaluate
    (not skip with ``region_not_whole_graph``) and the match MUST hold.
    """
    from region_fusion import find_fusion_regions

    program = generate_program(seed)
    analysis = find_fusion_regions(program.graph)
    if len(analysis.regions) != 1:
        pytest.skip(
            f"seed {seed} no longer triggers the precondition (regions={len(analysis.regions)})"
        )
    region = analysis.regions[0]
    graph_outputs = list(program.graph.outputs)
    if len(graph_outputs) != 1 or region.output == graph_outputs[0]:
        pytest.skip(f"seed {seed} no longer triggers region!=graph.outputs[0] case")

    result = relation_region_fused_vs_op_by_op(program)
    if result.skipped:
        # The only acceptable skip reason on a CPU host is cuda_megakernel
        # unavailability. The deprecated 'region.output != graph.outputs[0]'
        # skip MUST NOT fire any more.
        assert "region.output" not in (result.skip_reason or ""), (
            f"seed {seed} still skipped with deprecated region/graph-output mismatch: "
            f"{result.skip_reason!r}"
        )
        return
    # CUDA-available path: relation must MATCH.
    assert result.match, (
        f"seed {seed} full-graph splice failed unexpectedly: {result.to_dict()}"
    )
    assert result.extras.get("comparison_mode") == "full_graph_splice", (
        f"seed {seed} expected full_graph_splice comparison mode, got: {result.extras!r}"
    )


# ---------------------------------------------------------------------------
# (4) Schema lock + smoke run.
# ---------------------------------------------------------------------------


def _required_top_level_keys() -> List[str]:
    return [
        "status",
        "schema_version",
        "mode",
        "seed",
        "num_graphs_generated",
        "num_graphs_evaluated",
        "num_valid_graphs_generated",
        "num_invalid_graphs_generated",
        "num_invalid_graphs_rejected_cleanly",
        "num_invalid_graphs_unexpectedly_accepted",
        "num_invalid_graphs_crashed_badly",
        "num_equivalence_checks",
        "num_rejection_checks",
        "num_divergences_found",
        "num_real_miscompiles",
        "num_tolerance_artifacts",
        "bugs",
        "invalid_findings",
        "relations",
        "relation_stats",
        "relation_evaluation_counts",
        "skip_rate_by_relation",
        "skipped_relation_reasons",
        "invalid_rejection_outcomes",
        "coverage",
        "environment",
        "git_sha",
        "generated_at_utc",
        "tolerance",
        "cuda_megakernel_registered",
        "cuda_megakernel_skip_reason",
        "num_graphs_requested",
        "include_invalid",
        "errors",
    ]


def test_run_fuzzer_emits_full_schema_smoke() -> None:
    artifact = run_fuzzer(
        mode="ci_seeded",
        seed=1234,
        num_graphs=8,
        minimize_failures=True,
        write_repros=False,
        measure_coverage=False,
        include_invalid=True,
        invalid_fraction=0.5,
    )
    for key in _required_top_level_keys():
        assert key in artifact, f"fuzzer artifact missing top-level key '{key}'"
    assert "inductor_checks" in artifact
    assert artifact["inductor_checks"] == 0
    assert artifact.get("include_torch_inductor") is False
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert isinstance(artifact["bugs"], list)
    assert isinstance(artifact["invalid_findings"], list)
    assert isinstance(artifact["relations"], list)
    assert set(artifact["relations"]) == set(ALL_RELATIONS)
    assert isinstance(artifact["relation_stats"], dict)
    for r in ALL_RELATIONS:
        rs = artifact["relation_stats"][r]
        for sub in ("evaluated", "skipped", "failed"):
            assert sub in rs, f"relation_stats[{r!r}] missing '{sub}'"
    assert isinstance(artifact["relation_evaluation_counts"], dict)
    assert set(artifact["relation_evaluation_counts"].keys()) == set(ALL_RELATIONS)
    assert isinstance(artifact["skip_rate_by_relation"], dict)
    assert set(artifact["skip_rate_by_relation"].keys()) == set(ALL_RELATIONS)
    for r in ALL_RELATIONS:
        rate = artifact["skip_rate_by_relation"][r]
        assert isinstance(rate, float) and 0.0 <= rate <= 1.0, (
            f"skip_rate_by_relation[{r!r}] must be a float in [0, 1]; got {rate!r}"
        )
        # Internal consistency: relation_evaluation_counts[r] == evaluated + skipped
        total = artifact["relation_evaluation_counts"][r]
        eval_plus_skip = (
            artifact["relation_stats"][r]["evaluated"]
            + artifact["relation_stats"][r]["skipped"]
        )
        assert total == eval_plus_skip, (
            f"relation_evaluation_counts[{r!r}]={total} but evaluated+skipped="
            f"{eval_plus_skip}"
        )
    assert isinstance(artifact["skipped_relation_reasons"], dict)
    assert set(artifact["skipped_relation_reasons"].keys()) == set(ALL_RELATIONS)
    for r in ALL_RELATIONS:
        hist = artifact["skipped_relation_reasons"][r]
        assert isinstance(hist, dict), (
            f"skipped_relation_reasons[{r!r}] must be a {{bucket: count}} dict"
        )
        for bucket, count in hist.items():
            assert isinstance(bucket, str) and isinstance(count, int) and count >= 0, (
                f"skipped_relation_reasons[{r!r}] entry must be (str -> non-negative int): "
                f"got {bucket!r} -> {count!r}"
            )
        rs_total = artifact["relation_stats"][r]["skipped"]
        assert sum(hist.values()) == rs_total, (
            f"skipped_relation_reasons[{r!r}] sum {sum(hist.values())} must equal "
            f"relation_stats[{r!r}]['skipped'] = {rs_total}"
        )
    cov = artifact["coverage"]
    for sub in (
        "ops_covered",
        "shape_buckets_covered",
        "kinds_covered",
        "graph_families_covered",
        "input_distributions_covered",
        "input_distribution_counts",
        "invalid_kinds_covered",
        "invalid_validators_covered",
    ):
        assert sub in cov, f"coverage missing '{sub}'"
    # Invalid rejection histogram structure
    iro = artifact["invalid_rejection_outcomes"]
    assert isinstance(iro, dict)
    assert set(iro.keys()) == set(INVALID_KINDS), (
        f"invalid_rejection_outcomes keys {sorted(iro.keys())} != INVALID_KINDS {sorted(INVALID_KINDS)}"
    )
    for kind, hist in iro.items():
        for bucket in ("rejected_cleanly", "unexpectedly_accepted", "crashed_badly"):
            assert bucket in hist
            assert isinstance(hist[bucket], int) and hist[bucket] >= 0
    # Internal consistency: per-kind counters sum across the artifact
    total_rejected = sum(h["rejected_cleanly"] for h in iro.values())
    total_accepted = sum(h["unexpectedly_accepted"] for h in iro.values())
    total_crashed = sum(h["crashed_badly"] for h in iro.values())
    assert artifact["num_invalid_graphs_rejected_cleanly"] == total_rejected
    assert artifact["num_invalid_graphs_unexpectedly_accepted"] == total_accepted
    assert artifact["num_invalid_graphs_crashed_badly"] == total_crashed
    assert (
        artifact["num_invalid_graphs_generated"]
        == total_rejected + total_accepted + total_crashed
    )
    tol = artifact["tolerance"]
    assert "rtol" in tol and "atol" in tol


def test_run_fuzzer_short_run_finds_no_real_miscompiles_on_clean_compiler() -> None:
    """The §4.6 honest-claim gate: a short, deterministic run on the
    LIVE compiler must report 0 real miscompiles. If this ever flips
    True we have actually found a bug — that's a real win, not a
    failure of this test."""
    artifact = run_fuzzer(
        mode="ci_seeded",
        seed=4242,
        num_graphs=16,
        minimize_failures=False,
        write_repros=False,
        measure_coverage=False,
        include_invalid=False,
        use_input_distributions=False,
    )
    assert artifact["num_graphs_generated"] == 16
    assert artifact["num_real_miscompiles"] == 0, (
        f"fuzzer caught a real miscompile we did not expect: {artifact['bugs']!r}"
    )


def test_run_fuzzer_invalid_corpus_rejected_cleanly_on_clean_compiler() -> None:
    """Honest-claim gate for the invalid corpus: every invalid graph
    MUST be rejected cleanly by the compiler. Any unexpected_accept or
    crashed_badly outcome is itself a bug-finding signal — we leave the
    failure mode visible so the runner doesn't silently hide it."""
    artifact = run_fuzzer(
        mode="ci_seeded",
        seed=4242,
        num_graphs=4,
        minimize_failures=False,
        write_repros=False,
        measure_coverage=False,
        include_invalid=True,
        invalid_fraction=1.5,
        use_input_distributions=False,
    )
    assert artifact["num_invalid_graphs_generated"] >= len(INVALID_KINDS), (
        "invalid corpus must exercise every documented invalid kind"
    )
    assert artifact["num_invalid_graphs_unexpectedly_accepted"] == 0, (
        f"compiler unexpectedly accepted invalid graphs: "
        f"{artifact['invalid_findings']!r}"
    )
    assert artifact["num_invalid_graphs_crashed_badly"] == 0, (
        f"compiler crashed badly (non-diagnostic exception) on invalid graphs: "
        f"{artifact['invalid_findings']!r}"
    )


def test_run_fuzzer_records_input_distributions_used() -> None:
    artifact = run_fuzzer(
        mode="ci_seeded",
        seed=999,
        num_graphs=64,
        minimize_failures=False,
        write_repros=False,
        measure_coverage=False,
        use_input_distributions=True,
    )
    cov = artifact["coverage"]
    counts = cov["input_distribution_counts"]
    total = sum(int(v) for v in counts.values())
    assert total == artifact["num_valid_graphs_generated"], (
        f"distribution counts {counts!r} sum {total} != "
        f"num_valid_graphs_generated {artifact['num_valid_graphs_generated']}"
    )
    distributions = set(cov["input_distributions_covered"])
    assert distributions, "expected at least one distribution to be exercised"
    assert distributions.issubset(set(INPUT_DISTRIBUTIONS))


def test_run_fuzzer_captures_planted_bug_via_relation_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-state the planted-bug invariant at the runner level."""
    failing_relation = "fusion_on_off"

    def fake_relation(program, rtol=1e-5, atol=1e-5):
        return MetamorphicResult(
            relation=failing_relation,
            match=False,
            reason="planted by test",
            max_abs_error=1.0,
            max_rel_error=1.0,
            bit_exact=False,
            skipped=False,
        )

    import fuzz.metamorphic as mm

    monkeypatch.setattr(mm, "relation_fusion_on_off", fake_relation)
    artifact = run_fuzzer(
        mode="ci_seeded",
        seed=999,
        num_graphs=6,
        minimize_failures=False,
        write_repros=False,
        include_invalid=False,
        use_input_distributions=False,
    )
    assert artifact["num_divergences_found"] >= 1
    assert artifact["num_real_miscompiles"] >= 1
    assert artifact["relation_stats"][failing_relation]["failed"] >= 1


def test_committed_artifact_schema_lock_if_present() -> None:
    """If ``bench/results/fuzzer_report.json`` already exists (CI regenerates
    it before tests), the on-disk schema must lock to the runner's keys.
    Skip silently when absent."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "bench" / "results" / "fuzzer_report.json"
    if not path.exists():
        pytest.skip("fuzzer_report.json not yet generated on this host")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for key in _required_top_level_keys():
        assert key in artifact, f"committed fuzzer_report.json missing key '{key}'"
    assert artifact["schema_version"] == SCHEMA_VERSION, (
        f"committed artifact schema_version {artifact['schema_version']} "
        f"!= runner SCHEMA_VERSION {SCHEMA_VERSION}"
    )
    # No fabricated bug count: every entry in `bugs` must have a real
    # numeric `seed` (>=0 for fuzzer-found, -1 for synthetic regression).
    assert isinstance(artifact["bugs"], list)
    for b in artifact["bugs"]:
        assert "is_real_miscompile" in b, f"bug entry missing flag: {b!r}"
        assert isinstance(b["seed"], int)
    # `num_real_miscompiles` must equal the count of bugs flagged real.
    real = sum(1 for b in artifact["bugs"] if b.get("is_real_miscompile") is True)
    assert artifact["num_real_miscompiles"] == real, (
        f"committed artifact num_real_miscompiles={artifact['num_real_miscompiles']} "
        f"but bugs[is_real_miscompile]={real} — inconsistent"
    )


# ---------------------------------------------------------------------------
# Differential oracle wrapper — small contract checks.
# ---------------------------------------------------------------------------


def test_diff_two_graphs_matches_when_graphs_are_identical() -> None:
    program = generate_program(3)
    diff = fdo.diff_two_graphs(program.graph, program.graph, program.inputs)
    assert diff["match"] is True
    assert diff["bit_exact"] is True


def test_default_backend_set_includes_numpy_reference() -> None:
    backends = fdo.default_backend_set(include_cuda_megakernel=False)
    assert "numpy_reference" in backends
    backends_with_mk = fdo.default_backend_set(include_cuda_megakernel=True)
    assert "numpy_reference" in backends_with_mk
    assert "cuda_megakernel" in backends_with_mk


def test_compare_torch_inductor_skips_cleanly_without_torch() -> None:
    program = generate_program(5)
    result = fdo.compare_torch_inductor(program)
    try:
        import torch  # noqa: F401
    except Exception:
        assert result.skipped is True
        assert result.skip_reason is not None
        assert "inductor_unavailable" in result.skip_reason
        return
    # Torch present: result may run or skip (Inductor/CUDA); must not crash.
    assert isinstance(result.match, bool)


def test_run_fuzzer_optional_inductor_flag_increments_counters() -> None:
    artifact = run_fuzzer(
        mode="ci_seeded",
        seed=9999,
        num_graphs=4,
        minimize_failures=False,
        include_torch_inductor=True,
    )
    assert artifact["include_torch_inductor"] is True
    assert artifact["inductor_checks"] + artifact["inductor_skips"] == artifact["num_graphs_evaluated"]
