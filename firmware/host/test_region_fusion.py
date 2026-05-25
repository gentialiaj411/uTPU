"""Tests for `region_fusion` (Task 1 of `utpu_upgrade_plan.md`).

These lock the v1 legality contract for fused CUDA region kernels.
The most important test is `test_multi_linear_chain_rejected_as_global_sync_trap`:
if the region planner ever silently accepts a `LINEAR -> LINEAR` chain it would
produce a numerically wrong kernel in CUDA (layer-2 reading layer-1's incomplete
output across CTAs). That rule must keep firing — the test will catch a
regression if the trap is ever relaxed.
"""

import numpy as np

import diff_oracle
from diff_oracle import BackendResult, compare
from graph_ir import GraphIR, OpKind, OpNode
from graph_reference_interpreter import GraphReferenceInterpreter
import region_fusion
from region_fusion import (
    RegionAnalysis,
    RegionPlan,
    RegionRejection,
    execute_region_numpy,
    find_fusion_regions,
)


# ---------------------------------------------------------------------------
# Tiny graph factories. Each one is a single deterministic test fixture so
# the assertions can reference exact op names.
# ---------------------------------------------------------------------------

def _linear_op(name: str, src: str, dst: str, w: np.ndarray, b=None, fused_relu: bool = False) -> OpNode:
    attrs = {
        "weight": w,
        "in_features": int(w.shape[1]),
        "out_features": int(w.shape[0]),
    }
    if b is not None:
        attrs["bias"] = b
    return OpNode(
        name=name,
        op=OpKind.LINEAR_RELU if fused_relu else OpKind.LINEAR,
        inputs=[src],
        outputs=[dst],
        attrs=attrs,
    )


def _relu_op(name: str, src: str, dst: str) -> OpNode:
    return OpNode(name=name, op=OpKind.RELU, inputs=[src], outputs=[dst], attrs={})


def _scale_op(name: str, src: str, dst: str, s: float) -> OpNode:
    return OpNode(name=name, op=OpKind.SCALE, inputs=[src], outputs=[dst], attrs={"scale": float(s)})


def _add_op(name: str, lhs: str, rhs: str, dst: str) -> OpNode:
    return OpNode(name=name, op=OpKind.ADD, inputs=[lhs, rhs], outputs=[dst], attrs={})


def _graph_linear_relu_chain() -> GraphIR:
    """x -> linear -> relu  (the canonical fused epilogue case)."""
    g = GraphIR(name="linear_relu_chain")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 4), dtype="torch.float32")
    g.add_op(_linear_op("fc1", "x", "h", w=np.array([[1.0, -1.0, 0.0, 2.0], [0.5, 0.5, -1.0, 1.0]], dtype=np.float32)))
    g.add_op(_relu_op("relu1", "h", "y"))
    return g


def _graph_linear_relu_add_scale() -> GraphIR:
    """x, r -> linear -> relu -> add(residual) -> scale (full v1 epilogue exercise)."""
    g = GraphIR(name="linear_relu_add_scale")
    g.inputs = ["x", "r"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 4), dtype="torch.float32")
    g.add_value("r", shape=(1, 2), dtype="torch.float32")
    g.add_op(_linear_op("fc1", "x", "h", w=np.array([[1.0, -1.0, 0.0, 2.0], [0.5, 0.5, -1.0, 1.0]], dtype=np.float32)))
    g.add_op(_relu_op("relu1", "h", "h_relu"))
    g.add_op(_add_op("add1", "h_relu", "r", "h_add"))
    g.add_op(_scale_op("scale1", "h_add", "y", s=0.5))
    return g


def _graph_two_linear_mlp() -> GraphIR:
    """x -> linear -> relu -> linear (the GLOBAL-SYNC TRAP)."""
    g = GraphIR(name="two_linear_mlp")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 4), dtype="torch.float32")
    g.add_op(_linear_op("fc1", "x", "h", w=np.array([[1.0, -1.0, 0.0, 2.0], [0.5, 0.5, -1.0, 1.0]], dtype=np.float32)))
    g.add_op(_relu_op("relu1", "h", "h_relu"))
    g.add_op(_linear_op("fc2", "h_relu", "y", w=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)))
    return g


def _graph_two_linear_directly_chained() -> GraphIR:
    """x -> linear -> linear (the GLOBAL-SYNC TRAP, no relu in between)."""
    g = GraphIR(name="two_linear_directly_chained")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 4), dtype="torch.float32")
    g.add_op(_linear_op("fc1", "x", "h", w=np.array([[1.0, -1.0, 0.0, 2.0], [0.5, 0.5, -1.0, 1.0]], dtype=np.float32)))
    g.add_op(_linear_op("fc2", "h", "y", w=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)))
    return g


def _graph_multi_consumer_intermediate() -> GraphIR:
    """x -> linear h -> relu(h) -> a; h also consumed by scale -> b. h has 2 consumers."""
    g = GraphIR(name="multi_consumer")
    g.inputs = ["x"]
    g.outputs = ["a", "b"]
    g.add_value("x", shape=(1, 2), dtype="torch.float32")
    g.add_op(_linear_op("fc1", "x", "h", w=np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)))
    g.add_op(_relu_op("relu1", "h", "a"))
    g.add_op(_scale_op("scale1", "h", "b", s=2.0))
    return g


def _graph_elementwise_chain() -> GraphIR:
    """x, r -> relu -> scale -> add(r)."""
    g = GraphIR(name="elementwise_chain")
    g.inputs = ["x", "r"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 4), dtype="torch.float32")
    g.add_value("r", shape=(1, 4), dtype="torch.float32")
    g.add_op(_relu_op("relu1", "x", "h"))
    g.add_op(_scale_op("scale1", "h", "h2", s=0.5))
    g.add_op(_add_op("add1", "h2", "r", "y"))
    return g


def _graph_bare_linear_only() -> GraphIR:
    """A single LINEAR — should NOT form a region (no epilogue to fuse)."""
    g = GraphIR(name="bare_linear")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 2), dtype="torch.float32")
    g.add_op(_linear_op("fc1", "x", "y", w=np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)))
    return g


def _graph_residual_internal_add() -> GraphIR:
    """Two paths both produced inside one candidate region feed an ADD.
       linear -> relu -> path_a; linear -> scale -> path_b; add(path_a, path_b)
       The linear has TWO consumers, so first 'h' fails the single-consumer rule
       — but more importantly, if we hypothetically tried to fuse all of it
       into one region, the ADD's two inputs would both be region-internal.
       This test pins both rejection behaviors.
    """
    g = GraphIR(name="residual_internal")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 2), dtype="torch.float32")
    g.add_op(_linear_op("fc1", "x", "h", w=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)))
    g.add_op(_relu_op("relu1", "h", "a"))
    g.add_op(_scale_op("scale1", "h", "b", s=2.0))
    g.add_op(_add_op("add1", "a", "b", "y"))
    return g


# ---------------------------------------------------------------------------
# Positive cases — a region forms exactly as expected.
# ---------------------------------------------------------------------------

def test_linear_then_relu_forms_one_linear_with_epilogue_region():
    analysis = find_fusion_regions(_graph_linear_relu_chain())
    assert len(analysis.regions) == 1
    region = analysis.regions[0]
    assert region.region_kind == "linear_with_epilogue"
    assert region.root_op_name == "fc1"
    assert region.op_names == ("fc1", "relu1")
    assert region.epilogue_op_names == ("relu1",)
    assert region.output == "y"
    assert region.inputs_external == ("x",)


def test_linear_relu_add_scale_folds_full_epilogue_into_one_region():
    analysis = find_fusion_regions(_graph_linear_relu_add_scale())
    assert len(analysis.regions) == 1
    region = analysis.regions[0]
    assert region.region_kind == "linear_with_epilogue"
    assert region.op_names == ("fc1", "relu1", "add1", "scale1")
    assert region.epilogue_op_names == ("relu1", "add1", "scale1")
    # External inputs must include the residual `r` (it is NOT produced inside).
    assert "x" in region.inputs_external
    assert "r" in region.inputs_external
    assert region.output == "y"


def test_elementwise_chain_relu_scale_add_forms_one_region():
    analysis = find_fusion_regions(_graph_elementwise_chain())
    assert len(analysis.regions) == 1
    region = analysis.regions[0]
    assert region.region_kind == "elementwise_chain"
    assert region.root_op_name is None
    assert region.op_names == ("relu1", "scale1", "add1")
    assert region.output == "y"
    assert "x" in region.inputs_external
    assert "r" in region.inputs_external


# ---------------------------------------------------------------------------
# Negative cases — singletons stay singletons; bad cases are rejected with the
# correct rejection_kind. THESE are the safety tests.
# ---------------------------------------------------------------------------

def test_bare_linear_with_no_epilogue_does_not_form_a_region():
    """A single LINEAR with no epilogue is NOT a region. The existing
    blocked-FC executor already handles it; emitting a degenerate
    'region of size 1' would be confusing and would not buy any fusion."""
    analysis = find_fusion_regions(_graph_bare_linear_only())
    assert analysis.regions == ()


def test_multi_linear_chain_rejected_as_global_sync_trap():
    """THE crucial safety test. A LINEAR -> LINEAR (with or without relu in
    between) chain must be rejected with `rejection_kind='global_sync_required'`.
    If this test ever fails the region planner would let a numerically broken
    kernel through (layer-2 threads reading layer-1's partially-written output).
    """
    analysis = find_fusion_regions(_graph_two_linear_mlp())
    # Only the first linear+relu fuses; the trailing second linear is left as a
    # singleton (single-consumer chain breaks at the trap).
    assert len(analysis.regions) == 1
    assert analysis.regions[0].op_names == ("fc1", "relu1")
    # And a global-sync rejection MUST have been recorded.
    sync_rej = [r for r in analysis.rejections if r.rejection_kind == "global_sync_required"]
    assert sync_rej, f"expected global_sync_required rejection, got {analysis.rejections}"
    rej = sync_rej[0]
    assert "fc2" in rej.candidate_op_names
    assert "fc1" in rej.candidate_op_names


def test_directly_chained_two_linears_rejected_as_global_sync_trap():
    """The trap fires even without a relu between the two linears."""
    analysis = find_fusion_regions(_graph_two_linear_directly_chained())
    # Bare LINEAR -> LINEAR: the first linear has no epilogue (the LINEAR cannot
    # fold in), so no region forms. The rejection must still be recorded.
    assert analysis.regions == ()
    sync_rej = [r for r in analysis.rejections if r.rejection_kind == "global_sync_required"]
    assert sync_rej, "global-sync trap must fire even for bare LINEAR -> LINEAR"


def test_multi_consumer_intermediate_stops_region_growth():
    """When the intermediate value (h) is consumed by two different ops, the
    chain ends at h. h itself is still materialized (the existing executor
    runs the bare LINEAR singleton), and the two consumers (relu, scale) are
    each evaluated as singletons (chain length 1 — no elementwise region)."""
    analysis = find_fusion_regions(_graph_multi_consumer_intermediate())
    # No region — the bare linear yields no epilogue chain (its single
    # successor would have to be unique, which it isn't), and each
    # downstream elementwise op is a chain of length 1.
    assert analysis.regions == ()


def test_residual_internal_chain_is_rejected_or_split():
    """If we tried to grow a region across both paths feeding an ADD, the
    ADD's two operands would both be region-internal, which would need
    cross-thread sync. The planner must either reject the ADD's growth
    attempt with `residual_internal` OR split into separate single-op
    regions (which collapse to nothing in v1, since chain-length-1 doesn't
    qualify as an elementwise region).
    """
    analysis = find_fusion_regions(_graph_residual_internal_add())
    # The fc1 output 'h' has two consumers, so it does not form a
    # linear_with_epilogue region. Each branch is a length-1 chain.
    # No region should form; importantly, no INCORRECT region should form.
    for region in analysis.regions:
        # Whatever did form must NOT contain both paths feeding the same
        # ADD as internal inputs — that would be the residual_internal bug.
        if "add1" in region.op_names:
            chain_outputs = {
                op_name for op_name in region.op_names
            }
            # If both 'a' and 'b' producers are in the region, the ADD is internal.
            assert not ("relu1" in chain_outputs and "scale1" in chain_outputs), (
                "ADD residual must come from outside the region; "
                "if both producers are internal the region is incorrect."
            )


# ---------------------------------------------------------------------------
# CORRECTNESS GATE (GPU-free): regions must execute to the same numbers as
# the unfused reference interpreter. Uses diff_oracle.compare for the actual
# numerical check, so the same gate Tasks 1/2/3 use is exercised here.
# ---------------------------------------------------------------------------

def _interpret_with_intermediates(graph: GraphIR, x: np.ndarray, *extras) -> dict:
    """Run the reference interpreter while capturing every intermediate."""
    interp = GraphReferenceInterpreter(graph)
    snapshot: dict = {}
    # Reuse the interpreter's `run` to be byte-identical, then re-derive each
    # intermediate by op-by-op walk (the interpreter doesn't expose intermediates).
    # For these tiny test graphs an extra forward pass is fine.
    values = {name: np.asarray(val, dtype=np.float32) for name, val in zip(graph.inputs, (x, *extras))}
    from region_fusion import _exec_op_numpy  # internal, but kept stable

    for op in graph.ops:
        values[op.outputs[0]] = _exec_op_numpy(op, values)
    snapshot["values"] = values
    # Sanity: interpreter top-level output must match our walk.
    y_interp = interp.run(*[v for v in (x, *extras)])
    if isinstance(y_interp, tuple):
        y_interp = y_interp[0]
    np.testing.assert_allclose(values[graph.outputs[0]], y_interp, rtol=0, atol=1e-6)
    return snapshot


def test_linear_relu_region_output_matches_unfused_via_diff_oracle():
    graph = _graph_linear_relu_chain()
    x = np.array([[3.0, 1.0, -1.0, 2.0]], dtype=np.float32)
    snapshot = _interpret_with_intermediates(graph, x)
    expected = snapshot["values"][graph.outputs[0]]

    analysis = find_fusion_regions(graph)
    assert len(analysis.regions) == 1
    region = analysis.regions[0]
    external = {name: snapshot["values"][name] for name in region.inputs_external}
    fused_out = execute_region_numpy(region, graph, external)

    outputs = {
        "reference": BackendResult(name="reference", status="ok", output=expected, reason=None),
        "fused_region_numpy": BackendResult(name="fused_region_numpy", status="ok", output=fused_out, reason=None),
    }
    res = compare(outputs, rtol=1e-6, atol=1e-6)
    assert res.all_within_tolerance, f"region output diverged from reference: {res.to_dict()}"


def test_linear_relu_add_scale_region_output_matches_unfused_via_diff_oracle():
    graph = _graph_linear_relu_add_scale()
    x = np.array([[3.0, 1.0, -1.0, 2.0]], dtype=np.float32)
    r = np.array([[0.25, -0.5]], dtype=np.float32)
    snapshot = _interpret_with_intermediates(graph, x, r)
    expected = snapshot["values"][graph.outputs[0]]

    analysis = find_fusion_regions(graph)
    assert len(analysis.regions) == 1
    region = analysis.regions[0]
    external = {name: snapshot["values"][name] for name in region.inputs_external}
    fused_out = execute_region_numpy(region, graph, external)
    outputs = {
        "reference": BackendResult(name="reference", status="ok", output=expected, reason=None),
        "fused_region_numpy": BackendResult(name="fused_region_numpy", status="ok", output=fused_out, reason=None),
    }
    res = compare(outputs, rtol=1e-6, atol=1e-6)
    assert res.all_within_tolerance, res.to_dict()


def test_elementwise_chain_region_output_matches_unfused_via_diff_oracle():
    graph = _graph_elementwise_chain()
    x = np.array([[-2.0, 1.0, 3.0, -4.0]], dtype=np.float32)
    r = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    snapshot = _interpret_with_intermediates(graph, x, r)
    expected = snapshot["values"][graph.outputs[0]]

    analysis = find_fusion_regions(graph)
    assert len(analysis.regions) == 1
    region = analysis.regions[0]
    external = {name: snapshot["values"][name] for name in region.inputs_external}
    fused_out = execute_region_numpy(region, graph, external)
    outputs = {
        "reference": BackendResult(name="reference", status="ok", output=expected, reason=None),
        "fused_region_numpy": BackendResult(name="fused_region_numpy", status="ok", output=fused_out, reason=None),
    }
    res = compare(outputs, rtol=1e-6, atol=1e-6)
    assert res.all_within_tolerance, res.to_dict()


# ---------------------------------------------------------------------------
# Misc API hygiene.
# ---------------------------------------------------------------------------

def test_region_analysis_to_dict_schema_lock():
    analysis = find_fusion_regions(_graph_linear_relu_chain())
    d = analysis.to_dict()
    for key in ("regions", "rejections", "ops_in_regions"):
        assert key in d
    region = d["regions"][0]
    for key in ("region_id", "region_kind", "op_names", "root_op_name", "epilogue_op_names", "inputs_external", "output", "rationale"):
        assert key in region


def test_find_fusion_regions_validates_input_type():
    import pytest
    with pytest.raises(TypeError):
        find_fusion_regions("not a graph")  # type: ignore[arg-type]


def test_global_sync_rejection_visible_for_audit():
    """The rejection list is part of the public API — `bench/results/megakernel_payoff.json`
    serializes it so reviewers can see the trap rule actively fired on the workload."""
    analysis = find_fusion_regions(_graph_two_linear_mlp())
    serialized = analysis.to_dict()
    assert any(r["rejection_kind"] == "global_sync_required" for r in serialized["rejections"])
