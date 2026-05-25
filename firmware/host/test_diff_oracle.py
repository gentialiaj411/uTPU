"""Tests for `diff_oracle` (Task 0 / `utpu_upgrade_plan.md` §2.1).

Locks the API contract that Tasks 1 (megakernel correctness gate), 2 (fuzzer
differential / metamorphic oracle), and 3 (superoptimizer extraction
verification) will all rely on. The properties tested here are the
non-negotiable ones: never silently drop a backend, integer paths must be
bit-exact under rtol=atol=0, planted miscompiles must be caught.
"""

import math

import numpy as np
import pytest

import diff_oracle
from diff_oracle import (
    BackendResult,
    BackendUnavailable,
    PairCompare,
    available_backends,
    compare,
    register_backend,
    run_all_backends,
)
from graph_ir import GraphIR, OpKind, OpNode
from graph_reference_interpreter import GraphReferenceInterpreter


def _tiny_linear_graph() -> GraphIR:
    g = GraphIR(name="tiny_linear")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_value("x", shape=(1, 2), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="fc",
            op=OpKind.LINEAR,
            inputs=["x"],
            outputs=["y"],
            attrs={
                "weight": np.array([[1.0, 2.0], [-1.0, 1.0]], dtype=np.float32),
                "bias": np.array([0.0, 0.5], dtype=np.float32),
                "in_features": 2,
                "out_features": 2,
            },
        )
    )
    return g


def _tiny_input() -> np.ndarray:
    return np.array([[3.0, 1.0]], dtype=np.float32)


def _two_backend_outputs(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "ref": BackendResult(name="ref", status="ok", output=a, reason=None),
        "test": BackendResult(name="test", status="ok", output=b, reason=None),
    }


def test_default_backend_registry_contains_all_planned_names():
    names = available_backends()
    for required in ("numpy_reference", "cuda", "cuda_megakernel", "eager", "torch_inductor"):
        assert required in names, f"backend '{required}' missing from registry"


def test_numpy_reference_backend_returns_expected_output_on_tiny_linear():
    out = run_all_backends(_tiny_linear_graph(), [_tiny_input()], backends=("numpy_reference",))
    assert "numpy_reference" in out
    res = out["numpy_reference"]
    assert res.status == "ok"
    assert res.reason is None
    # y = x @ W.T + b = [[3*1+1*2, 3*-1+1*1+0.5]] = [[5.0, -1.5]]
    expected = np.array([[5.0, -1.5]], dtype=np.float32)
    np.testing.assert_allclose(res.output, expected)


def test_run_all_backends_never_silently_drops_a_backend():
    requested = ("numpy_reference", "cuda", "cuda_megakernel", "eager", "torch_inductor")
    out = run_all_backends(_tiny_linear_graph(), [_tiny_input()], backends=requested)
    assert set(out.keys()) == set(requested), (
        "every requested backend must appear in the result dict, even if skipped"
    )


def test_unavailable_backend_is_skipped_with_human_reason():
    out = run_all_backends(
        _tiny_linear_graph(),
        [_tiny_input()],
        backends=("cuda_megakernel",),
    )
    res = out["cuda_megakernel"]
    assert res.status == "skipped"
    assert res.output is None
    assert res.reason and "Task 1" in res.reason


def test_torch_backends_skip_when_torch_module_not_provided():
    out = run_all_backends(
        _tiny_linear_graph(),
        [_tiny_input()],
        backends=("eager", "torch_inductor"),
    )
    for name in ("eager", "torch_inductor"):
        res = out[name]
        assert res.status == "skipped", f"{name} should skip without torch_module, got {res.status}"
        assert res.reason and "torch_module" in res.reason


def test_unknown_backend_is_skipped_with_no_runner_message():
    out = run_all_backends(_tiny_linear_graph(), [_tiny_input()], backends=("does_not_exist",))
    res = out["does_not_exist"]
    assert res.status == "skipped"
    assert "no runner" in (res.reason or "")


def test_register_backend_plugs_in_a_new_runner_and_runs_it():
    sentinel = np.array([[123.0, 456.0]], dtype=np.float32)

    def fake_runner(graph, inputs, **_):
        return sentinel

    try:
        register_backend("__test_fake_backend__", fake_runner)
        out = run_all_backends(
            _tiny_linear_graph(),
            [_tiny_input()],
            backends=("__test_fake_backend__",),
        )
        res = out["__test_fake_backend__"]
        assert res.status == "ok"
        np.testing.assert_array_equal(res.output, sentinel)
    finally:
        diff_oracle._BACKEND_RUNNERS.pop("__test_fake_backend__", None)


def test_runner_exception_becomes_error_status_not_silent_drop():
    def explodes(graph, inputs, **_):
        raise RuntimeError("planned failure")

    try:
        register_backend("__test_exploder__", explodes)
        out = run_all_backends(
            _tiny_linear_graph(),
            [_tiny_input()],
            backends=("__test_exploder__",),
        )
        res = out["__test_exploder__"]
        assert res.status == "error"
        assert "planned failure" in (res.reason or "")
        assert res.output is None
    finally:
        diff_oracle._BACKEND_RUNNERS.pop("__test_exploder__", None)


def test_register_backend_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        register_backend("", lambda *a, **k: np.array([0]))
    with pytest.raises(TypeError):
        register_backend("not_callable", "not a function")  # type: ignore[arg-type]


def test_run_all_backends_validates_graph_type():
    with pytest.raises(TypeError):
        run_all_backends("not a graph", [np.array([1.0])])  # type: ignore[arg-type]


def test_run_all_backends_deduplicates_requested_backends():
    out = run_all_backends(
        _tiny_linear_graph(),
        [_tiny_input()],
        backends=("numpy_reference", "numpy_reference"),
    )
    assert list(out.keys()) == ["numpy_reference"]


def test_backend_unavailable_propagates_as_skipped():
    def picky(graph, inputs, **_):
        raise BackendUnavailable("not on this host")

    try:
        register_backend("__test_picky__", picky)
        out = run_all_backends(_tiny_linear_graph(), [_tiny_input()], backends=("__test_picky__",))
        res = out["__test_picky__"]
        assert res.status == "skipped"
        assert res.reason == "not on this host"
    finally:
        diff_oracle._BACKEND_RUNNERS.pop("__test_picky__", None)


def test_compare_bit_exact_for_identical_outputs():
    a = np.array([1, 2, 3], dtype=np.int32)
    out = _two_backend_outputs(a, a.copy())
    res = compare(out, rtol=0.0, atol=0.0)
    assert res.all_bit_exact is True
    assert res.all_within_tolerance is True
    assert len(res.pairs) == 1
    pair = res.pairs[0]
    assert pair.bit_exact is True
    assert pair.max_abs_error == 0.0
    assert pair.shape_match is True


def test_compare_integer_path_rejects_off_by_one_with_zero_tolerance():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([1, 2, 4], dtype=np.int32)
    out = _two_backend_outputs(a, b)
    res = compare(out, rtol=0.0, atol=0.0)
    assert res.all_bit_exact is False
    assert res.all_within_tolerance is False
    pair = res.pairs[0]
    assert pair.bit_exact is False
    assert pair.within_tolerance is False
    assert pair.max_abs_error == 1.0


def test_compare_float_within_tolerance_when_below_rtol_atol():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = a + 1e-4
    out = _two_backend_outputs(a, b)
    res = compare(out, rtol=1e-3, atol=1e-3)
    assert res.all_within_tolerance is True
    # bit_exact is independent of tolerance: small float perturbation is NOT bit-exact
    assert res.all_bit_exact is False
    pair = res.pairs[0]
    assert pair.max_abs_error == pytest.approx(1e-4, rel=1e-3)


def test_compare_float_caught_outside_tolerance():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = a + 1.0
    out = _two_backend_outputs(a, b)
    res = compare(out, rtol=1e-3, atol=1e-3)
    assert res.all_within_tolerance is False


def test_compare_shape_mismatch_is_caught_not_crashed():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float32)
    out = _two_backend_outputs(a, b)
    res = compare(out, rtol=1e-3, atol=1e-3)
    assert res.all_within_tolerance is False
    pair = res.pairs[0]
    assert pair.shape_match is False
    assert math.isinf(pair.max_abs_error)


def test_compare_skipped_backends_listed_and_not_compared():
    out = {
        "ref": BackendResult(name="ref", status="ok", output=np.array([1.0]), reason=None),
        "missing": BackendResult(name="missing", status="skipped", output=None, reason="no cuda"),
        "failed": BackendResult(name="failed", status="error", output=None, reason="boom"),
    }
    res = compare(out, rtol=1e-3, atol=1e-3)
    assert res.backends_compared == ["ref"]
    assert set(res.backends_skipped) == {"missing", "failed"}
    assert res.pairs == []
    # Vacuous truth when nothing to compare. Callers that want a "missing
    # oracle" failure mode must additionally check len(backends_compared)>=2.
    assert res.all_within_tolerance is True
    assert res.all_bit_exact is True


def test_compare_rejects_negative_tolerance():
    out = _two_backend_outputs(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        compare(out, rtol=-1e-3, atol=0.0)
    with pytest.raises(ValueError):
        compare(out, rtol=0.0, atol=-1e-3)


def test_compare_empty_arrays_compare_clean():
    a = np.zeros((0,), dtype=np.float32)
    out = _two_backend_outputs(a, a.copy())
    res = compare(out, rtol=1e-3, atol=1e-3)
    assert res.all_within_tolerance is True
    assert res.all_bit_exact is True


def test_pair_compare_n_choose_2_for_three_ok_backends():
    a = np.array([1.0, 2.0], dtype=np.float32)
    outputs = {
        "a": BackendResult(name="a", status="ok", output=a, reason=None),
        "b": BackendResult(name="b", status="ok", output=a.copy(), reason=None),
        "c": BackendResult(name="c", status="ok", output=a + 1e-5, reason=None),
    }
    res = compare(outputs, rtol=1e-3, atol=1e-3)
    assert len(res.pairs) == 3
    pair_names = {(p.backend_a, p.backend_b) for p in res.pairs}
    assert pair_names == {("a", "b"), ("a", "c"), ("b", "c")}


def test_planted_bug_caught_by_register_backend_plus_compare():
    """Planted-bug check required by `utpu_upgrade_plan.md` §9.

    A deliberately wrong backend (off-by-one integer output) MUST be caught
    by the full `run_all_backends + compare(rtol=0, atol=0)` pipeline. This
    is the "the harness has teeth" gate inherited from §9 — without it the
    Task 2 fuzzer would have no proof its differential layer can fire.
    """

    def wrong_backend(graph, inputs, **_):
        out = GraphReferenceInterpreter(graph).run(*inputs)
        if isinstance(out, tuple):
            out = out[0]
        return np.asarray(out).astype(np.int32) + 1

    try:
        register_backend("__planted_wrong__", wrong_backend)
        out = run_all_backends(
            _tiny_linear_graph(),
            [_tiny_input()],
            backends=("numpy_reference", "__planted_wrong__"),
        )
        assert out["numpy_reference"].status == "ok"
        assert out["__planted_wrong__"].status == "ok"
        cmp_int_strict = compare(out, rtol=0.0, atol=0.0)
        assert cmp_int_strict.all_bit_exact is False
        assert cmp_int_strict.all_within_tolerance is False
        # And the divergence is visible in the pair record (not buried).
        assert any(not p.bit_exact for p in cmp_int_strict.pairs)
    finally:
        diff_oracle._BACKEND_RUNNERS.pop("__planted_wrong__", None)


def test_compare_result_to_dict_schema_lock():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _two_backend_outputs(a, a.copy())
    d = compare(out, rtol=1e-3, atol=1e-3).to_dict()
    for key in (
        "rtol",
        "atol",
        "all_within_tolerance",
        "all_bit_exact",
        "backends_compared",
        "backends_skipped",
        "pairs",
    ):
        assert key in d, f"compare().to_dict() missing key '{key}'"
    assert isinstance(d["pairs"], list)
    pair = d["pairs"][0]
    for key in (
        "backend_a",
        "backend_b",
        "max_abs_error",
        "max_rel_error",
        "within_tolerance",
        "bit_exact",
        "shape_match",
    ):
        assert key in pair, f"pair dict missing key '{key}'"


def test_backend_result_to_dict_schema_lock():
    res = BackendResult(name="x", status="ok", output=np.array([1.0, 2.0]), reason=None)
    d = res.to_dict()
    for key in ("name", "status", "output_shape", "output_dtype", "reason"):
        assert key in d, f"BackendResult.to_dict() missing key '{key}'"
    assert d["output_shape"] == [2]
    skipped = BackendResult(name="y", status="skipped", output=None, reason="no cuda")
    d2 = skipped.to_dict()
    assert d2["output_shape"] is None
    assert d2["output_dtype"] is None
    assert d2["reason"] == "no cuda"
