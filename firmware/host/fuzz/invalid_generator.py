"""Invalid-graph negative fuzzer (Task 2 hardening pass, 2026-05-25).

Generates intentionally **invalid** Graph IR programs so we can verify the
compiler rejects them cleanly. The honest claim this artifact supports is:

  "Generated N invalid graphs; rejected K cleanly via explicit
   diagnostic; M unexpectedly accepted; B crashed badly."

A clean rejection means:

* `shape_inference_pass` raises one of the known shape-inference errors
  (KeyError, ValueError, IndexError, AssertionError).
* `backend_legality_pass(graph, target_backend)` raises
  `BackendLegalityError`.
* `GraphReferenceInterpreter(graph).run(*inputs)` raises
  `GraphReferenceInterpreterError` (or numpy's ValueError) when the
  invalidity only surfaces during execution.

An unexpected ACCEPT is a fuzzer-side finding: it means the compiler
silently accepted an illegal program. We report that count but **never**
inflate it into a "bug" without manual triage.

A crashed-badly outcome is when the validator raised an unexpected
exception type (e.g. plain RuntimeError, AttributeError) instead of one
of the documented diagnostic types. Those are reported separately so a
reviewer can inspect them.

Categories:

* `incompatible_matrix_shapes`        — LINEAR's ``in_features != x.shape[-1]``
* `invalid_broadcast_add`             — ADD operands with shapes that don't broadcast
* `invalid_view_size`                 — VIEW whose target shape's product != input size
* `invalid_permute_axes`              — PERMUTE with out-of-range or wrong-length axes
* `unsupported_op_for_backend`        — Op kind not in the requested backend's set
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import (
    BackendLegalityError,
    backend_legality_pass,
    shape_inference_pass,
)
from graph_reference_interpreter import (
    GraphReferenceInterpreter,
    GraphReferenceInterpreterError,
)


INVALID_KINDS: Tuple[str, ...] = (
    "incompatible_matrix_shapes",
    "invalid_broadcast_add",
    "invalid_view_size",
    "invalid_permute_axes",
    "unsupported_op_for_backend",
)


# Exceptions accepted as "clean rejection" diagnostics, partitioned by validator
# entry point. Anything outside these lists is bucketed as `crashed_badly`.
_SHAPE_INFER_CLEAN_TYPES: Tuple[type, ...] = (
    ValueError,
    KeyError,
    IndexError,
    AssertionError,
    TypeError,
)
_INTERPRETER_CLEAN_TYPES: Tuple[type, ...] = (
    GraphReferenceInterpreterError,
    ValueError,
    IndexError,
    KeyError,
    AssertionError,
)
_BACKEND_CLEAN_TYPES: Tuple[type, ...] = (BackendLegalityError,)


@dataclass(frozen=True)
class InvalidGeneratedProgram:
    """Intentionally invalid program bundle.

    Attributes:
      - ``seed``: seed that produced this program (deterministic).
      - ``graph``: a Graph IR object that is invalid in exactly one
        documented way.
      - ``inputs``: sample input tensors matching ``graph.inputs`` (may
        themselves be intentionally shape-incompatible with the ops).
      - ``invalid_kind``: which of ``INVALID_KINDS`` this program embodies.
      - ``expected_validator``: which validator we expect to catch the
        invalidity. One of ``"shape_inference"``,
        ``"backend_legality"``, ``"reference_interpreter"``.
      - ``expected_backend``: target backend for ``backend_legality``
        rejections (None otherwise).
    """

    seed: int
    graph: GraphIR
    inputs: List[np.ndarray]
    invalid_kind: str
    expected_validator: str
    expected_backend: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RejectionOutcome:
    """The result of running an invalid graph through its expected validator.

    Categories:
      - ``rejected_cleanly``: validator raised an expected exception type.
      - ``unexpectedly_accepted``: validator returned without raising.
      - ``crashed_badly``: validator raised an unexpected exception type.
    """

    invalid_kind: str
    validator: str
    rejected_cleanly: bool
    unexpectedly_accepted: bool
    crashed_badly: bool
    exception_type: Optional[str]
    exception_message: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invalid_kind": self.invalid_kind,
            "validator": self.validator,
            "rejected_cleanly": bool(self.rejected_cleanly),
            "unexpectedly_accepted": bool(self.unexpectedly_accepted),
            "crashed_badly": bool(self.crashed_badly),
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


def _name(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx}"


def _make_input(rng: random.Random, shape: Tuple[int, ...], scale: float = 1.0) -> np.ndarray:
    state = np.random.RandomState(rng.randint(0, 2**31 - 1))
    return state.uniform(-scale, scale, size=shape).astype(np.float32)


def _make_weight(rng: random.Random, shape: Tuple[int, ...], scale: float = 0.1) -> np.ndarray:
    state = np.random.RandomState(rng.randint(0, 2**31 - 1))
    return state.uniform(-scale, scale, size=shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-kind invalid graph builders
# ---------------------------------------------------------------------------


def _build_incompatible_matrix_shapes(seed: int) -> InvalidGeneratedProgram:
    """LINEAR with ``in_features != x.shape[-1]`` — caught by reference interpreter."""
    rng = random.Random(int(seed))
    M = rng.choice([1, 2, 4])
    K_actual = rng.choice([16, 32, 64])
    K_declared = K_actual + rng.choice([1, 8, 16])  # mismatch
    N = rng.choice([16, 32])
    g = GraphIR(name=f"invalid_linear_shape_seed{seed}")
    g.add_value("x", shape=(M, K_actual), dtype="torch.float32")
    g.inputs = ["x"]
    inputs_list = [_make_input(rng, (M, K_actual))]
    weight = _make_weight(rng, (N, K_declared))
    g.add_value("y", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="bad_linear",
            op=OpKind.LINEAR,
            inputs=["x"],
            outputs=["y"],
            attrs={
                "weight": weight,
                "in_features": int(K_declared),
                "out_features": int(N),
            },
        )
    )
    g.outputs = ["y"]
    return InvalidGeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        invalid_kind="incompatible_matrix_shapes",
        expected_validator="reference_interpreter",
        metadata={"K_actual": K_actual, "K_declared": K_declared},
    )


def _build_invalid_broadcast_add(seed: int) -> InvalidGeneratedProgram:
    """ADD with operands that NumPy cannot broadcast together."""
    rng = random.Random(int(seed))
    M = rng.choice([2, 4, 8])
    N = rng.choice([8, 16, 32])
    extra = rng.choice([3, 5, 7])  # coprime-ish so broadcast fails
    g = GraphIR(name=f"invalid_broadcast_seed{seed}")
    g.add_value("x", shape=(M, N), dtype="torch.float32")
    g.add_value("y", shape=(extra,), dtype="torch.float32")
    g.inputs = ["x", "y"]
    inputs_list = [_make_input(rng, (M, N)), _make_input(rng, (extra,))]
    g.add_value("out", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="bad_add",
            op=OpKind.ADD,
            inputs=["x", "y"],
            outputs=["out"],
            attrs={},
        )
    )
    g.outputs = ["out"]
    return InvalidGeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        invalid_kind="invalid_broadcast_add",
        expected_validator="reference_interpreter",
        metadata={"shape_a": (M, N), "shape_b": (extra,)},
    )


def _build_invalid_view_size(seed: int) -> InvalidGeneratedProgram:
    """VIEW whose target shape's product disagrees with the input element count."""
    rng = random.Random(int(seed))
    M = rng.choice([2, 4])
    K = rng.choice([8, 16, 32])
    # Target shape product = 2*K+3 (always different from M*K for the M values above)
    target = (2, K + 3)
    g = GraphIR(name=f"invalid_view_seed{seed}")
    g.add_value("x", shape=(M, K), dtype="torch.float32")
    g.inputs = ["x"]
    inputs_list = [_make_input(rng, (M, K))]
    g.add_value("y", shape=target, dtype="torch.float32")
    g.add_op(
        OpNode(
            name="bad_view",
            op=OpKind.VIEW,
            inputs=["x"],
            outputs=["y"],
            attrs={"args": (target,)},
        )
    )
    g.outputs = ["y"]
    return InvalidGeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        invalid_kind="invalid_view_size",
        expected_validator="reference_interpreter",
        metadata={"input_shape": (M, K), "target_shape": target},
    )


def _build_invalid_permute_axes(seed: int) -> InvalidGeneratedProgram:
    """PERMUTE with axes that are out of range for the input rank."""
    rng = random.Random(int(seed))
    M = rng.choice([2, 4])
    K = rng.choice([8, 16])
    bad_axes = (0, 2)  # input is rank-2 → axis 2 is out of range
    g = GraphIR(name=f"invalid_permute_seed{seed}")
    g.add_value("x", shape=(M, K), dtype="torch.float32")
    g.inputs = ["x"]
    inputs_list = [_make_input(rng, (M, K))]
    g.add_value("y", shape=(M, K), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="bad_permute",
            op=OpKind.PERMUTE,
            inputs=["x"],
            outputs=["y"],
            attrs={"args": (bad_axes,)},
        )
    )
    g.outputs = ["y"]
    return InvalidGeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        invalid_kind="invalid_permute_axes",
        expected_validator="reference_interpreter",
        metadata={"axes": bad_axes, "input_rank": 2},
    )


def _build_unsupported_op_for_backend(seed: int) -> InvalidGeneratedProgram:
    """Emit an op that's legal on cuda but NOT on the utpu backend.

    The utpu backend only supports LINEAR / LINEAR_RELU; emitting CONV2D
    in a graph and asking `backend_legality_pass(graph, "utpu")` MUST
    raise `BackendLegalityError`.
    """
    rng = random.Random(int(seed))
    # Tiny CONV2D shapes so weight/input alloc is cheap. The validator
    # only checks op kind, never executes; so we don't need real conv math.
    g = GraphIR(name=f"invalid_unsupported_op_seed{seed}")
    g.add_value("x", shape=(1, 1, 4, 4), dtype="torch.float32")
    g.inputs = ["x"]
    inputs_list = [_make_input(rng, (1, 1, 4, 4))]
    weight = _make_weight(rng, (1, 1, 3, 3))
    g.add_value("y", shape=(1, 1, 2, 2), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="bad_conv2d_for_utpu",
            op=OpKind.CONV2D,
            inputs=["x"],
            outputs=["y"],
            attrs={"weight": weight, "stride": 1, "padding": 0, "groups": 1},
        )
    )
    g.outputs = ["y"]
    return InvalidGeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        invalid_kind="unsupported_op_for_backend",
        expected_validator="backend_legality",
        expected_backend="utpu",
        metadata={"target_backend": "utpu", "offending_op": OpKind.CONV2D},
    )


_INVALID_DISPATCH = {
    "incompatible_matrix_shapes": _build_incompatible_matrix_shapes,
    "invalid_broadcast_add": _build_invalid_broadcast_add,
    "invalid_view_size": _build_invalid_view_size,
    "invalid_permute_axes": _build_invalid_permute_axes,
    "unsupported_op_for_backend": _build_unsupported_op_for_backend,
}
for _kind in INVALID_KINDS:
    assert _kind in _INVALID_DISPATCH, f"invalid_kind '{_kind}' has no dispatch entry"


def generate_invalid_program(seed: int, kind: Optional[str] = None) -> InvalidGeneratedProgram:
    """Generate one invalid program. Deterministic given (seed, kind)."""
    rng_pick = random.Random(int(seed))
    chosen = kind if kind is not None else rng_pick.choice(list(INVALID_KINDS))
    if chosen not in _INVALID_DISPATCH:
        raise ValueError(f"unknown invalid_kind {chosen!r}; valid: {sorted(_INVALID_DISPATCH)}")
    return _INVALID_DISPATCH[chosen](seed)


def check_rejection(program: InvalidGeneratedProgram) -> RejectionOutcome:
    """Run the invalid program through its expected validator.

    Returns a `RejectionOutcome` classifying the result. Never raises — a
    `crashed_badly` outcome is recorded with the offending exception
    type and message so the runner can inspect it later.
    """
    validator = program.expected_validator
    try:
        if validator == "shape_inference":
            shape_inference_pass(program.graph)
        elif validator == "backend_legality":
            target = program.expected_backend or "utpu"
            backend_legality_pass(program.graph, target)
        elif validator == "reference_interpreter":
            # Most invalidities only surface during execution; we run a
            # shape_inference_pass first (it may already raise for
            # straightforward shape contradictions) and fall through to
            # the interpreter otherwise.
            try:
                shape_inference_pass(program.graph)
            except _SHAPE_INFER_CLEAN_TYPES as e:
                return RejectionOutcome(
                    invalid_kind=program.invalid_kind,
                    validator="shape_inference",
                    rejected_cleanly=True,
                    unexpectedly_accepted=False,
                    crashed_badly=False,
                    exception_type=type(e).__name__,
                    exception_message=str(e)[:240],
                )
            GraphReferenceInterpreter(program.graph).run(*program.inputs)
        else:
            return RejectionOutcome(
                invalid_kind=program.invalid_kind,
                validator=validator,
                rejected_cleanly=False,
                unexpectedly_accepted=False,
                crashed_badly=True,
                exception_type="ValueError",
                exception_message=f"unknown validator '{validator}'",
            )
    except _BACKEND_CLEAN_TYPES as e:
        return RejectionOutcome(
            invalid_kind=program.invalid_kind,
            validator=validator,
            rejected_cleanly=True,
            unexpectedly_accepted=False,
            crashed_badly=False,
            exception_type=type(e).__name__,
            exception_message=str(e)[:240],
        )
    except _INTERPRETER_CLEAN_TYPES as e:
        return RejectionOutcome(
            invalid_kind=program.invalid_kind,
            validator=validator,
            rejected_cleanly=True,
            unexpectedly_accepted=False,
            crashed_badly=False,
            exception_type=type(e).__name__,
            exception_message=str(e)[:240],
        )
    except Exception as e:  # noqa: BLE001
        return RejectionOutcome(
            invalid_kind=program.invalid_kind,
            validator=validator,
            rejected_cleanly=False,
            unexpectedly_accepted=False,
            crashed_badly=True,
            exception_type=type(e).__name__,
            exception_message=str(e)[:240],
        )
    return RejectionOutcome(
        invalid_kind=program.invalid_kind,
        validator=validator,
        rejected_cleanly=False,
        unexpectedly_accepted=True,
        crashed_badly=False,
        exception_type=None,
        exception_message=None,
    )


def invalid_coverage_summary(
    programs: Sequence[InvalidGeneratedProgram],
) -> Dict[str, Any]:
    """Aggregate coverage across an invalid-graph corpus."""
    kinds: set = set()
    validators: set = set()
    for p in programs:
        kinds.add(p.invalid_kind)
        validators.add(p.expected_validator)
    return {
        "invalid_kinds_covered": sorted(kinds),
        "invalid_validators_covered": sorted(validators),
    }
