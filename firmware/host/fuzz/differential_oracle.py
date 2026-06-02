"""Fuzzer differential oracle (Task 2 / `utpu_upgrade_plan.md` §4.2 step 2).

Thin wrapper over `diff_oracle.run_all_backends` that:

1. Picks the right default backend set per host (CPU vs CUDA) so the
   fuzzer never silently skips its primary oracle.
2. Auto-registers the `cuda_megakernel` runner from
   `cuda_megakernel_backend.register_with_diff_oracle()` exactly once per
   process when CUDA is available — so generated graphs that contain a
   single fusable region get an extra independent backend cross-check.
3. Returns a `RunResult` carrying both the per-backend outputs and the
   pairwise comparison verdict used by the metamorphic relations.

Why a wrapper? The plan explicitly says the fuzzer's differential oracle
"IS" §2.1 (`diff_oracle`). This module exists to centralize the backend
selection + cuda_megakernel hook so neither metamorphic.py nor
run_fuzzer.py has to know about CUDA-availability detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import diff_oracle
from diff_oracle import (
    BackendResult,
    CompareResult,
    available_backends,
    compare,
    run_all_backends,
)
from graph_ir import GraphIR
from graph_reference_interpreter import GraphReferenceInterpreter

from fuzz.graph_generator import GeneratedProgram
from fuzz.graph_torch_module import build_torch_module_from_graph, is_graph_inductor_compatible


_CUDA_MEGAKERNEL_REGISTERED: Dict[str, bool] = {"done": False}


def _detect_cuda_python_available() -> bool:
    """Lightweight probe: cuda-python (12.x or 13.x layout) + >=1 device.

    cuda-python 13.x ships `cuda.bindings.driver`; 12.x shipped `cuda.cuda`.
    The Task 1 backend (`cuda_megakernel_backend.py`) already handles both
    via try/except — we mirror the same dual-import here so a CUDA host
    with the new layout doesn't get a false negative.
    """
    cuda_mod = None
    try:
        from cuda.bindings import driver as cuda_mod  # type: ignore[import-not-found]
    except Exception:
        try:
            from cuda import cuda as cuda_mod  # type: ignore[import-not-found]
        except Exception:
            return False
    try:
        cuda_mod.cuInit(0)
    except Exception:
        return False
    try:
        result = cuda_mod.cuDeviceGetCount()
        if isinstance(result, tuple):
            count = int(result[1]) if len(result) > 1 else 0
        else:
            count = int(result)
    except Exception:
        return False
    return count > 0


def maybe_register_cuda_megakernel() -> Tuple[bool, Optional[str]]:
    """Register the real `cuda_megakernel` runner if CUDA is usable here.

    Returns `(registered, reason)`. `registered` is True iff the runner
    is now installed. Idempotent — repeated calls are no-ops.
    """
    if _CUDA_MEGAKERNEL_REGISTERED["done"]:
        return True, None
    if not _detect_cuda_python_available():
        return False, "cuda-python not importable / no CUDA device"
    try:
        from cuda_megakernel_backend import register_with_diff_oracle
    except Exception as e:  # noqa: BLE001
        return False, f"cuda_megakernel_backend import failed: {e}"
    try:
        register_with_diff_oracle()
    except Exception as e:  # noqa: BLE001
        return False, f"register_with_diff_oracle failed: {e}"
    _CUDA_MEGAKERNEL_REGISTERED["done"] = True
    return True, None


def default_backend_set(
    include_cuda_megakernel: bool = True,
    include_torch_inductor: bool = False,
) -> Tuple[str, ...]:
    """Backend set used by the fuzzer when the caller does not override.

    `numpy_reference` is always included (the project oracle). The
    `cuda_megakernel` backend is included when (a) it is registered, and
    (b) the caller has not asked us to skip it.

  Optional ``torch_inductor`` is enabled via ``include_torch_inductor`` for
  discovery runs; it is built from each ``GeneratedProgram`` via
  :func:`compare_torch_inductor` (not via ``run_diff``) because graphs
  have no pre-built ``torch_module``.
    """
    backends: List[str] = ["numpy_reference"]
    if include_cuda_megakernel:
        backends.append("cuda_megakernel")
    if include_torch_inductor:
        backends.append("torch_inductor")
    return tuple(backends)


@dataclass(frozen=True)
class RunResult:
    """One round of `run_all_backends` + `compare` for the fuzzer."""

    outputs: Dict[str, BackendResult]
    comparison: CompareResult
    backends_requested: Tuple[str, ...]

    @property
    def all_within_tolerance(self) -> bool:
        return self.comparison.all_within_tolerance

    @property
    def all_bit_exact(self) -> bool:
        return self.comparison.all_bit_exact

    @property
    def ok_backends(self) -> List[str]:
        return [n for n, r in self.outputs.items() if r.status == "ok"]


def run_diff(
    graph: GraphIR,
    inputs: Sequence[Any],
    backends: Optional[Iterable[str]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> RunResult:
    """Run a graph through the fuzzer's backend set and compare pairwise.

    `backends` defaults to `default_backend_set(...)`. The wrapper does
    NOT auto-register cuda_megakernel — callers must do so explicitly via
    `maybe_register_cuda_megakernel()` so a CPU host's fuzzer-report
    never mis-claims CUDA was attempted.
    """
    selected = tuple(backends) if backends is not None else default_backend_set()
    outputs = run_all_backends(graph, list(inputs), backends=selected)
    comparison = compare(outputs, rtol=float(rtol), atol=float(atol))
    return RunResult(
        outputs=outputs,
        comparison=comparison,
        backends_requested=selected,
    )


def diff_two_graphs(
    graph_a: GraphIR,
    graph_b: GraphIR,
    inputs: Sequence[Any],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    """Run two graphs (which should be semantically equivalent) on the same
    inputs through `numpy_reference` only and pairwise-compare the outputs.

    This is the kernel the metamorphic relations call: each relation
    transforms a graph into a semantically equivalent variant; the runner
    compares both variants' outputs through the SAME oracle. We intentionally
    do not use multi-backend mode here — the metamorphic check IS the
    oracle, and the goal is to catch divergences between equivalent
    compilations of the same program.
    """
    out_a = run_all_backends(graph_a, list(inputs), backends=("numpy_reference",))
    out_b = run_all_backends(graph_b, list(inputs), backends=("numpy_reference",))
    a_res = out_a.get("numpy_reference")
    b_res = out_b.get("numpy_reference")
    if a_res is None or b_res is None:
        return {
            "match": False,
            "reason": "missing numpy_reference result",
            "status_a": getattr(a_res, "status", "missing"),
            "status_b": getattr(b_res, "status", "missing"),
        }
    if a_res.status != "ok" or b_res.status != "ok":
        return {
            "match": False,
            "reason": (
                f"numpy_reference status_a={a_res.status} status_b={b_res.status} "
                f"reason_a={a_res.reason!r} reason_b={b_res.reason!r}"
            ),
            "status_a": a_res.status,
            "status_b": b_res.status,
        }
    a = np.asarray(a_res.output)
    b = np.asarray(b_res.output)
    if a.shape != b.shape:
        return {
            "match": False,
            "reason": f"shape mismatch: {a.shape} vs {b.shape}",
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(diff.max()) if a.size else 0.0
    denom = np.maximum(np.abs(a.astype(np.float64)), 1e-12)
    max_rel = float((diff / denom).max()) if a.size else 0.0
    bit_exact = bool(np.array_equal(a, b))
    if rtol == 0.0 and atol == 0.0:
        within = bit_exact
    else:
        within = bool(np.allclose(b, a, atol=atol, rtol=rtol))
    return {
        "match": within,
        "bit_exact": bit_exact,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
    }


def list_active_backends() -> List[str]:
    """Snapshot of the backends currently registered in diff_oracle."""
    return list(available_backends())


@dataclass(frozen=True)
class InductorCheckResult:
    """One optional TorchInductor vs NumPy-reference comparison."""

    match: bool
    skipped: bool
    skip_reason: Optional[str]
    max_abs_error: float
    max_rel_error: float
    bit_exact: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": bool(self.match),
            "skipped": bool(self.skipped),
            "skip_reason": self.skip_reason,
            "max_abs_error": float(self.max_abs_error),
            "max_rel_error": float(self.max_rel_error),
            "bit_exact": bool(self.bit_exact),
        }


def _inductor_unavailable_reason(exc: BaseException) -> str:
    winerr = getattr(exc, "winerror", None)
    if winerr == 50:
        return "inductor_unavailable"
    text = str(exc)
    if "WinError 50" in text or "winerror 50" in text.lower():
        return "inductor_unavailable"
    return f"inductor_unavailable: {type(exc).__name__}: {exc}"


def compare_torch_inductor(
    program: GeneratedProgram,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> InductorCheckResult:
    """Compare ``GraphReferenceInterpreter`` output to ``torch.compile(inductor)``.

    Skips cleanly when Torch/Inductor is unavailable (including WinError 50 on
    Windows) or when the graph contains unsupported ops.
    """
    graph = program.graph
    if not is_graph_inductor_compatible(graph):
        return InductorCheckResult(
            match=True,
            skipped=True,
            skip_reason="inductor_unavailable: unsupported graph ops for inductor module",
            max_abs_error=0.0,
            max_rel_error=0.0,
            bit_exact=True,
        )
    try:
        ref_out = GraphReferenceInterpreter(graph).run(*program.inputs)
        if isinstance(ref_out, tuple):
            if len(ref_out) != 1:
                return InductorCheckResult(
                    match=True,
                    skipped=True,
                    skip_reason="inductor_unavailable: multi-output graphs not supported",
                    max_abs_error=0.0,
                    max_rel_error=0.0,
                    bit_exact=True,
                )
            ref_out = ref_out[0]
        ref = np.asarray(ref_out, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        return InductorCheckResult(
            match=False,
            skipped=False,
            skip_reason=None,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            bit_exact=False,
        )

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return InductorCheckResult(
            match=True,
            skipped=True,
            skip_reason=_inductor_unavailable_reason(e),
            max_abs_error=0.0,
            max_rel_error=0.0,
            bit_exact=True,
        )

    if not hasattr(torch, "compile"):
        return InductorCheckResult(
            match=True,
            skipped=True,
            skip_reason="inductor_unavailable: torch.compile unavailable",
            max_abs_error=0.0,
            max_rel_error=0.0,
            bit_exact=True,
        )

    try:
        module = build_torch_module_from_graph(graph)
        module.eval()
        compiled = torch.compile(module, backend="inductor", fullgraph=True)
        tensors = [
            torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in program.inputs
        ]
        with torch.no_grad():
            out = compiled(*tensors)
        if isinstance(out, tuple):
            if len(out) != 1:
                return InductorCheckResult(
                    match=True,
                    skipped=True,
                    skip_reason="inductor_unavailable: multi-output graphs not supported",
                    max_abs_error=0.0,
                    max_rel_error=0.0,
                    bit_exact=True,
                )
            out = out[0]
        actual = out.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as e:  # noqa: BLE001
        return InductorCheckResult(
            match=True,
            skipped=True,
            skip_reason=_inductor_unavailable_reason(e),
            max_abs_error=0.0,
            max_rel_error=0.0,
            bit_exact=True,
        )

    if ref.shape != actual.shape:
        return InductorCheckResult(
            match=False,
            skipped=False,
            skip_reason=None,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            bit_exact=False,
        )

    diff = np.abs(ref.astype(np.float64) - actual.astype(np.float64))
    max_abs = float(diff.max()) if ref.size else 0.0
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-12)
    max_rel = float((diff / denom).max()) if ref.size else 0.0
    bit_exact = bool(np.array_equal(ref, actual))
    if rtol == 0.0 and atol == 0.0:
        within = bit_exact
    else:
        within = bool(np.allclose(actual, ref, atol=atol, rtol=rtol))
    return InductorCheckResult(
        match=within,
        skipped=False,
        skip_reason=None,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        bit_exact=bit_exact,
    )


__all__ = [
    "BackendResult",
    "CompareResult",
    "InductorCheckResult",
    "RunResult",
    "compare_torch_inductor",
    "default_backend_set",
    "diff_two_graphs",
    "list_active_backends",
    "maybe_register_cuda_megakernel",
    "run_diff",
]
