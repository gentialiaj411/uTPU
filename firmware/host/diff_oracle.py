"""Differential oracle wrapper (Task 0 / `utpu_upgrade_plan.md` §2.1).

Thin reusable API on top of the existing differential infrastructure
(`differential_test_harness.py` + `graph_reference_interpreter.py`).
Tasks 1 (megakernel correctness gate), 2 (fuzzer differential / metamorphic
oracle), and 3 (superoptimizer extraction verification) all share this
wrapper so backend-probe + skip-with-reason + tolerance-comparison logic
is written ONCE.

Honesty contract (inherited from `CLAUDE.md` / `AGENTS.md`):

- A requested backend that cannot run on this host produces `status="skipped"`
  with a human-readable reason. Never a fabricated zero / NaN output.
- `compare(..., rtol=0.0, atol=0.0)` requires bit-exact equality. This is
  the integer-fragment rule (uTPU INT4 / INT8) — the comparator enforces it
  and also reports `bit_exact` independently of tolerance so float callers
  can both gate on tolerance and observe whether their result happened to
  be bit-identical.
- No backend's output is mutated; comparison runs in float64 to avoid
  silently swallowing dtype differences.

Public surface (Tasks 1/2/3 import these names):

- `BackendResult`, `PairCompare`, `CompareResult`
- `BackendUnavailable` (sentinel for "can't run here, skip with reason")
- `run_all_backends(graph, inputs, backends=..., torch_module=None, ...)`
- `compare(outputs, rtol=1e-3, atol=1e-3)`
- `register_backend(name, runner)`  — plug-in point used by Task 1 to wire
  the `cuda_megakernel` runner once it lands; the fuzzer / superopt re-use
  the same hook for their backend variants.
- `available_backends()`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from graph_ir import GraphIR
from graph_reference_interpreter import GraphReferenceInterpreter


class BackendUnavailable(RuntimeError):
    """Raised by a backend runner when the backend cannot execute on this host.

    The wrapper catches this and emits `status="skipped"` with the exception
    message as the reason. This is the canonical way for a backend to say
    "skip me, not an error" without polluting the result dict.
    """


@dataclass(frozen=True)
class BackendResult:
    name: str
    status: str  # "ok" | "skipped" | "error"
    output: Optional[np.ndarray]
    reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "output_shape": list(self.output.shape) if self.output is not None else None,
            "output_dtype": str(self.output.dtype) if self.output is not None else None,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PairCompare:
    backend_a: str
    backend_b: str
    max_abs_error: float
    max_rel_error: float
    within_tolerance: bool
    bit_exact: bool
    shape_match: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_a": self.backend_a,
            "backend_b": self.backend_b,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "within_tolerance": self.within_tolerance,
            "bit_exact": self.bit_exact,
            "shape_match": self.shape_match,
        }


@dataclass(frozen=True)
class CompareResult:
    rtol: float
    atol: float
    pairs: List[PairCompare]
    all_within_tolerance: bool
    all_bit_exact: bool
    backends_compared: List[str]
    backends_skipped: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rtol": self.rtol,
            "atol": self.atol,
            "all_within_tolerance": self.all_within_tolerance,
            "all_bit_exact": self.all_bit_exact,
            "backends_compared": list(self.backends_compared),
            "backends_skipped": list(self.backends_skipped),
            "pairs": [p.to_dict() for p in self.pairs],
        }


BackendRunner = Callable[..., np.ndarray]


def _numpy_reference_runner(graph: GraphIR, inputs: Sequence[Any], **_: Any) -> np.ndarray:
    out = GraphReferenceInterpreter(graph).run(*inputs)
    if isinstance(out, tuple):
        if len(out) != 1:
            raise BackendUnavailable(
                "multi-output graphs not supported by diff_oracle v1; "
                "call GraphReferenceInterpreter directly and compare per-output"
            )
        out = out[0]
    return np.asarray(out)


def _cuda_runner_stub(graph: GraphIR, inputs: Sequence[Any], **_: Any) -> np.ndarray:
    raise BackendUnavailable(
        "cuda backend not yet wired into diff_oracle "
        "(Task 1 will plug this in via compile_mlp_model + CompiledMLPRuntime; "
        "in the meantime callers can register a custom runner via register_backend)"
    )


def _cuda_megakernel_runner_stub(graph: GraphIR, inputs: Sequence[Any], **_: Any) -> np.ndarray:
    raise BackendUnavailable(
        "cuda_megakernel backend not yet implemented (Task 1 will register the real runner)"
    )


def _eager_runner(
    graph: GraphIR,
    inputs: Sequence[Any],
    torch_module: Any = None,
    **_: Any,
) -> np.ndarray:
    if torch_module is None:
        raise BackendUnavailable(
            "eager backend requires a `torch_module` keyword "
            "(the nn.Module the graph was lowered from)"
        )
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - importerror class varies by env
        raise BackendUnavailable(f"torch unavailable: {e}")
    tensors = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in inputs]
    with torch.no_grad():
        out = torch_module(*tensors)
    return out.detach().cpu().numpy()


def _torch_inductor_runner(
    graph: GraphIR,
    inputs: Sequence[Any],
    torch_module: Any = None,
    **_: Any,
) -> np.ndarray:
    # NOTE: A real Inductor oracle that is meant to run alongside an NVRTC kernel
    # in the same process MUST be invoked in a subprocess (the NVRTC and
    # Torch/Inductor CUDA contexts clash — see `inductor_oracle_subprocess.py`
    # and `_cublas_baseline_torch_subprocess.py`). This in-process runner is
    # the CPU / single-context path; Task 2 wires the subprocess variant.
    if torch_module is None:
        raise BackendUnavailable("torch_inductor backend requires a `torch_module` keyword")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise BackendUnavailable(f"torch unavailable: {e}")
    if not hasattr(torch, "compile"):
        raise BackendUnavailable("torch.compile unavailable on this torch build")
    try:
        compiled = torch.compile(torch_module, backend="inductor", fullgraph=True)
        tensors = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in inputs]
        with torch.no_grad():
            out = compiled(*tensors)
    except Exception as e:  # noqa: BLE001 - inductor raises a wide range
        raise BackendUnavailable(f"torch.compile(inductor) failed: {e}")
    return out.detach().cpu().numpy()


_BACKEND_RUNNERS: Dict[str, BackendRunner] = {
    "numpy_reference": _numpy_reference_runner,
    "cuda": _cuda_runner_stub,
    "cuda_megakernel": _cuda_megakernel_runner_stub,
    "eager": _eager_runner,
    "torch_inductor": _torch_inductor_runner,
}


DEFAULT_BACKENDS: tuple = (
    "numpy_reference",
    "cuda",
    "cuda_megakernel",
    "eager",
    "torch_inductor",
)


def register_backend(name: str, runner: BackendRunner) -> None:
    """Register or replace a backend runner.

    Task 1 calls this to plug in the real `cuda_megakernel` runner; Tasks 2/3
    use the same hook for their differential-oracle backend variants.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("backend name must be a non-empty string")
    if not callable(runner):
        raise TypeError("runner must be callable")
    _BACKEND_RUNNERS[name] = runner


def available_backends() -> List[str]:
    return list(_BACKEND_RUNNERS.keys())


def run_all_backends(
    graph: GraphIR,
    inputs: Sequence[Any],
    backends: Iterable[str] = DEFAULT_BACKENDS,
    torch_module: Any = None,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, BackendResult]:
    """Run one Graph IR program through every requested backend.

    Returns one `BackendResult` per requested backend keyed by name.

    Contract:
    - Every requested backend appears in the returned dict (never silently dropped).
    - An unavailable backend produces `status="skipped"` with a reason.
    - A runner raising any non-`BackendUnavailable` exception produces
      `status="error"` with the exception summary as the reason.
    - The output ndarray is whatever the runner returned, coerced via `np.asarray`.
    """
    if not isinstance(graph, GraphIR):
        raise TypeError(f"graph must be GraphIR, got {type(graph).__name__}")
    inputs_list = list(inputs)
    ctx: Dict[str, Any] = {"torch_module": torch_module}
    if extra_context:
        ctx.update(extra_context)

    results: Dict[str, BackendResult] = {}
    seen: set = set()
    for raw_name in backends:
        name = str(raw_name)
        if name in seen:
            continue
        seen.add(name)
        runner = _BACKEND_RUNNERS.get(name)
        if runner is None:
            results[name] = BackendResult(
                name=name,
                status="skipped",
                output=None,
                reason=f"no runner registered for backend '{name}'",
            )
            continue
        try:
            raw_output = runner(graph, inputs_list, **ctx)
        except BackendUnavailable as e:
            results[name] = BackendResult(name=name, status="skipped", output=None, reason=str(e))
            continue
        except Exception as e:  # noqa: BLE001 - we intentionally catch broadly here
            results[name] = BackendResult(
                name=name,
                status="error",
                output=None,
                reason=f"{type(e).__name__}: {e}",
            )
            continue
        try:
            arr = np.asarray(raw_output)
        except Exception as e:  # noqa: BLE001
            results[name] = BackendResult(
                name=name,
                status="error",
                output=None,
                reason=f"output coercion to ndarray failed: {e}",
            )
            continue
        results[name] = BackendResult(name=name, status="ok", output=arr, reason=None)
    return results


def _pair_compare(
    name_a: str,
    a: np.ndarray,
    name_b: str,
    b: np.ndarray,
    atol: float,
    rtol: float,
) -> PairCompare:
    shape_match = a.shape == b.shape
    if not shape_match:
        return PairCompare(
            backend_a=name_a,
            backend_b=name_b,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            within_tolerance=False,
            bit_exact=False,
            shape_match=False,
        )
    if a.size == 0:
        return PairCompare(
            backend_a=name_a,
            backend_b=name_b,
            max_abs_error=0.0,
            max_rel_error=0.0,
            within_tolerance=True,
            bit_exact=True,
            shape_match=True,
        )
    bit_exact = bool(np.array_equal(a, b))
    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    diff = np.abs(af - bf)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(af), 1e-12)
    max_rel = float((diff / denom).max())
    if atol == 0.0 and rtol == 0.0:
        within = bit_exact
    else:
        within = bool(np.allclose(bf, af, atol=atol, rtol=rtol))
    return PairCompare(
        backend_a=name_a,
        backend_b=name_b,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        within_tolerance=within,
        bit_exact=bit_exact,
        shape_match=True,
    )


def compare(
    outputs: Dict[str, BackendResult],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> CompareResult:
    """Pairwise compare every `(ok, ok)` backend output pair.

    Integer-fragment rule: pass `rtol=atol=0.0` to demand bit-exact equality.
    The result reports `all_bit_exact` independently of tolerance so callers
    can both gate strictly and report the max-abs envelope without re-running.

    Skipped / errored backends are listed in `backends_skipped` and excluded
    from pair comparison (so a missing CUDA host does not poison the gate).
    With no comparable pairs the result is vacuously
    `all_within_tolerance=True` / `all_bit_exact=True`; callers that want to
    fail on "no backends compared" should check `len(backends_compared) >= 2`.
    """
    if rtol < 0 or atol < 0:
        raise ValueError(f"rtol/atol must be non-negative, got rtol={rtol}, atol={atol}")

    ok_items: List[tuple] = [
        (name, res.output)
        for name, res in outputs.items()
        if res.status == "ok" and res.output is not None
    ]
    skipped = [name for name, res in outputs.items() if res.status != "ok"]

    pairs: List[PairCompare] = []
    for i in range(len(ok_items)):
        for j in range(i + 1, len(ok_items)):
            name_a, a = ok_items[i]
            name_b, b = ok_items[j]
            pairs.append(_pair_compare(name_a, a, name_b, b, atol=atol, rtol=rtol))

    all_within = all(p.within_tolerance for p in pairs) if pairs else True
    all_bit = all(p.bit_exact for p in pairs) if pairs else True
    return CompareResult(
        rtol=float(rtol),
        atol=float(atol),
        pairs=pairs,
        all_within_tolerance=all_within,
        all_bit_exact=all_bit,
        backends_compared=[n for n, _ in ok_items],
        backends_skipped=skipped,
    )
