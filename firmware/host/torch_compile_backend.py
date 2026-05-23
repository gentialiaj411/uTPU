import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from fx_importer import FXImportError, import_fx_graph_module
from graph_passes import BackendLegalityError, is_op_supported_for_backend, supported_ops_for_backend
from pytorch_compiler import compile_model


def _require_torch():
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for torch.compile backend") from e
    return torch


def _as_input_tuple(example_inputs: Any) -> Tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def _dynamo_available() -> bool:
    torch = _require_torch()
    return hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "register_backend")


def _target_backend_name() -> str:
    return (os.environ.get("UTPU_TORCH_COMPILE_TARGET", "cuda") or "cuda").strip().lower()


@dataclass
class _CompileBackendStats:
    backend_name: str = "utpu"
    subgraphs_seen: int = 0
    subgraphs_compiled: int = 0
    subgraphs_fallback: int = 0
    unsupported_op_counts: Dict[str, int] = field(default_factory=dict)
    models_exercised: List[str] = field(default_factory=list)

    def bump_unsupported(self, name: str) -> None:
        key = str(name)
        self.unsupported_op_counts[key] = int(self.unsupported_op_counts.get(key, 0)) + 1

    def as_dict(self) -> Dict[str, Any]:
        target = _target_backend_name()
        return {
            "backend_name": self.backend_name,
            "subgraphs_seen": int(self.subgraphs_seen),
            "subgraphs_compiled": int(self.subgraphs_compiled),
            "subgraphs_fallback": int(self.subgraphs_fallback),
            "unsupported_op_counts": {k: int(v) for k, v in sorted(self.unsupported_op_counts.items())},
            "supported_op_set": sorted(supported_ops_for_backend(target)),
            "models_exercised": sorted(set(self.models_exercised)),
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "torch_version": str(_require_torch().__version__),
            "dynamo_available": bool(_dynamo_available()),
        }


_STATS = _CompileBackendStats()
_REGISTERED = False


def reset_stats() -> None:
    global _STATS
    _STATS = _CompileBackendStats()


def get_stats() -> Dict[str, Any]:
    return copy.deepcopy(_STATS.as_dict())


def dump_stats_artifact(path: str = "build/reports/torch_compile_backend_report.json") -> Dict[str, Any]:
    payload = get_stats()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _classify_gm_supported(gm: Any, example_inputs: Tuple[Any, ...], target: str) -> Tuple[bool, List[str]]:
    torch = _require_torch()
    from torch.fx.passes.shape_prop import ShapeProp

    try:
        ShapeProp(gm).propagate(*example_inputs)
        graph = import_fx_graph_module(gm, name=gm.__class__.__name__)
    except FXImportError as e:
        return False, [f"fx_import:{e.__class__.__name__}"]

    unsupported = [op.op for op in graph.ops if not is_op_supported_for_backend(op.op, target)]
    return len(unsupported) == 0, unsupported


def utpu_backend(gm: Any, example_inputs: Any) -> Callable:
    inputs = _as_input_tuple(example_inputs)
    target = _target_backend_name()
    _STATS.subgraphs_seen += 1
    _STATS.models_exercised.append(getattr(gm, "__class__", type(gm)).__name__)

    supported, unsupported = _classify_gm_supported(gm, inputs, target)
    if not supported:
        _STATS.subgraphs_fallback += 1
        for name in unsupported:
            _STATS.bump_unsupported(name)
        return gm.forward

    try:
        compiled = compile_model(gm, inputs, target=target)
        if not compiled.ok or compiled.runtime is None:
            _STATS.subgraphs_fallback += 1
            _STATS.bump_unsupported("compile_pipeline_not_ok")
            return gm.forward
    except BackendLegalityError as e:
        _STATS.subgraphs_fallback += 1
        for offending in e.to_dict().get("offending_ops", []):
            _STATS.bump_unsupported(str(offending.get("op", "backend_legality_error")))
        return gm.forward
    except Exception:
        _STATS.subgraphs_fallback += 1
        _STATS.bump_unsupported("compile_pipeline_error")
        return gm.forward

    _STATS.subgraphs_compiled += 1

    def _compiled_callable(*args):
        return compiled(*args, mode="compiled")

    return _compiled_callable


def register_backend() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    if not _dynamo_available():
        return False
    torch = _require_torch()
    torch._dynamo.register_backend(name="utpu", compiler_fn=utpu_backend)
    _REGISTERED = True
    return True
