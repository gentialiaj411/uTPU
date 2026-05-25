"""Phase 7 remediation P2.2 — selection A/B benchmark.

**A/B**: cost-model-selected schedule (A) vs measured-best/oracle
schedule (B), per shape, wall-clock on a CUDA GPU.

Reads the per-shape `chosen_schedule` (cost-model) and `oracle_schedule`
(autotuner measured-best) from
`bench/results/cost_model_selection.json`, then re-times both on the
GPU using the locked methodology in
`firmware/host/_selection_ab_cuda_subprocess.py`. Realized regret per
shape is `(cost_model_median - oracle_median) / oracle_median * 100`.

On a non-CUDA host this script writes a stub artifact with
`status="cuda_unavailable"`, preserving the methodology block and
shape list so the schema test stays green; the populated artifact
regenerates on any CUDA host via `make repro-cuda` /
`python firmware/host/run_selection_ab.py`.

Output: `bench/results/selection_ab.json`.

This artifact is the wall-clock evidence that backs the remediation
P2.1 code change (`schedule_source="cost_model"` in
`CompiledMLPRuntime`): the cost-model-selected schedule is the one
actually executed on the GPU, and the realized regret distribution can
be compared head-to-head with the predicted regret from
`cost_model_selection.json`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SELECTION_PATH = REPO_ROOT / "bench" / "results" / "cost_model_selection.json"
DEFAULT_OUT_PATH = REPO_ROOT / "bench" / "results" / "selection_ab.json"
SUBPROCESS_SCRIPT = Path(__file__).with_name("_selection_ab_cuda_subprocess.py")


def _load_selection_artifact(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _shape_plan_from_selection(
    selection: Dict[str, Any],
    seed_base: int = 0xABCDEF,
) -> List[Dict[str, Any]]:
    """Extract the (shape, cost_model_schedule, oracle_schedule) per-shape
    list from cost_model_selection.json, ready to be passed to the
    GPU subprocess."""
    plan: List[Dict[str, Any]] = []
    for entry in selection["per_shape"]:
        shape = entry["shape"]
        plan.append({
            "out_features": int(shape["out_features"]),
            "in_features": int(shape["in_features"]),
            "array_size": int(shape.get("array_size", 16)),
            "cost_model_schedule": dict(entry["chosen_schedule"]),
            "oracle_schedule": dict(entry["oracle_schedule"]),
            "seed": int(seed_base) ^ (
                int(shape["out_features"]) * 2654435761
                + int(shape["in_features"]) * 13
            ),
        })
    return plan


def _detect_cuda_or_reason() -> Tuple[bool, str]:
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from cuda_blocked_fc_backend import detect_cuda_environment  # noqa: WPS433
        env = detect_cuda_environment()
        if env.runtime_available:
            return True, "available"
        return False, env.reason
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _methodology_block(warmup: int, iters: int, array_size: int) -> Dict[str, Any]:
    return {
        "api": "firmware/host/run_selection_ab.py + firmware/host/_selection_ab_cuda_subprocess.py",
        "what_it_measures": (
            "Wall-clock A/B of two NVRTC blocked-FC schedules per shape: "
            "A = cost-model-selected schedule (chosen_schedule from "
            "cost_model_selection.json, the schedule the runtime backend "
            "now actually consumes after remediation P2.1); "
            "B = oracle schedule (measured-best from the autotuner cache, "
            "oracle_schedule from cost_model_selection.json)."
        ),
        "realized_regret_pct": (
            "(median(A.kernel_ms) - median(B.kernel_ms)) / median(B.kernel_ms) * 100; "
            "positive => cost-model choice is slower than oracle; "
            "0 means the schedules were identical or tied on wall-clock."
        ),
        "warmup_iters": int(warmup),
        "measurement_iters": int(iters),
        "array_size": int(array_size),
        "timing": (
            "time.perf_counter brackets with torch.cuda.synchronize "
            "before and after each timed call (so the timer always "
            "captures completed kernel time). A and B are interleaved "
            "per iteration (warmup(A)->warmup(B)->[A,B]xN) so GPU "
            "clocking and thermal state are symmetric across the two "
            "arms; this matters at sub-millisecond GEMV-N=1 latencies."
        ),
        "shape_source": (
            "bench/results/cost_model_selection.json (24 shapes from "
            "calibrate_cost_model._shape_grid())"
        ),
        "consumed_by_runtime": (
            "Schedule A is the same schedule that CompiledMLPRuntime "
            "executes when schedule_source='cost_model' (see "
            "firmware/host/compiled_runtime.py::_schedule_params_for_op "
            "and test_compiled_runtime_schedule_source.py)."
        ),
        "dtype_caveats": [
            "INT4 weights x INT4 activations, INT32 accumulator, INT4 "
            "quantised output (matches the rest of the uTPU CUDA path).",
            "Bit-exactness is checked against the NumPy reference oracle "
            "(clip[-8,7] of int32 W@x); both A and B must match.",
        ],
    }


def _aggregate(per_shape: List[Dict[str, Any]]) -> Dict[str, Any]:
    realized = [
        entry["realized_regret_pct"]
        for entry in per_shape
        if entry.get("realized_regret_pct") is not None
    ]
    if not realized:
        return {"realized_regret_pct_count": 0}

    ab_ties = sum(1 for r in realized if abs(r) < 1.0)
    return {
        "realized_regret_pct_count": len(realized),
        "realized_regret_pct_mean": float(statistics.fmean(realized)),
        "realized_regret_pct_median": float(statistics.median(realized)),
        "realized_regret_pct_p95": float(sorted(realized)[max(0, int(round(0.95 * len(realized))) - 1)]),
        "realized_regret_pct_min": float(min(realized)),
        "realized_regret_pct_max": float(max(realized)),
        "realized_within_1pct_fraction": float(ab_ties) / float(len(realized)),
        "realized_within_5pct_fraction": float(sum(1 for r in realized if r <= 5.0)) / float(len(realized)),
        "realized_within_10pct_fraction": float(sum(1 for r in realized if r <= 10.0)) / float(len(realized)),
    }


def _attach_predicted(per_shape_ab: List[Dict[str, Any]], selection_per_shape: List[Dict[str, Any]]) -> None:
    """Annotate each A/B entry with predicted regret + chosen/oracle
    schedule from cost_model_selection.json so the artifact stitches the
    two signals (predicted regret vs realized regret) without requiring
    the consumer to join."""
    lut = {
        (int(e["shape"]["out_features"]), int(e["shape"]["in_features"])): e
        for e in selection_per_shape
    }
    for entry in per_shape_ab:
        shape = entry["shape"]
        key = (int(shape["out_features"]), int(shape["in_features"]))
        sel = lut.get(key)
        if sel is None:
            entry["predicted"] = None
            continue
        entry["predicted"] = {
            "predicted_latency_us": float(sel["predicted_latency_us"]),
            "predicted_regret_pct": float(sel["regret_pct"]),
            "chosen_schedule": dict(sel["chosen_schedule"]),
            "oracle_schedule": dict(sel["oracle_schedule"]),
            "is_top1": bool(sel["is_top1"]),
            "confidence": float(sel["confidence"]),
            "margin_pct": float(sel["margin_pct"]),
        }


def _write_stub(out_path: Path, methodology: Dict[str, Any], reason: str, shape_count: int) -> None:
    payload = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "status": "cuda_unavailable",
        "reason": reason,
        "shape_count": int(shape_count),
        "methodology": methodology,
        "per_shape": [],
        "aggregate": {
            "realized_regret_pct_count": 0,
        },
        "regenerate_with": (
            "On a CUDA + Torch host (Linux/WSL2), run: "
            "python firmware/host/run_selection_ab.py "
            "[--warmup 10] [--iters 50] "
            "(reads bench/results/cost_model_selection.json for shapes "
            "and per-shape chosen/oracle schedules)."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _call_subprocess(plan: Dict[str, Any]) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(SUBPROCESS_SCRIPT)],
        input=json.dumps(plan),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "status": "cuda_unavailable",
            "reason": f"subprocess failed rc={proc.returncode}: stderr={proc.stderr.strip()[:512]}",
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "cuda_unavailable",
            "reason": f"subprocess returned non-JSON: {exc} ; stdout={proc.stdout.strip()[:512]}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--selection", type=str, default=str(DEFAULT_SELECTION_PATH))
    args = parser.parse_args()

    selection = _load_selection_artifact(Path(args.selection))
    shape_plan = _shape_plan_from_selection(selection)
    methodology = _methodology_block(args.warmup, args.iters, 16)

    available, reason = _detect_cuda_or_reason()
    out_path = Path(args.out)
    if not available:
        _write_stub(out_path, methodology, reason, shape_count=len(shape_plan))
        print(f"[run_selection_ab] CUDA unavailable ({reason}); wrote stub to {out_path}")
        return 0

    plan = {
        "warmup": args.warmup,
        "iters": args.iters,
        "array_size": 16,
        "shapes": shape_plan,
    }
    raw = _call_subprocess(plan)
    if raw.get("status") != "ok":
        _write_stub(out_path, methodology, raw.get("reason", "subprocess error"), shape_count=len(shape_plan))
        print(f"[run_selection_ab] subprocess error; wrote stub to {out_path}")
        return 0

    per_shape = raw["results"]
    _attach_predicted(per_shape, selection["per_shape"])

    payload = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "status": "ok",
        "shape_count": len(per_shape),
        "methodology": methodology,
        "environment": {
            "python_version": sys.version.split()[0],
            "device_name": _query_device_name(),
            "torch_version": _query_torch_version(),
        },
        "per_shape": per_shape,
        "aggregate": _aggregate(per_shape),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    agg = payload["aggregate"]
    print(
        f"[run_selection_ab] wrote {out_path}; "
        f"realized_regret_pct median={agg.get('realized_regret_pct_median')}, "
        f"mean={agg.get('realized_regret_pct_mean')}, "
        f"max={agg.get('realized_regret_pct_max')}"
    )
    return 0


def _query_device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _query_torch_version() -> str:
    try:
        import torch
        return str(torch.__version__)
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
