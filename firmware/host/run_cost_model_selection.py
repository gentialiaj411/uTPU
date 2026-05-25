"""Phase 1 evidence script: measure how well `cost_model.select` picks the
winning CUDA blocked-FC schedule per shape.

Methodology
-----------
1. Load the frozen calibration artifact
   ``build/reports/cost_model_calibration.json``.
2. For every shape, take the median measured latency per schedule (matches
   the convention used by ``evaluate_pruned_autotuner.py``).
3. Treat the schedule with the minimum measured latency as the oracle.
4. Call ``cost_model.select`` over the same candidate set with the fitted
   coefficients from the calibration artifact and record the predicted
   winner.
5. Compute top-1 accuracy and regret (extra latency vs the oracle) for
   every shape. Aggregate.

The output artifact is ``bench/results/cost_model_selection.json``. The
companion test (``test_cost_model_selection.py``) consumes it and asserts
floor values that were captured from this run; the test never invents a
threshold the cost model has not already cleared.
"""

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cost_model import select as cost_model_select
from cuda_autotuner import load_cost_model_target


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_JSON = REPO_ROOT / "build" / "reports" / "cost_model_calibration.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "cost_model_selection.json"


def _median_by_shape_schedule(
    calibration: Dict[str, Any],
) -> Dict[Tuple[int, int, int], List[Dict[str, Any]]]:
    """Group calibration measurements into one row per (shape, schedule)
    holding the median measured latency. Returns ``{shape_key: [rows]}``
    keyed by ``(in_features, out_features, array_size)``.
    """
    raw: Dict[Tuple[int, int, int, int, int], List[float]] = defaultdict(list)
    shapes_seen: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for row in calibration["per_point"]:
        shape = row["shape_used"]
        schedule = row["schedule"]
        shape_key = (
            int(shape["in_features"]),
            int(shape["out_features"]),
            int(shape.get("array_size", 16)),
        )
        full_key = (
            *shape_key,
            int(schedule["threads_per_block"]),
            int(schedule["unroll_factor"]),
        )
        raw[full_key].append(float(row["measured_latency_us"]))
        shapes_seen.setdefault(
            shape_key,
            {
                "in_features": shape_key[0],
                "out_features": shape_key[1],
                "array_size": shape_key[2],
                "batch": int(shape.get("batch", 1)),
                "apply_quant": bool(shape.get("apply_quant", True)),
            },
        )

    by_shape: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for (in_f, out_f, arr, tpb, unroll), latencies in raw.items():
        median_us = float(statistics.median(latencies))
        by_shape[(in_f, out_f, arr)].append(
            {
                "shape": dict(shapes_seen[(in_f, out_f, arr)]),
                "schedule": {"threads_per_block": int(tpb), "unroll_factor": int(unroll)},
                "measured_latency_us_median": median_us,
                "replicate_count": int(len(latencies)),
            }
        )
    return by_shape


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    rank = int(round(pct * (len(sorted_vals) - 1)))
    return float(sorted_vals[rank])


def build_selection_report(
    calibration_json: Path = DEFAULT_CALIBRATION_JSON,
) -> Dict[str, Any]:
    calibration = json.loads(Path(calibration_json).read_text(encoding="utf-8"))
    target = load_cost_model_target(str(calibration_json))
    by_shape = _median_by_shape_schedule(calibration)

    per_shape: List[Dict[str, Any]] = []
    regrets: List[float] = []
    confidences: List[float] = []
    margin_pcts: List[float] = []
    top1_hits = 0

    for shape_key, candidates in sorted(by_shape.items()):
        in_features, out_features, array_size = shape_key
        cost_model_shape = {
            "out_features": int(out_features),
            "in_features": int(in_features),
            "batch": int(candidates[0]["shape"]["batch"]),
            "array_size": int(array_size),
            "apply_quant": bool(candidates[0]["shape"]["apply_quant"]),
        }
        schedules = [dict(c["schedule"]) for c in candidates]
        choice = cost_model_select(cost_model_shape, schedules, target=target)

        measured_by_schedule: Dict[Tuple[int, int], Dict[str, Any]] = {
            (int(c["schedule"]["threads_per_block"]), int(c["schedule"]["unroll_factor"])): c
            for c in candidates
        }
        chosen_key = (int(choice.schedule["threads_per_block"]), int(choice.schedule["unroll_factor"]))
        chosen_row = measured_by_schedule[chosen_key]
        oracle_row = min(candidates, key=lambda c: float(c["measured_latency_us_median"]))

        chosen_measured = float(chosen_row["measured_latency_us_median"])
        oracle_measured = float(oracle_row["measured_latency_us_median"])
        oracle_key = (
            int(oracle_row["schedule"]["threads_per_block"]),
            int(oracle_row["schedule"]["unroll_factor"]),
        )
        regret_pct = (chosen_measured - oracle_measured) / max(oracle_measured, 1e-9) * 100.0
        is_top1 = chosen_key == oracle_key
        if is_top1:
            top1_hits += 1

        regrets.append(float(regret_pct))
        confidences.append(float(choice.confidence))
        margin_pcts.append(float(choice.margin_pct))

        per_shape.append(
            {
                "shape": {
                    "in_features": int(in_features),
                    "out_features": int(out_features),
                    "array_size": int(array_size),
                },
                "candidates_considered": int(choice.candidates_considered),
                "chosen_schedule": dict(choice.schedule),
                "predicted_latency_us": float(choice.predicted_latency_us),
                "runner_up_schedule": dict(choice.runner_up_schedule)
                if choice.runner_up_schedule is not None
                else None,
                "runner_up_predicted_latency_us": (
                    float(choice.runner_up_predicted_latency_us)
                    if choice.runner_up_predicted_latency_us is not None
                    else None
                ),
                "margin_pct": float(choice.margin_pct),
                "confidence": float(choice.confidence),
                "chosen_measured_latency_us_median": chosen_measured,
                "oracle_schedule": dict(oracle_row["schedule"]),
                "oracle_measured_latency_us_median": oracle_measured,
                "regret_pct": float(regret_pct),
                "is_top1": bool(is_top1),
            }
        )

    shape_count = len(per_shape)
    top1_accuracy = float(top1_hits / shape_count) if shape_count else 0.0
    summary = {
        "shape_count": int(shape_count),
        "top1_accuracy": top1_accuracy,
        "mean_regret_pct": float(statistics.fmean(regrets)) if regrets else 0.0,
        "median_regret_pct": float(statistics.median(regrets)) if regrets else 0.0,
        "p95_regret_pct": _percentile(regrets, 0.95),
        "max_regret_pct": float(max(regrets)) if regrets else 0.0,
        "within_1pct_fraction": float(sum(r <= 1.0 for r in regrets) / shape_count) if shape_count else 0.0,
        "within_5pct_fraction": float(sum(r <= 5.0 for r in regrets) / shape_count) if shape_count else 0.0,
        "within_10pct_fraction": float(sum(r <= 10.0 for r in regrets) / shape_count) if shape_count else 0.0,
        "mean_confidence": float(statistics.fmean(confidences)) if confidences else 0.0,
        "median_confidence": float(statistics.median(confidences)) if confidences else 0.0,
        "mean_margin_pct": float(statistics.fmean(margin_pcts)) if margin_pcts else 0.0,
    }

    report = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_calibration_json": str(calibration_json),
        "calibration_timestamp_utc": calibration.get("timestamp_utc"),
        "cost_model_coefficients": calibration.get("fitted_coefficients"),
        "methodology": {
            "api": "firmware/host/cost_model.py::select",
            "candidate_set": "every schedule with measurements for the shape (16 schedules per shape on the standard 4x4 grid)",
            "oracle": "min median measured CUDA-event latency per shape over calibration replicates",
            "regret_pct": "(measured(chosen) - measured(oracle)) / measured(oracle) * 100",
            "top1_accuracy": "fraction of shapes where the cost-model choice equals the oracle schedule",
            "scope": "blocked-FC schedule selection on CUDA; no fused-vs-unfused or backend-class decision",
            "notes": "Calibration coefficients are loaded from the same file the runtime uses (cuda_autotuner.load_cost_model_target).",
        },
        "summary": summary,
        "per_shape": per_shape,
    }
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = build_selection_report(calibration_json=args.calibration_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
