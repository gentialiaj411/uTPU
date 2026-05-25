"""Frozen CUDA cost-model regression set and replay helpers.

The 24 layer shapes are the unique ``(in_features, out_features)`` pairs produced
by ``calibrate_cost_model._shape_grid()`` (fc1/fc2 layers across the calibration
grid). They are the same shapes evaluated by ``evaluate_pruned_autotuner.py``.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from cost_model import predict_latency_us
from cuda_autotuner import CUDATuningSearchSpace, load_cost_model_target, rank_candidates_by_cost_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_JSON = REPO_ROOT / "build" / "reports" / "cost_model_calibration.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "cost_model_regression.json"

# Frozen regression shapes: unique layer (in, out) pairs from calibrate_cost_model._shape_grid().
REGRESSION_SHAPES: Tuple[Tuple[int, int], ...] = (
    (16, 16),
    (16, 64),
    (16, 256),
    (16, 512),
    (32, 16),
    (32, 64),
    (32, 256),
    (32, 512),
    (64, 16),
    (64, 64),
    (64, 256),
    (64, 512),
    (128, 16),
    (128, 64),
    (128, 256),
    (128, 512),
    (256, 16),
    (256, 64),
    (256, 256),
    (256, 512),
    (512, 16),
    (512, 64),
    (512, 256),
    (512, 512),
)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _schedule_key(schedule: Dict[str, int]) -> Tuple[int, int]:
    return (int(schedule["threads_per_block"]), int(schedule["unroll_factor"]))


def _abs_percent_error(predicted_us: float, measured_us: float) -> float:
    return abs(float(predicted_us) - float(measured_us)) / max(float(measured_us), 1e-9) * 100.0


def _median_shape_schedule_rows(calibration: Dict[str, Any]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in calibration["per_point"]:
        shape = row["shape_used"]
        schedule = row["schedule"]
        key = (
            int(shape["in_features"]),
            int(shape["out_features"]),
            int(schedule["threads_per_block"]),
            int(schedule["unroll_factor"]),
        )
        grouped[key].append(row)

    by_shape: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for (in_features, out_features, threads, unroll), rows in grouped.items():
        measured = statistics.median(float(r["measured_latency_us"]) for r in rows)
        by_shape[(in_features, out_features)].append(
            {
                "schedule": {
                    "threads_per_block": int(threads),
                    "unroll_factor": int(unroll),
                },
                "measured_latency_us": float(measured),
                "replicate_count": int(len(rows)),
            }
        )
    return by_shape


def compute_replay_metrics(
    per_shape_rows: Sequence[Dict[str, Any]],
    coefficients: Dict[str, float],
) -> Dict[str, float]:
    target = {"name": "cuda", "cost_model_coefficients": dict(coefficients)}
    per_shape_medians: List[float] = []
    per_shape_p95: List[float] = []
    all_abs_errors: List[float] = []

    for row in per_shape_rows:
        shape = {
            "out_features": int(row["out_features"]),
            "in_features": int(row["in_features"]),
            "batch": 1,
            "array_size": int(row.get("array_size", 16)),
            "apply_quant": True,
        }
        abs_errors: List[float] = []
        for sched_row in row["schedules"]:
            schedule = sched_row["schedule"]
            predicted = predict_latency_us(shape, schedule, target=target)
            measured = float(sched_row["measured_latency_us"])
            err = _abs_percent_error(predicted, measured)
            abs_errors.append(err)
            all_abs_errors.append(err)
        per_shape_medians.append(float(statistics.median(abs_errors)))
        if abs_errors:
            per_shape_p95.append(float(sorted(abs_errors)[int(round(0.95 * (len(abs_errors) - 1)))]))

    medians_sorted = sorted(per_shape_medians)
    p95_idx = int(round(0.95 * (len(medians_sorted) - 1))) if medians_sorted else 0
    return {
        "median_abs_percent_error": float(statistics.median(per_shape_medians)) if per_shape_medians else 0.0,
        "mean_abs_percent_error": float(statistics.fmean(all_abs_errors)) if all_abs_errors else 0.0,
        "p95_abs_percent_error": float(medians_sorted[p95_idx]) if medians_sorted else 0.0,
        "max_abs_percent_error": float(max(per_shape_medians)) if per_shape_medians else 0.0,
        "per_shape_median_p95": float(statistics.fmean(per_shape_p95)) if per_shape_p95 else 0.0,
    }


def build_regression_report(
    calibration_json: Path = DEFAULT_CALIBRATION_JSON,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    coefficients: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    calibration = json.loads(Path(calibration_json).read_text(encoding="utf-8"))
    target = load_cost_model_target(str(calibration_json))
    coeffs = dict(coefficients or target["cost_model_coefficients"])
    by_shape = _median_shape_schedule_rows(calibration)
    missing = [shape for shape in REGRESSION_SHAPES if shape not in by_shape]
    if missing:
        raise ValueError(f"calibration artifact missing regression shapes: {missing}")

    search_space = CUDATuningSearchSpace()
    full_candidate_count = len(search_space.candidates())
    per_shape: List[Dict[str, Any]] = []

    for in_features, out_features in REGRESSION_SHAPES:
        candidates = by_shape[(in_features, out_features)]
        if len(candidates) != full_candidate_count:
            raise ValueError(
                f"shape ({in_features},{out_features}) has {len(candidates)} schedules, expected {full_candidate_count}"
            )
        shape = {
            "out_features": int(out_features),
            "in_features": int(in_features),
            "batch": 1,
            "array_size": 16,
            "apply_quant": True,
        }
        ranked = rank_candidates_by_cost_model(
            out_features=out_features,
            in_features=in_features,
            array_size=16,
            candidates=[c["schedule"] for c in candidates],
            target={"name": "cuda", "cost_model_coefficients": coeffs},
        )
        rank_by_schedule = {
            _schedule_key(item["schedule"]): int(item["cost_model_rank"]) for item in ranked
        }
        schedule_rows: List[Dict[str, Any]] = []
        abs_errors: List[float] = []
        for candidate in candidates:
            schedule = candidate["schedule"]
            measured = float(candidate["measured_latency_us"])
            predicted = predict_latency_us(shape, schedule, target={"name": "cuda", "cost_model_coefficients": coeffs})
            abs_err = _abs_percent_error(predicted, measured)
            abs_errors.append(abs_err)
            schedule_rows.append(
                {
                    "schedule": dict(schedule),
                    "measured_latency_us": measured,
                    "baseline_predicted_latency_us": float(predicted),
                    "baseline_abs_percent_error": float(abs_err),
                    "cost_model_rank": int(rank_by_schedule[_schedule_key(schedule)]),
                }
            )
        exhaustive_best = min(schedule_rows, key=lambda item: float(item["measured_latency_us"]))
        per_shape.append(
            {
                "in_features": int(in_features),
                "out_features": int(out_features),
                "array_size": 16,
                "median_abs_percent_error": float(statistics.median(abs_errors)),
                "exhaustive_best_schedule": dict(exhaustive_best["schedule"]),
                "exhaustive_best_measured_us": float(exhaustive_best["measured_latency_us"]),
                "exhaustive_best_baseline_predicted_us": float(exhaustive_best["baseline_predicted_latency_us"]),
                "exhaustive_best_abs_percent_error": float(exhaustive_best["baseline_abs_percent_error"]),
                "cost_model_rank_of_exhaustive_best": int(exhaustive_best["cost_model_rank"]),
                "schedules": schedule_rows,
            }
        )

    per_shape_medians = [float(row["median_abs_percent_error"]) for row in per_shape]
    medians_sorted = sorted(per_shape_medians)
    baseline_median = float(statistics.median(per_shape_medians))
    baseline_p95 = float(medians_sorted[int(round(0.95 * (len(medians_sorted) - 1)))])
    tolerance = float(baseline_p95 - baseline_median)

    report = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "source_calibration_json": str(calibration_json.relative_to(REPO_ROOT)).replace("\\", "/"),
        "calibration_timestamp_utc": calibration.get("timestamp_utc"),
        "shape_source": "firmware/host/calibrate_cost_model.py::_shape_grid() unique layer (in,out) pairs",
        "autotuner_reference": "firmware/host/evaluate_pruned_autotuner.py",
        "regression_shape_count": len(REGRESSION_SHAPES),
        "regression_shapes": [{"in_features": i, "out_features": o} for i, o in REGRESSION_SHAPES],
        "schedule_candidate_count": int(full_candidate_count),
        "baseline_coefficients": coeffs,
        "baseline": {
            "median_abs_percent_error": baseline_median,
            "mean_abs_percent_error": float(statistics.fmean(per_shape_medians)),
            "p95_abs_percent_error": baseline_p95,
            "max_abs_percent_error": float(max(per_shape_medians)),
            "regression_tolerance_pct_points": tolerance,
            "regression_ceiling_abs_percent_error": float(baseline_median + tolerance),
        },
        "methodology": {
            "mode": "measured-data replay",
            "notes": (
                "Each shape uses median measured CUDA-event latency per schedule from the calibration artifact. "
                "Replay recomputes predictions with current cost_model.py and frozen baseline coefficients."
            ),
            "abs_percent_error": "abs(predicted - measured) / measured * 100",
            "per_shape_aggregate": "median abs_percent_error across 16 schedule candidates",
            "regression_gate": "fail if replay median_abs_percent_error exceeds baseline median + tolerance_pct_points",
        },
        "per_shape": per_shape,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def replay_regression_report(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = compute_replay_metrics(report["per_shape"], report["baseline_coefficients"])
    baseline = report["baseline"]
    ceiling = float(baseline["regression_ceiling_abs_percent_error"])
    replay_median = float(metrics["median_abs_percent_error"])
    return {
        "replay_metrics": metrics,
        "baseline_median_abs_percent_error": float(baseline["median_abs_percent_error"]),
        "regression_tolerance_pct_points": float(baseline["regression_tolerance_pct_points"]),
        "regression_ceiling_abs_percent_error": ceiling,
        "within_envelope": bool(replay_median <= ceiling + 1e-9),
        "delta_from_baseline_median": float(replay_median - float(baseline["median_abs_percent_error"])),
    }
