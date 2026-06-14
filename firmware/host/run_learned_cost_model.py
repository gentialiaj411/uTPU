"""Phase 1 head-to-head: learned CUDA cost model vs analytical baseline.

This replays the existing calibration measurements in a shape-level
leave-one-shape-out cross-validation loop. Each fold trains a
gradient-boosted tree regressor on the training shapes only, then ranks
the held-out schedules using:

* the learned model,
* the existing analytical model from ``cost_model.py``,
* and two deterministic trivial baselines (mean-latency and
  random-schedule).

The artifact is written to ``bench/results/learned_cost_model_comparison.json``.
It is replay-only: no new CUDA timing is performed here.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from learned_cost_model import (
    DEFAULT_CALIBRATION_JSON,
    FEATURE_NAMES,
    MODEL_HYPERPARAMETERS,
    SKLEARN_AVAILABLE,
    SKLEARN_IMPORT_ERROR,
    SKLEARN_VERSION,
    analytical_predictions,
    evaluate_mean_latency_baseline,
    evaluate_random_schedule_baseline,
    evaluate_shape_predictions,
    fit_learned_model,
    grouped_rows_by_shape,
    load_calibration_rows,
    shape_key,
    predict_rows,
    summarize_fold_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "learned_cost_model_comparison.json"
MODEL_SEED = 1729
RANDOM_BASELINE_SEED = "learned-cost-model-trivial-random-v1"


def _git_sha() -> str:
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _load_analytical_target(calibration_json: Path) -> Dict[str, Any]:
    blob = json.loads(Path(calibration_json).read_text(encoding="utf-8"))
    coeffs = dict(blob.get("fitted_coefficients", {}))
    if not coeffs:
        raise RuntimeError(f"{calibration_json} is missing fitted_coefficients")
    return {"name": "cuda", "cost_model_coefficients": coeffs}


def _fold_record(
    fold_index: int,
    held_out_shape: Tuple[int, int],
    train_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    analytical_target: Dict[str, Any],
) -> Dict[str, Any]:
    learned_model = fit_learned_model(train_rows, random_state=MODEL_SEED)
    # The learned model uses the same feature extraction as the analytical model.
    # Predict explicitly through the shared helper so the fold record stays
    # tied to the replay rows rather than to a cached matrix.
    learned_metrics = evaluate_shape_predictions(
        test_rows,
        predict_rows(learned_model, test_rows),
        prediction_name="learned",
    )
    analytical_metrics = evaluate_shape_predictions(
        test_rows,
        analytical_predictions(test_rows, analytical_target),
        prediction_name="analytical",
    )
    mean_baseline_metrics = evaluate_mean_latency_baseline(test_rows, train_rows)
    random_baseline_metrics = evaluate_random_schedule_baseline(
        test_rows,
        seed=RANDOM_BASELINE_SEED,
        shape=held_out_shape,
    )
    return {
        "fold_index": int(fold_index),
        "held_out_shape": {"in_features": int(held_out_shape[0]), "out_features": int(held_out_shape[1])},
        "n_train_rows": int(len(train_rows)),
        "n_test_rows": int(len(test_rows)),
        "n_schedules": int(learned_metrics["n_schedules"]),
        "models": {
            "learned": {k: v for k, v in learned_metrics.items() if k != "schedule_rows"},
            "analytical": {k: v for k, v in analytical_metrics.items() if k != "schedule_rows"},
            "trivial_baseline": {
                "mean_latency": {k: v for k, v in mean_baseline_metrics.items() if k != "schedule_rows"},
                "random_schedule": {k: v for k, v in random_baseline_metrics.items() if k != "schedule_rows"},
            },
        },
    }


def run(
    calibration_json: Path = DEFAULT_CALIBRATION_JSON,
    output_json: Path = DEFAULT_OUTPUT_JSON,
) -> Dict[str, Any]:
    rows, metadata = load_calibration_rows(calibration_json)
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(f"scikit-learn unavailable: {SKLEARN_IMPORT_ERROR}")

    grouped = grouped_rows_by_shape(rows)
    shape_keys = sorted(grouped)
    if len(shape_keys) < 2:
        raise RuntimeError("need at least two shapes for leave-one-shape-out cross-validation")

    analytical_target = _load_analytical_target(calibration_json)
    fold_records: List[Dict[str, Any]] = []
    learned_folds: List[Dict[str, Any]] = []
    analytical_folds: List[Dict[str, Any]] = []
    trivial_mean_folds: List[Dict[str, Any]] = []
    trivial_random_folds: List[Dict[str, Any]] = []

    for fold_index, held_out_shape in enumerate(shape_keys):
        test_rows = list(grouped[held_out_shape])
        train_rows = [row for key in shape_keys if key != held_out_shape for row in grouped[key]]
        record = _fold_record(fold_index, held_out_shape, train_rows, test_rows, analytical_target)
        fold_records.append(record)
        learned_folds.append(record["models"]["learned"])
        analytical_folds.append(record["models"]["analytical"])
        trivial_mean_folds.append(record["models"]["trivial_baseline"]["mean_latency"])
        trivial_random_folds.append(record["models"]["trivial_baseline"]["random_schedule"])

    learned_summary = summarize_fold_metrics(learned_folds)
    analytical_summary = summarize_fold_metrics(analytical_folds)
    trivial_mean_summary = summarize_fold_metrics(trivial_mean_folds)
    trivial_random_summary = summarize_fold_metrics(trivial_random_folds)

    comparison = {
        "top1_accuracy_pp": float((learned_summary["top1_accuracy"] - analytical_summary["top1_accuracy"]) * 100.0),
        "within_1pct_pp": float((learned_summary["within_1pct_fraction"] - analytical_summary["within_1pct_fraction"]) * 100.0),
        "within_5pct_pp": float((learned_summary["within_5pct_fraction"] - analytical_summary["within_5pct_fraction"]) * 100.0),
        "mean_regret_pp": float(learned_summary["mean_regret_pct"] - analytical_summary["mean_regret_pct"]),
        "mean_spearman_rho_pp": float(learned_summary["mean_spearman_rho"] - analytical_summary["mean_spearman_rho"]),
        "winner_by_top1": (
            "learned"
            if learned_summary["top1_accuracy"] > analytical_summary["top1_accuracy"]
            else "analytical"
            if analytical_summary["top1_accuracy"] > learned_summary["top1_accuracy"]
            else "tie"
        ),
        "winner_by_within_1pct": (
            "learned"
            if learned_summary["within_1pct_fraction"] > analytical_summary["within_1pct_fraction"]
            else "analytical"
            if analytical_summary["within_1pct_fraction"] > learned_summary["within_1pct_fraction"]
            else "tie"
        ),
    }

    total_rows = int(len(rows))
    num_shapes = int(len(shape_keys))
    # The calibration grid is fixed at 16 schedules per shape; compute the
    # same value directly from the first held-out shape instead of assuming.
    first_shape_rows = grouped[shape_keys[0]]
    num_schedules = int(len({(int(r["schedule"]["threads_per_block"]), int(r["schedule"]["unroll_factor"])) for r in first_shape_rows}))

    payload = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "source_calibration_json": str(calibration_json),
        "source_calibration_timestamp_utc": metadata.get("source_timestamp_utc"),
        "methodology": {
            "harness": "firmware/host/run_learned_cost_model.py",
            "split_unit": "shape_used = (in_features, out_features)",
            "split_policy": "leave-one-shape-out cross-validation",
            "claims_scope": (
                "Replay-only evaluation on existing calibration measurements; "
                "CPU-trained gradient-boosted trees; no new CUDA timing; "
                "ranking metrics computed on held-out shapes."
            ),
            "seed": int(MODEL_SEED),
            "trivial_baseline_seed": RANDOM_BASELINE_SEED,
            "feature_source": "firmware/host/calibrate_cost_model.py::_feature_terms() coefficient features",
            "feature_names": list(FEATURE_NAMES),
        },
        "dataset": {
            "num_rows": total_rows,
            "num_shapes": num_shapes,
            "num_schedules_per_shape": num_schedules,
            "fold_count": num_shapes,
            "shape_keys": [
                {"in_features": int(k[0]), "out_features": int(k[1])}
                for k in shape_keys
            ],
        },
        "model": {
            "type": "HistGradientBoostingRegressor",
            "hyperparameters": dict(MODEL_HYPERPARAMETERS),
            "sklearn_available": bool(SKLEARN_AVAILABLE),
            "sklearn_version": SKLEARN_VERSION,
        },
        "summary": {
            "learned": learned_summary,
            "analytical": analytical_summary,
            "trivial_baseline": {
                "mean_latency": trivial_mean_summary,
                "random_schedule": trivial_random_summary,
            },
            "comparison": comparison,
        },
        "per_fold": fold_records,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    if not SKLEARN_AVAILABLE:
        print(f"[learned_cost_model] skipped: scikit-learn unavailable: {SKLEARN_IMPORT_ERROR}")
        return 0

    payload = run(calibration_json=args.calibration_json, output_json=args.output_json)
    learned = payload["summary"]["learned"]
    analytical = payload["summary"]["analytical"]
    trivial_random = payload["summary"]["trivial_baseline"]["random_schedule"]
    print(f"[learned_cost_model] wrote {args.output_json}")
    print(
        "[learned_cost_model] learned: "
        f"top1={learned['top1_accuracy']:.3f} "
        f"within1={learned['within_1pct_fraction']:.3f} "
        f"within5={learned['within_5pct_fraction']:.3f} "
        f"rho={learned['mean_spearman_rho']:.3f}"
    )
    print(
        "[learned_cost_model] analytical: "
        f"top1={analytical['top1_accuracy']:.3f} "
        f"within1={analytical['within_1pct_fraction']:.3f} "
        f"within5={analytical['within_5pct_fraction']:.3f} "
        f"rho={analytical['mean_spearman_rho']:.3f}"
    )
    print(
        "[learned_cost_model] trivial random baseline: "
        f"top1={trivial_random['top1_accuracy']:.3f} "
        f"within1={trivial_random['within_1pct_fraction']:.3f} "
        f"within5={trivial_random['within_5pct_fraction']:.3f} "
        f"rho={trivial_random['mean_spearman_rho']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
