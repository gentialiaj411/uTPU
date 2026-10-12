"""Evaluate a frozen cost-model pruning policy on shape-disjoint held-out data.

This is replay-only: schedule latencies come from the committed RTX 5070
calibration capture.  It never fits on a held-out shape.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from calibrate_cost_model import _fit_coefficients
from cuda_blocked_fc_backend import DEFAULT_CUDA_SCHEDULE_PARAMS
from run_cost_model_heldout import (
    HOLDOUT_FRAC,
    SPLIT_SEED,
    _deterministic_holdout_shapes,
    _load_rows,
    _predict_rows,
    _shape_key,
    _split_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "bench" / "fixtures" / "cost_model_calibration_seed.json"
DEFAULT_OUTPUT = REPO_ROOT / "bench" / "results" / "pruning_policy_heldout.json"
FROZEN_TOP_K = 4
BASELINE_SCHEDULE = dict(DEFAULT_CUDA_SCHEDULE_PARAMS)


def _schedule_key(schedule: Dict[str, Any]) -> Tuple[int, int]:
    return (int(schedule["threads_per_block"]), int(schedule["unroll_factor"]))


def _median_schedule_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_schedule_key(row["schedule"]), []).append(row)
    out = []
    for key, bucket in grouped.items():
        out.append({
            "schedule": {"threads_per_block": key[0], "unroll_factor": key[1]},
            "predicted_latency_us": float(statistics.median(r["predicted_latency_us"] for r in bucket)),
            "measured_latency_us": float(statistics.median(r["measured_latency_us"] for r in bucket)),
        })
    return sorted(out, key=lambda r: (r["predicted_latency_us"], _schedule_key(r["schedule"])))


def run(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> Dict[str, Any]:
    rows, source_meta = _load_rows(source)
    holdout_shapes = _deterministic_holdout_shapes([_shape_key(r) for r in rows], HOLDOUT_FRAC, SPLIT_SEED)
    train_rows, test_rows = _split_rows(rows, holdout_shapes)
    train_shapes = {_shape_key(r) for r in train_rows}
    test_shapes = {_shape_key(r) for r in test_rows}
    if not train_rows or not test_rows or train_shapes & test_shapes:
        raise RuntimeError("train/test split must be nonempty and shape-disjoint")

    fit = _fit_coefficients(train_rows)
    predicted_test = _predict_rows(test_rows, fit["coefficients"])
    rows_by_shape: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in predicted_test:
        rows_by_shape.setdefault(_shape_key(row), []).append(row)

    per_shape = []
    regrets = []
    for shape, shape_rows in sorted(rows_by_shape.items()):
        ranked = _median_schedule_rows(shape_rows)
        retained = ranked[:FROZEN_TOP_K]
        exhaustive_best = min(ranked, key=lambda r: (r["measured_latency_us"], _schedule_key(r["schedule"])))
        retained_best = min(retained, key=lambda r: (r["measured_latency_us"], _schedule_key(r["schedule"])))
        retained_keys = {_schedule_key(r["schedule"]) for r in retained}
        baseline_in_retained = _schedule_key(BASELINE_SCHEDULE) in retained_keys
        regret = 100.0 * (retained_best["measured_latency_us"] - exhaustive_best["measured_latency_us"]) / max(exhaustive_best["measured_latency_us"], 1e-9)
        regrets.append(float(regret))
        per_shape.append({
            "shape": {"in_features": shape[0], "out_features": shape[1]},
            "full_menu_candidate_count": len(ranked),
            "retained_candidate_count": len(retained),
            "live_tuner_baseline_schedule": BASELINE_SCHEDULE,
            "baseline_is_also_retained": baseline_in_retained,
            "live_measurement_count_including_baseline": len(retained) + 1,
            "retained_schedules": [r["schedule"] for r in retained],
            "best_in_menu_schedule": exhaustive_best["schedule"],
            "best_in_menu_retained": _schedule_key(exhaustive_best["schedule"]) in retained_keys,
            "best_in_menu_latency_us": exhaustive_best["measured_latency_us"],
            "retained_best_schedule": retained_best["schedule"],
            "retained_best_latency_us": retained_best["measured_latency_us"],
            "regret_pct": float(regret),
        })

    payload = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "source": source_meta["source"],
            "split": "existing deterministic shape-heldout split from run_cost_model_heldout.py",
            "fit": "production fitter trained on train shapes only",
            "frozen_policy": {"top_k": FROZEN_TOP_K, "selection": "lowest train-fit predicted latency"},
            "candidate_count_note": "live_tuner baseline measurement is counted separately, even when its schedule is retained",
            "capture_command_for_expansion": "python firmware/host/calibrate_cost_model.py --output-json build/reports/cost_model_calibration.json --output-md build/reports/cost_model_calibration.md",
        },
        "split": {
            "train_shapes": [list(s) for s in sorted(train_shapes)],
            "test_shapes": [list(s) for s in sorted(test_shapes)],
            "shape_sets_disjoint": True,
        },
        "fit": {"coefficients_train_only": fit["coefficients"]},
        "per_heldout_shape": per_shape,
        "aggregate": {
            "heldout_shape_count": len(per_shape),
            "fraction_best_in_menu_retained": sum(r["best_in_menu_retained"] for r in per_shape) / len(per_shape),
            "fraction_within_1pct": sum(r <= 1.0 for r in regrets) / len(regrets),
            "max_regret_pct": max(regrets),
            "mean_live_measurement_count_including_baseline": statistics.fmean(r["live_measurement_count_including_baseline"] for r in per_shape),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(json.dumps(result["aggregate"], indent=2))
