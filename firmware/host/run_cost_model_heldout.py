"""Phase 7 generalization replay: cost-model held-out test on unseen shapes.

This script extends the existing
``build/reports/cost_model_holdout_validation.json`` (which already
records latency-prediction generalization) by adding a *headline-track*
artifact at ``bench/results/cost_model_heldout.json`` that:

* re-runs a deterministic 80/20 split of the calibration measurements
  partitioned by **(in_features, out_features) layer shape** (the unit a
  user actually sees as "a layer the model never saw"),
* refits the analytical cost-model coefficients on TRAIN rows only via
  the production fitter (``calibrate_cost_model._fit_coefficients``),
* evaluates **latency prediction quality** on the held-out TEST rows
  (log-R^2 / MAPE / p95 abs-rel error),
* evaluates **selection quality** on held-out shapes by predicting the
  best schedule for each unseen shape and comparing its
  *measured-actual* latency to the *measured-best* schedule's latency
  (top-1, mean / p95 regret),
* and writes a stable, sorted-key JSON downstream tooling can lock.

Reproducible without CUDA: the calibration measurements were captured
once on the WSL2 + RTX 5070 Laptop host (see
``build/reports/cost_model_calibration.json::environment``); this replay
runs offline against that JSON, so the artifact regenerates on a
CPU-only laptop in seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from calibrate_cost_model import _fit_coefficients, _metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "build" / "reports" / "cost_model_calibration.json"
DEFAULT_OUTPUT = REPO_ROOT / "bench" / "results" / "cost_model_heldout.json"

HOLDOUT_FRAC = 0.20
SPLIT_SEED = "phase7-heldout-v1"


# ---------------------------------------------------------------------------
# Data loading + split
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _load_rows(source: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    blob = json.loads(source.read_text(encoding="utf-8"))
    rows = list(blob.get("per_point", []))
    if not rows:
        raise RuntimeError(f"{source} has no per_point rows")
    metadata = {
        "source": str(source.relative_to(REPO_ROOT)) if source.is_absolute() else str(source),
        "source_git_sha": blob.get("git_sha"),
        "source_timestamp_utc": blob.get("timestamp_utc"),
        "source_environment": blob.get("environment"),
        "source_machine_info": blob.get("machine_info"),
        "source_n_rows": len(rows),
    }
    return rows, metadata


def _shape_key(row: Dict[str, Any]) -> Tuple[int, int]:
    s = row["shape_used"]
    return (int(s["in_features"]), int(s["out_features"]))


def _deterministic_holdout_shapes(
    shape_keys: List[Tuple[int, int]],
    holdout_frac: float,
    seed: str,
) -> List[Tuple[int, int]]:
    """Pick the held-out shapes by stable SHA-256 hash so re-runs match.

    Hash space is the seeded string ``f"{seed}|{in_features}|{out_features}"``
    interpreted as a hex int; lowest ``ceil(N * holdout_frac)`` shapes
    by hash become the test set. This avoids a Python-version-dependent
    ``random`` state without sacrificing determinism.
    """
    sorted_keys = sorted(set(shape_keys))
    target_count = max(1, int(math.ceil(len(sorted_keys) * holdout_frac)))

    def _hash(key: Tuple[int, int]) -> int:
        token = f"{seed}|{key[0]}|{key[1]}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")

    sorted_keys.sort(key=lambda k: (_hash(k), k))
    return sorted_keys[:target_count]


def _split_rows(
    rows: List[Dict[str, Any]],
    holdout_keys: List[Tuple[int, int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    holdout_set = set(holdout_keys)
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for row in rows:
        if _shape_key(row) in holdout_set:
            test.append(row)
        else:
            train.append(row)
    return train, test


# ---------------------------------------------------------------------------
# Prediction with refit coefficients
# ---------------------------------------------------------------------------


_FEATURE_COEFF_MAP: List[Tuple[str, str, int]] = [
    ("memory_us_per_kib", "memory_kib", +1),
    ("cta_memory_us_per_kib", "cta_memory_kib", +1),
    ("underoccupancy_penalty_us", "underoccupancy_penalty", +1),
    ("tile_tail_penalty_us", "tail_ratio", +1),
    ("unroll_gain_us", "unroll_norm", -1),
    ("unroll_k_tail_penalty_us", "unroll_k_tail", +1),
    ("unroll_shape_interaction_us", "unroll_shape_interaction", -1),
    ("small_out_tpb_interaction_us", "small_out_tpb_interaction", +1),
    ("small_out_unroll_interaction_us", "small_out_unroll_interaction", +1),
    ("idle_thread_ratio_us", "idle_thread_ratio", +1),
    ("wave_tpb_interaction_us", "wave_tpb_interaction", +1),
    ("small_out_idle_penalty_us", "small_out_idle_penalty", +1),
    ("large_k_unroll_gain_us", "large_k_unroll_gain", -1),
    ("small_out_unroll_penalty_us", "small_out_unroll_penalty", +1),
    ("large_out_small_k_wave_tpb_efficiency_us", "large_out_small_k_wave_tpb_efficiency", -1),
]


def _predict_latency_us(features: Dict[str, float], coeffs: Dict[str, float]) -> float:
    """Predict latency for a single row using the refit coefficients.

    Mirrors the closed-form prediction in
    ``calibrate_cost_model._fit_coefficients::_predict_latency`` so the
    held-out test exercises exactly the production model form.
    """
    eps = 1e-9
    lat = float(coeffs.get("intercept_us", 0.0))
    for coeff_name, feat_name, sign in _FEATURE_COEFF_MAP:
        c = float(coeffs.get(coeff_name, 0.0))
        f = float(features.get(feat_name, 0.0))
        lat += sign * c * f
    return max(lat, eps)


def _predict_rows(
    rows: List[Dict[str, Any]],
    coeffs: Dict[str, float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy["predicted_latency_us"] = float(
            _predict_latency_us(row["features"], coeffs)
        )
        out.append(copy)
    return out


# ---------------------------------------------------------------------------
# Selection quality on held-out shapes
# ---------------------------------------------------------------------------


def _selection_metrics(
    test_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Top-1 / regret on held-out shapes.

    For each held-out shape, group rows by schedule, take the median
    measured latency per (shape, schedule), then compare:

    * *measured-best schedule*: argmin over schedules of measured median.
    * *predicted-best schedule*: argmin over schedules of predicted median.

    The regret metric is computed using the **measured** latency of the
    predicted-best schedule, NOT the predicted latency: this is what a
    user actually pays at runtime if they trust the model.
    """
    by_shape: Dict[Tuple[int, int], Dict[Tuple[int, int], List[Dict[str, Any]]]] = {}
    for row in test_rows:
        sk = _shape_key(row)
        sch = row["schedule"]
        ck = (int(sch["threads_per_block"]), int(sch["unroll_factor"]))
        by_shape.setdefault(sk, {}).setdefault(ck, []).append(row)

    per_shape_records: List[Dict[str, Any]] = []
    top1_hits = 0
    regrets: List[float] = []
    n_evaluated = 0
    for shape_key, schedules in sorted(by_shape.items()):
        if not schedules:
            continue
        schedule_rows: List[Dict[str, Any]] = []
        for ck, rows in schedules.items():
            measured_median = float(
                statistics.median(float(r["measured_latency_us"]) for r in rows)
            )
            predicted_median = float(
                statistics.median(float(r["predicted_latency_us"]) for r in rows)
            )
            schedule_rows.append(
                {
                    "schedule": {
                        "threads_per_block": int(ck[0]),
                        "unroll_factor": int(ck[1]),
                    },
                    "measured_median_us": measured_median,
                    "predicted_median_us": predicted_median,
                }
            )
        if len(schedule_rows) < 2:
            continue
        schedule_rows.sort(
            key=lambda r: (r["schedule"]["threads_per_block"], r["schedule"]["unroll_factor"])
        )
        measured_best = min(schedule_rows, key=lambda r: r["measured_median_us"])
        predicted_best = min(schedule_rows, key=lambda r: r["predicted_median_us"])
        same = (
            predicted_best["schedule"]["threads_per_block"]
            == measured_best["schedule"]["threads_per_block"]
            and predicted_best["schedule"]["unroll_factor"]
            == measured_best["schedule"]["unroll_factor"]
        )
        regret_pct = (
            (predicted_best["measured_median_us"] - measured_best["measured_median_us"])
            / max(measured_best["measured_median_us"], 1e-9)
            * 100.0
        )
        per_shape_records.append(
            {
                "shape": {"in_features": shape_key[0], "out_features": shape_key[1]},
                "n_schedules": len(schedule_rows),
                "measured_best": measured_best,
                "predicted_best": predicted_best,
                "top1_hit": bool(same),
                "regret_pct": float(regret_pct),
            }
        )
        if same:
            top1_hits += 1
        regrets.append(float(regret_pct))
        n_evaluated += 1

    if n_evaluated == 0:
        summary: Dict[str, Any] = {
            "n_held_out_shapes_with_multi_schedule": 0,
            "top1_accuracy": None,
            "mean_regret_pct": None,
            "median_regret_pct": None,
            "p95_regret_pct": None,
            "max_regret_pct": None,
            "within_5pct_fraction": None,
            "within_10pct_fraction": None,
        }
    else:
        summary = {
            "n_held_out_shapes_with_multi_schedule": int(n_evaluated),
            "top1_accuracy": float(top1_hits / n_evaluated),
            "mean_regret_pct": float(statistics.fmean(regrets)),
            "median_regret_pct": float(statistics.median(regrets)),
            "p95_regret_pct": float(np.percentile(regrets, 95.0)) if regrets else 0.0,
            "max_regret_pct": float(max(regrets)),
            "within_5pct_fraction": float(sum(1 for r in regrets if r <= 5.0) / n_evaluated),
            "within_10pct_fraction": float(sum(1 for r in regrets if r <= 10.0) / n_evaluated),
        }
    return {"summary": summary, "per_shape": per_shape_records}


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    holdout_frac: float = HOLDOUT_FRAC,
    seed: str = SPLIT_SEED,
) -> Dict[str, Any]:
    rows, metadata = _load_rows(source)
    shape_keys = [_shape_key(r) for r in rows]
    holdout_keys = _deterministic_holdout_shapes(shape_keys, holdout_frac, seed)
    train_rows, test_rows = _split_rows(rows, holdout_keys)
    if not train_rows or not test_rows:
        raise RuntimeError(
            "split produced empty train or test set; "
            f"check holdout_frac={holdout_frac} and source data"
        )

    fit = _fit_coefficients(train_rows)
    coeffs = fit["coefficients"]

    train_pred = _predict_rows(train_rows, coeffs)
    test_pred = _predict_rows(test_rows, coeffs)
    train_metrics = _metrics(train_pred)
    test_metrics = _metrics(test_pred)

    selection = _selection_metrics(test_pred)

    payload = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "phase": "phase7_generalization_replay",
        "methodology": {
            "harness": "firmware/host/run_cost_model_heldout.py",
            "fit_function": "calibrate_cost_model._fit_coefficients (production fitter)",
            "split_unit": "shape_used = (in_features, out_features)",
            "holdout_fraction": float(holdout_frac),
            "split_seed": str(seed),
            "split_policy": (
                "deterministic SHA-256 hash of "
                "f'{seed}|{in_features}|{out_features}'; lowest "
                "ceil(N * holdout_frac) shapes by hash become test set."
            ),
            "selection_metric_definition": (
                "For each held-out shape with >= 2 distinct schedules, "
                "predicted-best schedule is argmin over schedules of "
                "predicted median latency (refit on TRAIN only). Regret "
                "uses the *measured* latency of the predicted-best "
                "schedule vs the measured-best schedule; this is what a "
                "user pays at runtime. Top-1 == 1.0 means the cost model "
                "picked the same schedule the autotuner would have."
            ),
            "claims_scope": (
                "Generalization to layer shapes the cost model was not "
                "fitted on. Sim-only / replay-only artifact: no live "
                "CUDA timing here; uses measurements captured once on "
                "the WSL2 + RTX 5070 host that produced the source "
                "calibration JSON. Re-running this script on a CPU-only "
                "laptop produces the same numbers byte-for-byte."
            ),
        },
        "source": metadata,
        "split": {
            "n_train_rows": len(train_rows),
            "n_test_rows": len(test_rows),
            "n_unique_train_shapes": len({_shape_key(r) for r in train_rows}),
            "n_unique_test_shapes": len({_shape_key(r) for r in test_rows}),
            "holdout_shapes": [
                {"in_features": int(k[0]), "out_features": int(k[1])}
                for k in holdout_keys
            ],
        },
        "fit": {
            "coefficients_train_only": coeffs,
            "fit_status": fit.get("fit_status"),
            "fit_objective": fit.get("fit_objective"),
            "model_form": fit.get("model_form"),
        },
        "latency_prediction": {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_over_train_ratios": {
                "log_r2": (
                    float(test_metrics["log_r2"] / train_metrics["log_r2"])
                    if train_metrics["log_r2"] not in (0.0, None)
                    else None
                ),
                "mape": (
                    float(test_metrics["mape_pct"] / train_metrics["mape_pct"])
                    if train_metrics["mape_pct"] not in (0.0, None)
                    else None
                ),
                "p95_abs_rel_error": (
                    float(test_metrics["p95_abs_rel_error_pct"]
                          / train_metrics["p95_abs_rel_error_pct"])
                    if train_metrics["p95_abs_rel_error_pct"] not in (0.0, None)
                    else None
                ),
            },
        },
        "selection_quality": selection,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    parser.add_argument("--seed", type=str, default=SPLIT_SEED)
    args = parser.parse_args()

    payload = run(
        source=Path(args.source),
        output=Path(args.output),
        holdout_frac=float(args.holdout_frac),
        seed=str(args.seed),
    )
    test = payload["latency_prediction"]["test_metrics"]
    sel = payload["selection_quality"]["summary"]
    print(f"[cost_model_heldout] wrote {args.output}")
    print(
        f"[cost_model_heldout] test latency: "
        f"log_R^2={test['log_r2']:.4f} MAPE={test['mape_pct']:.2f}% "
        f"p95={test['p95_abs_rel_error_pct']:.2f}%"
    )
    if sel["n_held_out_shapes_with_multi_schedule"]:
        print(
            f"[cost_model_heldout] selection on held-out shapes: "
            f"top-1={sel['top1_accuracy']:.3f} "
            f"mean_regret={sel['mean_regret_pct']:.2f}% "
            f"max_regret={sel['max_regret_pct']:.2f}% "
            f"within-5%={sel['within_5pct_fraction']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
