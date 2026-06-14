from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from calibrate_cost_model import _feature_terms  # noqa: E402

try:  # pragma: no cover - import gate is exercised by the script/test path
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn import __version__ as SKLEARN_VERSION

    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - clean skip path
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    SKLEARN_VERSION = None
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc

from cost_model import predict_latency_us  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_JSON = REPO_ROOT / "build" / "reports" / "cost_model_calibration.json"

# Keep the learned model apples-to-apples with the analytical baseline:
# train only on the 15 coefficient-driving features used by cost_model.py.
FEATURE_NAMES: Tuple[str, ...] = (
    "memory_kib",
    "cta_memory_kib",
    "underoccupancy_penalty",
    "tail_ratio",
    "unroll_norm",
    "unroll_k_tail",
    "unroll_shape_interaction",
    "small_out_tpb_interaction",
    "small_out_unroll_interaction",
    "idle_thread_ratio",
    "wave_tpb_interaction",
    "small_out_idle_penalty",
    "large_k_unroll_gain",
    "small_out_unroll_penalty",
    "large_out_small_k_wave_tpb_efficiency",
)

MODEL_HYPERPARAMETERS: Dict[str, Any] = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_depth": 3,
    "min_samples_leaf": 2,
    "l2_regularization": 0.0,
    "early_stopping": False,
    "random_state": 1729,
}


def load_calibration_rows(calibration_json: Path = DEFAULT_CALIBRATION_JSON) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    blob = json.loads(Path(calibration_json).read_text(encoding="utf-8"))
    rows = list(blob.get("per_point", []))
    if not rows:
        raise RuntimeError(f"{calibration_json} has no per_point rows")
    metadata = {
        "source_calibration_json": str(calibration_json),
        "source_git_sha": blob.get("git_sha"),
        "source_timestamp_utc": blob.get("timestamp_utc"),
        "source_environment": blob.get("environment"),
        "source_machine_info": blob.get("machine_info"),
        "source_n_rows": int(len(rows)),
    }
    return rows, metadata


def shape_key(row: Dict[str, Any]) -> Tuple[int, int]:
    shape = row["shape_used"]
    return (int(shape["in_features"]), int(shape["out_features"]))


def schedule_key(row_or_schedule: Dict[str, Any]) -> Tuple[int, int]:
    schedule = row_or_schedule["schedule"] if "schedule" in row_or_schedule else row_or_schedule
    return (int(schedule["threads_per_block"]), int(schedule["unroll_factor"]))


def grouped_rows_by_shape(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(shape_key(row), []).append(row)
    return grouped


def feature_vector(shape_used: Dict[str, Any], schedule: Dict[str, Any]) -> List[float]:
    terms = _feature_terms(shape_used, schedule)
    return [float(terms[name]) for name in FEATURE_NAMES]


def build_feature_matrix(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    matrix = np.asarray(
        [feature_vector(row["shape_used"], row["schedule"]) for row in rows],
        dtype=np.float64,
    )
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def build_target_vector(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["measured_latency_us"]) for row in rows], dtype=np.float64)


def make_regressor(random_state: int = 1729) -> HistGradientBoostingRegressor:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(f"scikit-learn unavailable: {SKLEARN_IMPORT_ERROR}")
    params = dict(MODEL_HYPERPARAMETERS)
    params["random_state"] = int(random_state)
    return HistGradientBoostingRegressor(**params)


def fit_learned_model(rows: Sequence[Dict[str, Any]], random_state: int = 1729) -> HistGradientBoostingRegressor:
    model = make_regressor(random_state=random_state)
    X = build_feature_matrix(rows)
    y = build_target_vector(rows)
    model.fit(X, y)
    return model


def predict_rows(model: Any, rows: Sequence[Dict[str, Any]]) -> List[float]:
    if not rows:
        return []
    X = build_feature_matrix(rows)
    preds = model.predict(X)
    return [float(max(float(v), 1e-9)) for v in preds]


def analytical_predictions(rows: Sequence[Dict[str, Any]], target: Any) -> List[float]:
    return [float(predict_latency_us(row["shape_used"], row["schedule"], target=target)) for row in rows]


def _stable_random_score(seed: str, shape: Tuple[int, int], schedule: Tuple[int, int]) -> float:
    token = f"{seed}|{shape[0]}|{shape[1]}|{schedule[0]}|{schedule[1]}".encode("utf-8")
    digest = hashlib.sha256(token).digest()[:8]
    return int.from_bytes(digest, "big") / float(2**64)


def _spearman_rho(measured: Sequence[float], predicted: Sequence[float]) -> float:
    measured_arr = np.asarray(measured, dtype=np.float64)
    predicted_arr = np.asarray(predicted, dtype=np.float64)
    if measured_arr.size <= 1:
        return 0.0
    rank_m = np.argsort(np.argsort(measured_arr)).astype(np.float64)
    rank_p = np.argsort(np.argsort(predicted_arr)).astype(np.float64)
    rank_m -= float(np.mean(rank_m))
    rank_p -= float(np.mean(rank_p))
    denom = float(np.linalg.norm(rank_m) * np.linalg.norm(rank_p))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(rank_m, rank_p) / denom)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _shape_schedule_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(schedule_key(row), []).append(row)
    schedule_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        items = grouped[key]
        measured = [float(r["measured_latency_us"]) for r in items]
        schedule_rows.append(
            {
                "schedule": {
                    "threads_per_block": int(key[0]),
                    "unroll_factor": int(key[1]),
                },
                "measured_median_us": float(statistics.median(measured)),
                "replicate_count": int(len(items)),
                "rows": items,
            }
        )
    return schedule_rows


def evaluate_shape_predictions(
    rows: Sequence[Dict[str, Any]],
    predicted_latencies: Sequence[float],
    *,
    prediction_name: str,
) -> Dict[str, Any]:
    if len(rows) != len(predicted_latencies):
        raise ValueError("rows/predicted_latencies length mismatch")
    by_schedule: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row, pred in zip(rows, predicted_latencies):
        key = schedule_key(row)
        entry = by_schedule.setdefault(
            key,
            {
                "schedule": {
                    "threads_per_block": int(key[0]),
                    "unroll_factor": int(key[1]),
                },
                "measured_values": [],
                "predicted_values": [],
            },
        )
        entry["measured_values"].append(float(row["measured_latency_us"]))
        entry["predicted_values"].append(float(pred))

    schedule_rows: List[Dict[str, Any]] = []
    for key in sorted(by_schedule):
        entry = by_schedule[key]
        schedule_rows.append(
            {
                "schedule": dict(entry["schedule"]),
                "measured_median_us": float(statistics.median(entry["measured_values"])),
                "predicted_median_us": float(statistics.median(entry["predicted_values"])),
            }
        )

    if not schedule_rows:
        return {
            "prediction_name": prediction_name,
            "n_schedules": 0,
            "top1_hit": False,
            "within_1pct": False,
            "within_5pct": False,
            "regret_pct": 0.0,
            "spearman_rho": 0.0,
            "chosen_schedule": None,
            "oracle_schedule": None,
            "chosen_measured_latency_us_median": None,
            "oracle_measured_latency_us_median": None,
        }

    measured_best = min(
        schedule_rows,
        key=lambda item: (
            item["measured_median_us"],
            item["schedule"]["threads_per_block"],
            item["schedule"]["unroll_factor"],
        ),
    )
    predicted_best = min(
        schedule_rows,
        key=lambda item: (
            item["predicted_median_us"],
            item["schedule"]["threads_per_block"],
            item["schedule"]["unroll_factor"],
        ),
    )
    regret_pct = (
        (float(predicted_best["measured_median_us"]) - float(measured_best["measured_median_us"]))
        / max(float(measured_best["measured_median_us"]), 1e-9)
        * 100.0
    )
    measured_values = [float(item["measured_median_us"]) for item in schedule_rows]
    predicted_values = [float(item["predicted_median_us"]) for item in schedule_rows]
    top1_hit = (
        predicted_best["schedule"]["threads_per_block"] == measured_best["schedule"]["threads_per_block"]
        and predicted_best["schedule"]["unroll_factor"] == measured_best["schedule"]["unroll_factor"]
    )
    return {
        "prediction_name": prediction_name,
        "n_schedules": int(len(schedule_rows)),
        "top1_hit": bool(top1_hit),
        "within_1pct": bool(regret_pct <= 1.0),
        "within_5pct": bool(regret_pct <= 5.0),
        "regret_pct": float(regret_pct),
        "spearman_rho": float(_spearman_rho(measured_values, predicted_values)),
        "chosen_schedule": dict(predicted_best["schedule"]),
        "oracle_schedule": dict(measured_best["schedule"]),
        "chosen_measured_latency_us_median": float(predicted_best["measured_median_us"]),
        "oracle_measured_latency_us_median": float(measured_best["measured_median_us"]),
        "schedule_rows": schedule_rows,
    }


def evaluate_mean_latency_baseline(rows: Sequence[Dict[str, Any]], train_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mean_latency = float(np.mean(build_target_vector(train_rows))) if train_rows else 0.0
    predicted = [mean_latency for _ in rows]
    return evaluate_shape_predictions(rows, predicted, prediction_name="mean_latency")


def evaluate_random_schedule_baseline(
    rows: Sequence[Dict[str, Any]],
    seed: str,
    shape: Tuple[int, int],
) -> Dict[str, Any]:
    predicted = [
        _stable_random_score(seed, shape, schedule_key(row))
        for row in rows
    ]
    return evaluate_shape_predictions(rows, predicted, prediction_name="random_schedule")


def summarize_fold_metrics(fold_metrics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not fold_metrics:
        return {
            "fold_count": 0,
            "top1_accuracy": 0.0,
            "within_1pct_fraction": 0.0,
            "within_5pct_fraction": 0.0,
            "mean_regret_pct": 0.0,
            "median_regret_pct": 0.0,
            "p95_regret_pct": 0.0,
            "max_regret_pct": 0.0,
            "mean_spearman_rho": 0.0,
        }
    regrets = [float(item["regret_pct"]) for item in fold_metrics]
    return {
        "fold_count": int(len(fold_metrics)),
        "top1_accuracy": float(sum(1 for item in fold_metrics if item["top1_hit"]) / len(fold_metrics)),
        "within_1pct_fraction": float(sum(1 for item in fold_metrics if item["within_1pct"]) / len(fold_metrics)),
        "within_5pct_fraction": float(sum(1 for item in fold_metrics if item["within_5pct"]) / len(fold_metrics)),
        "mean_regret_pct": float(statistics.fmean(regrets)),
        "median_regret_pct": float(statistics.median(regrets)),
        "p95_regret_pct": float(_percentile(regrets, 95.0)),
        "max_regret_pct": float(max(regrets)),
        "mean_spearman_rho": float(statistics.fmean(float(item["spearman_rho"]) for item in fold_metrics)),
    }
