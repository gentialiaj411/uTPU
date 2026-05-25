import argparse
import json
import math
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

from cost_model import predict_latency_us
from cuda_autotuner import CUDATuningSearchSpace
from cuda_blocked_fc_backend import CUDABlockedFCExecutor, detect_cuda_environment
from lowering_types import BlockedFCLoweringRequest


REPORT_JSON_PATH = Path("build/reports/cost_model_calibration.json")
REPORT_MD_PATH = Path("build/reports/cost_model_calibration.md")

@dataclass(frozen=True)
class CalibrationConfig:
    warmup: int
    iters: int
    array_size: int
    repeat_launches_per_sample: int


def _require_torch():
    import torch

    return torch


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _shape_grid() -> List[Dict[str, int]]:
    in_features = (16, 32, 64, 128, 256, 512)
    hidden_features = (16, 64, 256, 512)
    out_features = (16, 64, 256, 512)
    grid = []
    for i in in_features:
        for h in hidden_features:
            for o in out_features:
                grid.append(
                    {
                        "in_features": int(i),
                        "hidden_features": int(h),
                        "out_features": int(o),
                    }
                )
    return grid


def _deterministic_request(out_features: int, in_features: int, array_size: int, seed: int) -> BlockedFCLoweringRequest:
    rng = np.random.default_rng(seed)
    weights = rng.integers(-8, 8, size=(out_features, in_features), dtype=np.int8)
    activations = rng.integers(-8, 8, size=(in_features,), dtype=np.int8)
    return BlockedFCLoweringRequest(
        weights_int4=weights,
        activations_int4=activations,
        out_features=int(out_features),
        in_features=int(in_features),
        array_size=int(array_size),
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )


def _feature_terms(shape: Dict[str, Any], schedule: Dict[str, int]) -> Dict[str, float]:
    out_features = int(shape["out_features"])
    in_features = int(shape["in_features"])
    batch = int(shape.get("batch", 1))
    array_size = int(shape.get("array_size", 16))
    threads = int(schedule["threads_per_block"])
    unroll = int(schedule["unroll_factor"])

    out_padded = int(math.ceil(out_features / array_size) * array_size)
    in_padded = int(math.ceil(in_features / array_size) * array_size)

    weights_bytes = float(out_padded * in_padded)
    activations_bytes = float(batch * in_padded)
    outputs_bytes = float(batch * out_padded * 4)
    memory_kib = (weights_bytes + activations_bytes + outputs_bytes) / 1024.0
    cta_rows = min(float(out_padded), float(threads))
    cta_memory_kib = ((cta_rows * float(in_padded)) + activations_bytes + (float(batch) * cta_rows * 4.0)) / 1024.0

    warp_width = 32.0
    warps_per_block = max(1.0, threads / warp_width)
    active_warps = (out_padded / threads) * warps_per_block
    occupancy_proxy = min(1.0, active_warps / 8.0)
    underoccupancy_penalty = (1.0 - occupancy_proxy) ** 2
    wave_count = float(math.ceil(out_padded / float(threads)))

    out_tail = float(out_padded - out_features) / float(out_padded)
    in_tail = float(in_padded - in_features) / float(in_padded)
    tail_ratio = 0.5 * (out_tail + in_tail)
    unroll_log2 = math.log2(float(unroll))
    max_unroll_log2 = math.log2(8.0)
    unroll_norm = unroll_log2 / max_unroll_log2
    # This term is zero on fully aligned grids (e.g., padded K divisible by {1,2,4,8}).
    unroll_k_tail = float(in_padded % unroll) / float(unroll)
    shape_compute_scale = math.log2(1.0 + (float(out_padded) * float(in_padded) / 1024.0))
    unroll_shape_interaction = unroll_norm * shape_compute_scale
    tpb_norm = math.log2(float(threads) / 32.0) / math.log2(8.0)
    small_out_ratio = max(0.0, 64.0 - float(out_padded)) / 64.0
    small_out_tpb_interaction = small_out_ratio * tpb_norm
    small_out_unroll_interaction = small_out_ratio * unroll_norm
    idle_thread_ratio = max(0.0, float(threads) - float(out_padded)) / float(threads)
    waves = float(math.ceil(float(out_padded) / float(threads)))
    wave_tpb_interaction = waves * tpb_norm
    small_out_idle_penalty = small_out_ratio * idle_thread_ratio
    large_k_unroll_gain = unroll_log2 * math.log2(1.0 + (float(in_padded) / 128.0))
    small_out_unroll_penalty = small_out_ratio * unroll_log2
    small_k_ratio = max(0.0, 1.0 - (float(in_padded) / 256.0))
    large_out_small_k_wave_tpb_efficiency = (
        tpb_norm * math.log2(1.0 + (float(out_padded) / 128.0)) / max(1.0, waves) * small_k_ratio
    )

    return {
        "memory_kib": float(memory_kib),
        "cta_memory_kib": float(cta_memory_kib),
        "underoccupancy_penalty": float(underoccupancy_penalty),
        "tail_ratio": float(tail_ratio),
        "unroll_norm": float(unroll_norm),
        "unroll_k_tail": float(unroll_k_tail),
        "unroll_shape_interaction": float(unroll_shape_interaction),
        "tpb_norm": float(tpb_norm),
        "small_out_ratio": float(small_out_ratio),
        "small_out_tpb_interaction": float(small_out_tpb_interaction),
        "small_out_unroll_interaction": float(small_out_unroll_interaction),
        "idle_thread_ratio": float(idle_thread_ratio),
        "waves": float(waves),
        "wave_tpb_interaction": float(wave_tpb_interaction),
        "small_out_idle_penalty": float(small_out_idle_penalty),
        "large_k_unroll_gain": float(large_k_unroll_gain),
        "small_out_unroll_penalty": float(small_out_unroll_penalty),
        "large_out_small_k_wave_tpb_efficiency": float(large_out_small_k_wave_tpb_efficiency),
        "in_padded": float(in_padded),
        "out_padded": float(out_padded),
        "cta_rows": float(cta_rows),
        "wave_count": wave_count,
    }


def _fit_coefficients(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eps = 1e-9
    y = np.asarray([float(r["measured_latency_us"]) for r in rows], dtype=np.float64)
    log_y = np.log(np.maximum(y, eps))
    pairwise_lambda = 0.1
    pairwise_scale = 20.0
    pairwise_tie_rel = 0.01
    pairwise_near_winner_rel = 0.05
    p0 = {
        "intercept_us": 2.0,
        "memory_us_per_kib": 0.01,
        "cta_memory_us_per_kib": 0.2,
        "underoccupancy_penalty_us": 2.0,
        "tile_tail_penalty_us": 0.5,
        "unroll_gain_us": 0.2,
        "unroll_k_tail_penalty_us": 0.4,
        "unroll_shape_interaction_us": 0.15,
        "small_out_tpb_interaction_us": 0.25,
        "small_out_unroll_interaction_us": 0.25,
        "idle_thread_ratio_us": 0.2,
        "wave_tpb_interaction_us": 0.1,
        "small_out_idle_penalty_us": 0.3,
        "large_k_unroll_gain_us": 0.25,
        "small_out_unroll_penalty_us": 0.2,
        "large_out_small_k_wave_tpb_efficiency_us": 0.25,
    }
    feature_key_for_coeff = {
        "memory_us_per_kib": "memory_kib",
        "cta_memory_us_per_kib": "cta_memory_kib",
        "underoccupancy_penalty_us": "underoccupancy_penalty",
        "tile_tail_penalty_us": "tail_ratio",
        "unroll_gain_us": "unroll_norm",
        "unroll_k_tail_penalty_us": "unroll_k_tail",
        "unroll_shape_interaction_us": "unroll_shape_interaction",
        "small_out_tpb_interaction_us": "small_out_tpb_interaction",
        "small_out_unroll_interaction_us": "small_out_unroll_interaction",
        "idle_thread_ratio_us": "idle_thread_ratio",
        "wave_tpb_interaction_us": "wave_tpb_interaction",
        "small_out_idle_penalty_us": "small_out_idle_penalty",
        "large_k_unroll_gain_us": "large_k_unroll_gain",
        "small_out_unroll_penalty_us": "small_out_unroll_penalty",
        "large_out_small_k_wave_tpb_efficiency_us": "large_out_small_k_wave_tpb_efficiency",
    }
    active_coeffs = []
    fixed_coeffs = []
    for coeff_name, feat_name in feature_key_for_coeff.items():
        vals = np.asarray([float(r["features"][feat_name]) for r in rows], dtype=np.float64)
        if float(np.std(vals)) <= 1e-12:
            fixed_coeffs.append(coeff_name)
        else:
            active_coeffs.append(coeff_name)

    row_groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, row in enumerate(rows):
        shape = row["shape_used"]
        key = (int(shape["in_features"]), int(shape["out_features"]))
        row_groups.setdefault(key, []).append(idx)
    pairwise_indices: List[Tuple[int, int, float]] = []
    for indices in row_groups.values():
        by_schedule: Dict[Tuple[int, int], List[int]] = {}
        for idx in indices:
            schedule = rows[idx]["schedule"]
            key = (int(schedule["threads_per_block"]), int(schedule["unroll_factor"]))
            by_schedule.setdefault(key, []).append(idx)
        schedule_rows = []
        for schedule_key, schedule_indices in by_schedule.items():
            measured = np.asarray([float(y[idx]) for idx in schedule_indices], dtype=np.float64)
            schedule_rows.append(
                {
                    "schedule": schedule_key,
                    "representative_index": int(schedule_indices[0]),
                    "measured_latency_us": float(np.median(measured)),
                }
            )
        if len(schedule_rows) <= 1:
            continue
        best_latency = min(float(item["measured_latency_us"]) for item in schedule_rows)
        near_winner = {
            item["schedule"]
            for item in schedule_rows
            if float(item["measured_latency_us"]) <= best_latency * (1.0 + pairwise_near_winner_rel)
        }
        for pos_i, left in enumerate(schedule_rows):
            for right in schedule_rows[pos_i + 1 :]:
                if left["schedule"] not in near_winner and right["schedule"] not in near_winner:
                    continue
                dm = float(left["measured_latency_us"] - right["measured_latency_us"])
                if abs(dm) / max(best_latency, eps) <= pairwise_tie_rel:
                    continue
                i = int(left["representative_index"])
                j = int(right["representative_index"])
                pairwise_indices.append((i, j, 1.0 if dm > 0.0 else -1.0))

    def _predict_latency(params: np.ndarray) -> np.ndarray:
        coeffs = dict(p0)
        coeffs["intercept_us"] = float(params[0])
        for i, name in enumerate(active_coeffs, start=1):
            coeffs[name] = float(params[i])
        preds = []
        for row in rows:
            f = row["features"]
            lat = coeffs["intercept_us"] + coeffs["cta_memory_us_per_kib"] * float(f["cta_memory_kib"])
            lat += coeffs["underoccupancy_penalty_us"] * float(f["underoccupancy_penalty"])
            lat += coeffs["tile_tail_penalty_us"] * float(f["tail_ratio"])
            lat += coeffs["memory_us_per_kib"] * float(f["memory_kib"])
            lat -= coeffs["unroll_gain_us"] * float(f["unroll_norm"])
            lat += coeffs["unroll_k_tail_penalty_us"] * float(f["unroll_k_tail"])
            lat -= coeffs["unroll_shape_interaction_us"] * float(f["unroll_shape_interaction"])
            lat += coeffs["small_out_tpb_interaction_us"] * float(f["small_out_tpb_interaction"])
            lat += coeffs["small_out_unroll_interaction_us"] * float(f["small_out_unroll_interaction"])
            lat += coeffs["idle_thread_ratio_us"] * float(f["idle_thread_ratio"])
            lat += coeffs["wave_tpb_interaction_us"] * float(f["wave_tpb_interaction"])
            lat += coeffs["small_out_idle_penalty_us"] * float(f["small_out_idle_penalty"])
            lat -= coeffs["large_k_unroll_gain_us"] * float(f["large_k_unroll_gain"])
            lat += coeffs["small_out_unroll_penalty_us"] * float(f["small_out_unroll_penalty"])
            lat -= coeffs["large_out_small_k_wave_tpb_efficiency_us"] * float(f["large_out_small_k_wave_tpb_efficiency"])
            preds.append(max(lat, eps))
        return np.asarray(preds, dtype=np.float64)

    def _residuals(params: np.ndarray) -> np.ndarray:
        pred = _predict_latency(params)
        log_pred = np.log(np.maximum(pred, eps))
        latency_residual = (log_pred - log_y) / math.sqrt(max(1, len(rows)))
        if not pairwise_indices:
            return latency_residual
        pair_residuals = []
        pair_scale = math.sqrt(pairwise_lambda / max(1, len(pairwise_indices)))
        for i, j, sign_dm in pairwise_indices:
            dp = float(log_pred[i] - log_pred[j])
            penalty = float(np.logaddexp(0.0, -pairwise_scale * sign_dm * dp))
            pair_residuals.append(pair_scale * math.sqrt(max(0.0, penalty)))
        return np.concatenate([latency_residual, np.asarray(pair_residuals, dtype=np.float64)])

    def _predict_log(params: np.ndarray) -> np.ndarray:
        pred = _predict_latency(params)
        return np.log(np.maximum(pred, eps))

    p0_vec = np.asarray([p0["intercept_us"]] + [p0[name] for name in active_coeffs], dtype=np.float64)
    fit = least_squares(
        fun=_residuals,
        x0=p0_vec,
        bounds=(np.zeros_like(p0_vec), np.full_like(p0_vec, np.inf)),
        method="trf",
    )
    fit_vec = fit.x
    moved = float(np.linalg.norm(fit_vec - p0_vec, ord=2))
    if moved <= 1e-9:
        raise RuntimeError(
            "Calibration fit did not move from p0. Refusing to emit uncalibrated coefficients."
        )
    resolved = dict(p0)
    resolved["intercept_us"] = float(fit_vec[0])
    for i, name in enumerate(active_coeffs, start=1):
        resolved[name] = float(fit_vec[i])
    return {
        "coefficients": resolved,
        "fit_status": "ok",
        "fit_message": str(fit.message),
        "fit_nfev": int(fit.nfev),
        "fit_cost": float(fit.cost),
        "fit_l2_delta_from_p0": moved,
        "model_form": "intercept + cta_memory_kib + total_memory_kib + underoccupancy + tile_tail - unroll_gain + unroll_k_tail_penalty - unroll_shape_interaction + small_out_tpb_interaction + small_out_unroll_interaction + idle_thread_ratio + wave_tpb_interaction + small_out_idle_penalty - large_k_unroll_gain + small_out_unroll_penalty - large_out_small_k_wave_tpb_efficiency",
        "fit_initial_guess": p0,
        "fit_active_coefficients": ["intercept_us"] + list(active_coeffs),
        "fit_fixed_coefficients": list(fixed_coeffs),
        "fit_objective": "mean_log_latency_mse_plus_pairwise_ordering",
        "pairwise_ordering": {
            "lambda": float(pairwise_lambda),
            "scale": float(pairwise_scale),
            "tie_rel": float(pairwise_tie_rel),
            "near_winner_rel": float(pairwise_near_winner_rel),
            "pair_filter": "schedule_median_pairs_touching_winner_or_within_5pct_skip_1pct_ties",
            "pair_count": int(len(pairwise_indices)),
        },
    }


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    measured = np.asarray([float(r["measured_latency_us"]) for r in rows], dtype=np.float64)
    predicted = np.asarray([float(r["predicted_latency_us"]) for r in rows], dtype=np.float64)
    residual = measured - predicted
    eps = 1e-9
    log_measured = np.log(np.maximum(measured, eps))
    log_predicted = np.log(np.maximum(predicted, eps))
    log_resid = log_measured - log_predicted
    ss_res = float(np.sum(log_resid ** 2))
    ss_tot = float(np.sum((log_measured - np.mean(log_measured)) ** 2))
    log_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    abs_rel = np.abs(residual) / np.maximum(np.abs(measured), eps)
    mape = float(np.mean(abs_rel) * 100.0)
    p95 = float(np.percentile(abs_rel * 100.0, 95.0))
    return {"log_r2": log_r2, "mape_pct": mape, "p95_abs_rel_error_pct": p95}


def _size_bin(shape_used: Dict[str, Any]) -> str:
    m = int(shape_used["out_features"])
    k = int(shape_used["in_features"])
    scale = max(m, k)
    if scale <= 64:
        return "small"
    if scale <= 256:
        return "medium"
    return "large"


def _residual_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[float]] = {"small": [], "medium": [], "large": []}
    for row in rows:
        buckets[_size_bin(row["shape_used"])].append(float(row["percent_error"]))
    out: Dict[str, Any] = {}
    for name, vals in buckets.items():
        if not vals:
            out[name] = {
                "count": 0,
                "mean_percent_error": None,
                "mean_abs_percent_error": None,
                "positive_error_fraction": None,
            }
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[name] = {
            "count": int(arr.size),
            "mean_percent_error": float(np.mean(arr)),
            "mean_abs_percent_error": float(np.mean(np.abs(arr))),
            "positive_error_fraction": float(np.mean(arr > 0.0)),
        }
    return out


def _spearman_rho(measured: np.ndarray, predicted: np.ndarray) -> float:
    if measured.size <= 1:
        return 0.0
    rank_m = np.argsort(np.argsort(measured))
    rank_p = np.argsort(np.argsort(predicted))
    rm = rank_m.astype(np.float64)
    rp = rank_p.astype(np.float64)
    rm -= np.mean(rm)
    rp -= np.mean(rp)
    denom = float(np.linalg.norm(rm) * np.linalg.norm(rp))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(rm, rp) / denom)


def _pairwise_ordering_accuracy(measured: np.ndarray, predicted: np.ndarray, eps: float = 1e-9) -> float:
    n = int(measured.size)
    if n <= 1:
        return 1.0
    total = 0
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            dm = float(measured[i] - measured[j])
            dp = float(predicted[i] - predicted[j])
            if abs(dm) <= eps:
                continue
            total += 1
            if dm * dp > 0:
                correct += 1
    if total == 0:
        return 1.0
    return float(correct / total)


def _ranking_diagnostics(rows: List[Dict[str, Any]], top_k: int = 4) -> Dict[str, Any]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        shape = row["shape_used"]
        key = (int(shape["in_features"]), int(shape["out_features"]))
        grouped.setdefault(key, []).append(row)

    per_shape = []
    top1_hits = 0
    topk_hits = 0
    for (in_features, out_features), vals in sorted(grouped.items()):
        measured = np.asarray([float(v["measured_latency_us"]) for v in vals], dtype=np.float64)
        predicted = np.asarray([float(v["predicted_latency_us"]) for v in vals], dtype=np.float64)
        order_pred = np.argsort(predicted)
        order_meas = np.argsort(measured)
        winner_idx = int(order_meas[0])
        top1_idx = int(order_pred[0])
        topk_idx = set(int(i) for i in order_pred[: max(1, int(top_k))])
        top1_match = top1_idx == winner_idx
        topk_contains = winner_idx in topk_idx
        if top1_match:
            top1_hits += 1
        if topk_contains:
            topk_hits += 1
        per_shape.append(
            {
                "shape": {"in_features": int(in_features), "out_features": int(out_features)},
                "top1_matches_winner": bool(top1_match),
                "winner_in_top_k": bool(topk_contains),
                "spearman_rho": _spearman_rho(measured, predicted),
                "pairwise_ordering_accuracy": _pairwise_ordering_accuracy(measured, predicted),
            }
        )
    count = len(per_shape)
    return {
        "top_k": int(top_k),
        "shape_count": int(count),
        "top1_winner_accuracy": float(top1_hits / count) if count else None,
        "topk_contains_winner_accuracy": float(topk_hits / count) if count else None,
        "mean_spearman_rho": float(np.mean([p["spearman_rho"] for p in per_shape])) if per_shape else None,
        "mean_pairwise_ordering_accuracy": float(np.mean([p["pairwise_ordering_accuracy"] for p in per_shape])) if per_shape else None,
        "per_shape": per_shape,
    }


def _write_md(report: Dict[str, Any], path: Path) -> None:
    rows = report["per_point"]
    metrics = report["aggregate_metrics"]
    residuals = report.get("residual_diagnostics", {})
    ranking = report.get("ranking_diagnostics", {})
    lines = []
    lines.append("# Cost Model Calibration")
    lines.append("")
    lines.append(f"- timestamp_utc: {report['timestamp_utc']}")
    if report.get("refit_from_existing_measurements"):
        lines.append("- refit_from_existing_measurements: true")
        lines.append(f"- measurement_timestamp_utc: {report.get('measurement_timestamp_utc', 'unknown')}")
    lines.append(f"- git_sha: {report['git_sha']}")
    lines.append(f"- total_shapes: {len(report['shape_grid'])}")
    lines.append(f"- total_measurements: {report['total_measurement_count']}")
    lines.append(f"- model_form: {report.get('model_form', 'unknown')}")
    lines.append(f"- fit_objective: {report.get('fit_objective', 'mean_log_latency_mse')}")
    pairwise = report.get("pairwise_ordering", {})
    if pairwise:
        lines.append(
            f"- pairwise_ordering: lambda={pairwise.get('lambda', 0.0):.3f}, "
            f"scale={pairwise.get('scale', 0.0):.1f}, pairs={pairwise.get('pair_count', 0)}, "
            f"filter={pairwise.get('pair_filter', 'all_non_tied_pairs')}"
        )
    lines.append(
        f"- warmup: {report['methodology']['warmup_runs']}, "
        f"iters: {report['methodology']['timed_runs_per_point']}, "
        f"repeat_launches_per_sample: {report['methodology']['repeat_launches_per_sample']}"
    )
    lines.append(
        f"- log_R2: {metrics['log_r2']:.4f}, MAPE: {metrics['mape_pct']:.2f}%, "
        f"p95_abs_rel_error: {metrics['p95_abs_rel_error_pct']:.2f}%"
    )
    if ranking:
        lines.append(
            f"- ranking(top-{ranking.get('top_k', 4)}): top1={ranking.get('top1_winner_accuracy', 0.0):.4f}, "
            f"topk_contains={ranking.get('topk_contains_winner_accuracy', 0.0):.4f}, "
            f"mean_spearman={ranking.get('mean_spearman_rho', 0.0):.4f}, "
            f"mean_pairwise={ranking.get('mean_pairwise_ordering_accuracy', 0.0):.4f}"
        )
    lines.append("")
    lines.append("Residual diagnostics by size bin")
    lines.append("")
    lines.append("| bin | count | mean_pct_error | mean_abs_pct_error | positive_error_fraction |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ("small", "medium", "large"):
        r = residuals.get(name, {})
        if not r or r.get("count", 0) == 0:
            lines.append(f"| {name} | 0 | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {name} | {r['count']} | {r['mean_percent_error']:.2f}% | "
                f"{r['mean_abs_percent_error']:.2f}% | {r['positive_error_fraction']:.2f} |"
            )
    lines.append("")
    if ranking and ranking.get("per_shape"):
        lines.append("Worst 5 shapes by pairwise ordering accuracy")
        lines.append("")
        lines.append("| shape(in,out) | top1_match | winner_in_topk | spearman | pairwise_acc |")
        lines.append("|---|---:|---:|---:|---:|")
        worst_rank = sorted(ranking["per_shape"], key=lambda r: float(r["pairwise_ordering_accuracy"]))[:5]
        for r in worst_rank:
            s = r["shape"]
            lines.append(
                f"| ({s['in_features']},{s['out_features']}) | "
                f"{int(bool(r['top1_matches_winner']))} | "
                f"{int(bool(r['winner_in_top_k']))} | "
                f"{float(r['spearman_rho']):.3f} | "
                f"{float(r['pairwise_ordering_accuracy']):.3f} |"
            )
        lines.append("")

    lines.append("")
    lines.append("Top 5 worst absolute percent errors")
    lines.append("")
    lines.append("| shape(in,h,out) | schedule(tpb,u) | measured_us | predicted_us | abs_pct_error |")
    lines.append("|---|---:|---:|---:|---:|")
    worst = sorted(rows, key=lambda r: abs(float(r["percent_error"])), reverse=True)[:5]
    for r in worst:
        s = r["shape_triplet"]
        sch = r["schedule"]
        lines.append(
            f"| ({s['in_features']},{s['hidden_features']},{s['out_features']}) | "
            f"({sch['threads_per_block']},{sch['unroll_factor']}) | "
            f"{r['measured_latency_us']:.3f} | {r['predicted_latency_us']:.3f} | {abs(r['percent_error']):.2f}% |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _measure_point_cuda_event(
    executor: CUDABlockedFCExecutor,
    request: BlockedFCLoweringRequest,
    schedule: Dict[str, int],
    warmup: int,
    iters: int,
    repeat_launches_per_sample: int,
) -> float:
    from cuda_blocked_fc_backend import _numpy_blocked_fc_reference

    cuda, _ = executor._load_cuda_bindings()
    executor._ensure_context()
    ref = _numpy_blocked_fc_reference(
        request.weights_int4,
        request.activations_int4,
        request.out_features,
        request.in_features,
        request.array_size,
        request.apply_relu,
        request.apply_quant,
    )
    sched = ref["schedule"]
    fn, _, _, _ = executor._get_kernel(request, sched, schedule)
    w_pad = ref["weights_padded"]
    x_pad = ref["inputs_padded"]
    out_elems = int(sched.out_padded)

    d_w, _, _ = executor._get_buffer("calib_weights", int(w_pad.size))
    d_x, _, _ = executor._get_buffer("calib_inputs", int(x_pad.size))
    d_out, _, _ = executor._get_buffer("calib_outputs", int(out_elems * np.dtype(np.int32).itemsize))
    executor._check_cuda(cuda.cuMemcpyHtoD(d_w, w_pad.tobytes(), int(w_pad.size))[0], "cuMemcpyHtoD(calib_w)")
    executor._check_cuda(cuda.cuMemcpyHtoD(d_x, x_pad.tobytes(), int(x_pad.size))[0], "cuMemcpyHtoD(calib_x)")

    import ctypes

    arg_w = ctypes.c_void_p(int(d_w))
    arg_x = ctypes.c_void_p(int(d_x))
    arg_out = ctypes.c_void_p(int(d_out))
    arg_in = ctypes.c_int32(int(sched.in_padded))
    arg_out_elems = ctypes.c_int32(int(out_elems))
    kernel_args = (ctypes.c_void_p * 5)(
        ctypes.cast(ctypes.pointer(arg_w), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_x), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_out), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_in), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_out_elems), ctypes.c_void_p),
    )

    threads = int(schedule["threads_per_block"])
    blocks = int(math.ceil(out_elems / threads))
    stream = 0
    err, start_evt = cuda.cuEventCreate(0)
    executor._check_cuda(err, "cuEventCreate(start)")
    err, end_evt = cuda.cuEventCreate(0)
    executor._check_cuda(err, "cuEventCreate(end)")
    try:
        for _ in range(warmup):
            for _ in range(repeat_launches_per_sample):
                err, = cuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream, kernel_args, 0)
                executor._check_cuda(err, "cuLaunchKernel(warmup)")
        executor._check_cuda(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize(warmup)")

        samples_ms = []
        for _ in range(iters):
            executor._check_cuda(cuda.cuEventRecord(start_evt, stream)[0], "cuEventRecord(start)")
            for _ in range(repeat_launches_per_sample):
                err, = cuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream, kernel_args, 0)
                executor._check_cuda(err, "cuLaunchKernel(timed)")
            executor._check_cuda(cuda.cuEventRecord(end_evt, stream)[0], "cuEventRecord(end)")
            executor._check_cuda(cuda.cuEventSynchronize(end_evt)[0], "cuEventSynchronize(end)")
            err, elapsed_ms = cuda.cuEventElapsedTime(start_evt, end_evt)
            executor._check_cuda(err, "cuEventElapsedTime")
            samples_ms.append(float(elapsed_ms) / float(repeat_launches_per_sample))
        return float(np.median(np.asarray(samples_ms, dtype=np.float64)) * 1000.0)
    finally:
        cuda.cuEventDestroy(start_evt)
        cuda.cuEventDestroy(end_evt)


def run_calibration(config: CalibrationConfig) -> Dict[str, Any]:
    torch = _require_torch()
    env = detect_cuda_environment()
    if not env.runtime_available or not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable for calibration. cuda_backend={env.runtime_available}, torch_cuda={torch.cuda.is_available()}, reason={env.reason}"
        )

    shape_grid = _shape_grid()
    schedule_grid = [c.to_dict() for c in CUDATuningSearchSpace().candidates()]
    executor = CUDABlockedFCExecutor(verbose=False)
    per_point = []
    idx = 0
    for triplet in shape_grid:
        layer_shapes = [
            {
                "layer_name": "fc1",
                "in_features": int(triplet["in_features"]),
                "out_features": int(triplet["hidden_features"]),
            },
            {
                "layer_name": "fc2",
                "in_features": int(triplet["hidden_features"]),
                "out_features": int(triplet["out_features"]),
            },
        ]
        for layer in layer_shapes:
            req = _deterministic_request(
                out_features=layer["out_features"],
                in_features=layer["in_features"],
                array_size=config.array_size,
                seed=idx,
            )
            for schedule in schedule_grid:
                measured_us = _measure_point_cuda_event(
                    executor=executor,
                    request=req,
                    schedule=schedule,
                    warmup=config.warmup,
                    iters=config.iters,
                    repeat_launches_per_sample=config.repeat_launches_per_sample,
                )
                shape = {
                    "out_features": int(layer["out_features"]),
                    "in_features": int(layer["in_features"]),
                    "batch": 1,
                    "array_size": config.array_size,
                    "apply_quant": True,
                }
                per_point.append(
                    {
                        "shape_triplet": dict(triplet),
                        "layer_name": layer["layer_name"],
                        "shape_used": dict(shape),
                        "schedule": dict(schedule),
                        "features": _feature_terms(shape, schedule),
                        "measured_latency_us": float(measured_us),
                    }
                )
            idx += 1

    fit_info = _fit_coefficients(per_point)
    fitted = fit_info["coefficients"]
    target = {"name": "cuda", "cost_model_coefficients": fitted}
    for row in per_point:
        pred = predict_latency_us(row["shape_used"], row["schedule"], target=target)
        row["predicted_latency_us"] = float(pred)
        row["percent_error"] = float(((pred - row["measured_latency_us"]) / row["measured_latency_us"]) * 100.0)

    aggregate = _metrics(per_point)
    residual = _residual_diagnostics(per_point)
    ranking = _ranking_diagnostics(per_point, top_k=4)
    device_name = torch.cuda.get_device_name(0)
    report = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "machine_info": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        },
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": str(torch.version.cuda),
            "gpu_name": device_name,
        },
        "shape_grid": shape_grid,
        "schedule_grid": schedule_grid,
        "total_measurement_count": int(len(per_point)),
        "fitted_coefficients": fitted,
        "fit_status": fit_info["fit_status"],
        "fit_message": fit_info["fit_message"],
        "fit_nfev": fit_info["fit_nfev"],
        "fit_cost": fit_info["fit_cost"],
        "fit_l2_delta_from_p0": fit_info["fit_l2_delta_from_p0"],
        "model_form": fit_info["model_form"],
        "fit_initial_guess": fit_info["fit_initial_guess"],
        "fit_active_coefficients": fit_info.get("fit_active_coefficients", []),
        "fit_fixed_coefficients": fit_info.get("fit_fixed_coefficients", []),
        "fit_objective": fit_info.get("fit_objective", "mean_log_latency_mse"),
        "pairwise_ordering": fit_info.get("pairwise_ordering", {}),
        "aggregate_metrics": aggregate,
        "residual_diagnostics": residual,
        "ranking_diagnostics": ranking,
        "methodology": {
            "timer_source": "cuda_driver_events",
            "warmup_runs": int(config.warmup),
            "timed_runs_per_point": int(config.iters),
            "repeat_launches_per_sample": int(config.repeat_launches_per_sample),
            "point_estimator": "median",
            "notes": "Each measured point is a single blocked-FC op using deterministic signed int4 inputs/weights and CUDA event timing around kernel launch.",
        },
        "per_point": per_point,
    }
    return report


def run_refit_from_existing_measurements(existing_json: Path) -> Dict[str, Any]:
    src = json.loads(existing_json.read_text(encoding="utf-8"))
    src_points = src.get("per_point")
    if not isinstance(src_points, list) or not src_points:
        raise ValueError("existing calibration JSON must contain non-empty per_point")

    per_point = []
    for row in src_points:
        shape = dict(row["shape_used"])
        schedule = dict(row["schedule"])
        per_point.append(
            {
                "shape_triplet": dict(row.get("shape_triplet", {})),
                "layer_name": row.get("layer_name", "unknown"),
                "shape_used": shape,
                "schedule": schedule,
                "features": _feature_terms(shape, schedule),
                "measured_latency_us": float(row["measured_latency_us"]),
            }
        )

    fit_info = _fit_coefficients(per_point)
    fitted = fit_info["coefficients"]
    target = {"name": "cuda", "cost_model_coefficients": fitted}
    for row in per_point:
        pred = predict_latency_us(row["shape_used"], row["schedule"], target=target)
        row["predicted_latency_us"] = float(pred)
        row["percent_error"] = float(((pred - row["measured_latency_us"]) / row["measured_latency_us"]) * 100.0)

    aggregate = _metrics(per_point)
    residual = _residual_diagnostics(per_point)
    ranking = _ranking_diagnostics(per_point, top_k=4)
    methodology = dict(src.get("methodology", {}))
    methodology["mode"] = "refit_from_existing_measurements"
    methodology["source_measurement_json"] = str(existing_json)
    methodology.setdefault("notes", "Refit from existing measured per_point data; no new CUDA timing.")
    report = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "machine_info": dict(src.get("machine_info", {})),
        "environment": dict(src.get("environment", {})),
        "shape_grid": src.get("shape_grid", _shape_grid()),
        "schedule_grid": src.get("schedule_grid", [c.to_dict() for c in CUDATuningSearchSpace().candidates()]),
        "total_measurement_count": int(len(per_point)),
        "fitted_coefficients": fitted,
        "fit_status": fit_info["fit_status"],
        "fit_message": fit_info["fit_message"],
        "fit_nfev": fit_info["fit_nfev"],
        "fit_cost": fit_info["fit_cost"],
        "fit_l2_delta_from_p0": fit_info["fit_l2_delta_from_p0"],
        "model_form": fit_info["model_form"],
        "fit_initial_guess": fit_info["fit_initial_guess"],
        "fit_active_coefficients": fit_info.get("fit_active_coefficients", []),
        "fit_fixed_coefficients": fit_info.get("fit_fixed_coefficients", []),
        "fit_objective": fit_info.get("fit_objective", "mean_log_latency_mse"),
        "pairwise_ordering": fit_info.get("pairwise_ordering", {}),
        "aggregate_metrics": aggregate,
        "residual_diagnostics": residual,
        "ranking_diagnostics": ranking,
        "methodology": methodology,
        "refit_from_existing_measurements": True,
        "measurement_timestamp_utc": src.get("timestamp_utc"),
        "source_measurement_json": str(existing_json),
        "per_point": per_point,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate analytical CUDA cost model coefficients.")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--array-size", type=int, default=16)
    parser.add_argument("--repeat-launches-per-sample", type=int, default=64)
    parser.add_argument("--output-json", type=str, default=str(REPORT_JSON_PATH))
    parser.add_argument("--output-md", type=str, default=str(REPORT_MD_PATH))
    parser.add_argument("--refit-from-existing-json", type=str, default=None)
    args = parser.parse_args()

    if args.refit_from_existing_json:
        report = run_refit_from_existing_measurements(Path(args.refit_from_existing_json))
    else:
        config = CalibrationConfig(
            warmup=int(args.warmup),
            iters=int(args.iters),
            array_size=int(args.array_size),
            repeat_launches_per_sample=int(args.repeat_launches_per_sample),
        )
        report = run_calibration(config)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(report, Path(args.output_md))
    print(json.dumps(report["aggregate_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
