import argparse
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from cost_model import predict_latency_us
from cuda_autotuner import CUDATuningSearchSpace, load_cost_model_target, select_pruned_candidates


REPORT_JSON_PATH = Path("build/reports/pruned_autotuner_report.json")
REPORT_MD_PATH = Path("build/reports/pruned_autotuner_report.md")
DEFAULT_CALIBRATION_JSON = Path("build/reports/cost_model_calibration.json")


def _parse_excluded_shapes(values: Iterable[str]) -> set[Tuple[int, int]]:
    excluded = set()
    for value in values:
        if not value:
            continue
        parts = value.replace("x", ",").split(",")
        if len(parts) != 2:
            raise ValueError(f"excluded shape must be 'in,out', got {value!r}")
        excluded.add((int(parts[0]), int(parts[1])))
    return excluded


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
                "shape": {
                    "in_features": int(in_features),
                    "out_features": int(out_features),
                    "array_size": int(rows[0]["shape_used"].get("array_size", 16)),
                },
                "schedule": {
                    "threads_per_block": int(threads),
                    "unroll_factor": int(unroll),
                },
                "measured_latency_us": float(measured),
                "replicate_count": int(len(rows)),
            }
        )
    return by_shape


def evaluate_pruned_search(
    calibration_json: Path = DEFAULT_CALIBRATION_JSON,
    top_k: int = 4,
    output_json: Path = REPORT_JSON_PATH,
    output_md: Path = REPORT_MD_PATH,
    exclude_shapes: Iterable[str] = (),
) -> Dict[str, Any]:
    calibration = json.loads(Path(calibration_json).read_text(encoding="utf-8"))
    target = load_cost_model_target(str(calibration_json))
    excluded = _parse_excluded_shapes(exclude_shapes)
    by_shape = _median_shape_schedule_rows(calibration)
    search_space = CUDATuningSearchSpace()
    full_candidate_count = len(search_space.candidates())

    per_shape = []
    regressions = []
    strict_regressions = []
    top1_hits = 0
    topk_contains_winner = 0
    strict_topk_contains_winner = 0
    within_1pct_hits = 0
    strict_within_1pct_hits = 0
    policy_profiled_counts: List[int] = []
    for (in_features, out_features), candidates in sorted(by_shape.items()):
        if (in_features, out_features) in excluded:
            continue
        shape = {
            "out_features": int(out_features),
            "in_features": int(in_features),
            "batch": 1,
            "array_size": int(candidates[0]["shape"]["array_size"]),
            "apply_quant": True,
        }
        ranked = []
        measured_by_schedule = {}
        for candidate in candidates:
            schedule = candidate["schedule"]
            key = (int(schedule["threads_per_block"]), int(schedule["unroll_factor"]))
            measured_by_schedule[key] = candidate
            predicted = predict_latency_us(shape, schedule, target=target)
            ranked.append((float(predicted), key, candidate))
        ranked.sort(key=lambda item: (item[0], item[1]))
        strict_top = [item[2] for item in ranked[: int(top_k)]]
        policy_selected_schedules, _policy_pruned, policy_meta = select_pruned_candidates(
            out_features=out_features,
            in_features=in_features,
            array_size=shape["array_size"],
            candidates=[c["schedule"] for c in candidates],
            top_k=top_k,
            target=target,
        )
        selected_keys = {(int(s["threads_per_block"]), int(s["unroll_factor"])) for s in policy_selected_schedules}
        policy_top = [c for c in candidates if (int(c["schedule"]["threads_per_block"]), int(c["schedule"]["unroll_factor"])) in selected_keys]
        exhaustive_best = min(candidates, key=lambda item: float(item["measured_latency_us"]))
        pruned_best = min(policy_top, key=lambda item: float(item["measured_latency_us"]))
        strict_best = min(strict_top, key=lambda item: float(item["measured_latency_us"]))
        top1 = strict_top[0]
        winner_in_topk = any(item["schedule"] == exhaustive_best["schedule"] for item in policy_top)
        winner_in_strict_topk = any(item["schedule"] == exhaustive_best["schedule"] for item in strict_top)
        if top1["schedule"] == exhaustive_best["schedule"]:
            top1_hits += 1
        if winner_in_topk:
            topk_contains_winner += 1
        if winner_in_strict_topk:
            strict_topk_contains_winner += 1
        regression = (
            (float(pruned_best["measured_latency_us"]) - float(exhaustive_best["measured_latency_us"]))
            / max(float(exhaustive_best["measured_latency_us"]), 1e-9)
            * 100.0
        )
        strict_regression = (
            (float(strict_best["measured_latency_us"]) - float(exhaustive_best["measured_latency_us"]))
            / max(float(exhaustive_best["measured_latency_us"]), 1e-9)
            * 100.0
        )
        if regression <= 1.0:
            within_1pct_hits += 1
        if strict_regression <= 1.0:
            strict_within_1pct_hits += 1
        regressions.append(float(regression))
        strict_regressions.append(float(strict_regression))
        policy_profiled_counts.append(int(len(policy_top)))
        per_shape.append(
            {
                "shape": {"in_features": int(in_features), "out_features": int(out_features)},
                "full_candidate_count": int(len(candidates)),
                "profiled_candidate_count": int(len(policy_top)),
                "search_reduction_x": float(len(candidates) / max(1, int(len(policy_top)))),
                "strict_top_k_profiled_candidate_count": int(top_k),
                "strict_top_k_search_reduction_x": float(len(candidates) / max(1, int(top_k))),
                "exhaustive_best_schedule": dict(exhaustive_best["schedule"]),
                "exhaustive_best_latency_us": float(exhaustive_best["measured_latency_us"]),
                "pruned_best_schedule": dict(pruned_best["schedule"]),
                "pruned_best_latency_us": float(pruned_best["measured_latency_us"]),
                "quality_regression_pct": float(regression),
                "strict_top_k_best_schedule": dict(strict_best["schedule"]),
                "strict_top_k_best_latency_us": float(strict_best["measured_latency_us"]),
                "strict_top_k_quality_regression_pct": float(strict_regression),
                "top1_schedule": dict(top1["schedule"]),
                "top1_matches_winner": bool(top1["schedule"] == exhaustive_best["schedule"]),
                "winner_in_top_k": bool(winner_in_topk),
                "winner_in_strict_top_k": bool(winner_in_strict_topk),
                "top_k_schedules": [dict(item["schedule"]) for item in policy_top],
                "strict_top_k_schedules": [dict(item["schedule"]) for item in strict_top],
                "pruning_policy": policy_meta,
            }
        )

    regressions_sorted = sorted(regressions)
    strict_regressions_sorted = sorted(strict_regressions)
    p95 = regressions_sorted[int(round(0.95 * (len(regressions_sorted) - 1)))] if regressions_sorted else None
    strict_p95 = strict_regressions_sorted[int(round(0.95 * (len(strict_regressions_sorted) - 1)))] if strict_regressions_sorted else None
    avg_profiled = float(statistics.fmean(policy_profiled_counts)) if policy_profiled_counts else float(top_k)
    summary = {
        "shape_count": int(len(per_shape)),
        "full_candidate_count": int(full_candidate_count),
        "profiled_candidate_count": float(avg_profiled),
        "search_reduction_x": float(full_candidate_count / max(1.0, avg_profiled)),
        "strict_top_k_profiled_candidate_count": int(top_k),
        "strict_top_k_search_reduction_x": float(full_candidate_count / max(1, int(top_k))),
        "mean_quality_regression_pct": float(statistics.fmean(regressions)) if regressions else None,
        "median_quality_regression_pct": float(statistics.median(regressions)) if regressions else None,
        "p95_quality_regression_pct": float(p95) if p95 is not None else None,
        "max_quality_regression_pct": float(max(regressions)) if regressions else None,
        "strict_top_k_mean_quality_regression_pct": float(statistics.fmean(strict_regressions)) if strict_regressions else None,
        "strict_top_k_median_quality_regression_pct": float(statistics.median(strict_regressions)) if strict_regressions else None,
        "strict_top_k_p95_quality_regression_pct": float(strict_p95) if strict_p95 is not None else None,
        "strict_top_k_max_quality_regression_pct": float(max(strict_regressions)) if strict_regressions else None,
        "within_5pct_fraction": float(sum(r <= 5.0 for r in regressions) / len(regressions)) if regressions else None,
        "within_1pct_fraction": float(within_1pct_hits / len(regressions)) if regressions else None,
        "strict_top_k_within_1pct_fraction": float(strict_within_1pct_hits / len(strict_regressions)) if strict_regressions else None,
        "top1_winner_accuracy": float(top1_hits / len(per_shape)) if per_shape else None,
        "topk_contains_winner_accuracy": float(topk_contains_winner / len(per_shape)) if per_shape else None,
        "strict_topk_contains_winner_accuracy": float(strict_topk_contains_winner / len(per_shape)) if per_shape else None,
        "policy_contains_winner_accuracy": float(topk_contains_winner / len(per_shape)) if per_shape else None,
    }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_calibration_json": os.fspath(calibration_json),
        "calibration_timestamp_utc": calibration.get("timestamp_utc"),
        "cost_model_coefficients": calibration.get("fitted_coefficients"),
        "methodology": {
            "mode": "measured-data replay",
            "notes": "Candidates are ranked by the fitted cost model; quality uses median measured CUDA-event latencies from the calibration artifact.",
            "top_k": int(top_k),
            "excluded_shape_used_pairs": [list(x) for x in sorted(excluded)],
        },
        "summary": summary,
        "per_shape": per_shape,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(report, output_md)
    return report


def _write_md(report: Dict[str, Any], output_md: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Pruned Autotuner Evaluation",
        "",
        f"- timestamp_utc: {report['timestamp_utc']}",
        f"- source_calibration_json: {report['source_calibration_json']}",
        f"- calibrated_shapes: {summary['shape_count']}",
        f"- full_candidates_per_shape: {summary['full_candidate_count']}",
        f"- profiled_candidates_per_shape: {summary['profiled_candidate_count']}",
        f"- search_reduction_x: {summary['search_reduction_x']:.2f}",
        f"- mean_quality_regression_pct: {summary['mean_quality_regression_pct']:.2f}%",
        f"- p95_quality_regression_pct: {summary['p95_quality_regression_pct']:.2f}%",
        f"- max_quality_regression_pct: {summary['max_quality_regression_pct']:.2f}%",
        f"- within_5pct_fraction: {summary['within_5pct_fraction']:.2f}",
        "",
        "| shape(in,out) | exhaustive_schedule | pruned_schedule | exhaustive_us | pruned_us | regression |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(report["per_shape"], key=lambda item: item["quality_regression_pct"], reverse=True)[:10]:
        shape = row["shape"]
        ex = row["exhaustive_best_schedule"]
        pr = row["pruned_best_schedule"]
        lines.append(
            f"| ({shape['in_features']},{shape['out_features']}) | "
            f"({ex['threads_per_block']},{ex['unroll_factor']}) | "
            f"({pr['threads_per_block']},{pr['unroll_factor']}) | "
            f"{row['exhaustive_best_latency_us']:.3f} | {row['pruned_best_latency_us']:.3f} | "
            f"{row['quality_regression_pct']:.2f}% |"
        )
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate cost-model-pruned CUDA autotuning against measured calibration data.")
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=REPORT_JSON_PATH)
    parser.add_argument("--output-md", type=Path, default=REPORT_MD_PATH)
    parser.add_argument("--exclude-shape", action="append", default=[], help="Exclude a shape_used pair formatted as 'in,out'.")
    args = parser.parse_args()
    report = evaluate_pruned_search(
        calibration_json=args.calibration_json,
        top_k=args.top_k,
        output_json=args.output_json,
        output_md=args.output_md,
        exclude_shapes=args.exclude_shape,
    )
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
