import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cuda_autotuner import (
    CUDATuningSearchSpace,
    load_schedule_cache,
    load_cost_model_target,
    make_cache_key,
    rank_candidates_by_cost_model,
    select_pruned_candidates,
    save_schedule_cache,
    tune_blocked_fc_shape,
)
from evaluate_pruned_autotuner import evaluate_pruned_search
from calibrate_cost_model import _feature_terms, _fit_coefficients, _ranking_diagnostics, run_refit_from_existing_measurements
from cuda_blocked_fc_backend import DEFAULT_CUDA_SCHEDULE_PARAMS, detect_cuda_environment
from pytorch_compiler import compile_mlp_model


class TinyIntegerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
            self.fc2.weight.copy_(torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _temp_cache_path():
    return os.path.join(tempfile.mkdtemp(prefix="utpu_autotune_"), "cache.json")


def _write_tiny_cache(path):
    cache = {"version": 1, "backend": "cuda_blocked_fc", "results": {}}
    for out_features, in_features in ((3, 4), (2, 3)):
        key = make_cache_key(out_features, in_features, 16, "int4_i32", "cuda")
        cache["results"][key] = {
            "best_schedule": dict(DEFAULT_CUDA_SCHEDULE_PARAMS),
            "shape": {"M": out_features, "N": 1, "K": in_features, "array_size": 16},
        }
    save_schedule_cache(cache, path)


def test_autotuner_search_space_nonempty():
    space = CUDATuningSearchSpace(threads_per_block=(64, 128), unroll_factor=(1, 2))
    candidates = space.candidates()
    assert candidates
    assert space.schema()["threads_per_block"]["values"] == [64, 128]


def test_autotuner_runs_one_shape():
    env = detect_cuda_environment()
    path = _temp_cache_path()
    result = tune_blocked_fc_shape(
        out_features=3,
        in_features=4,
        search_space=CUDATuningSearchSpace(threads_per_block=(64,), unroll_factor=(1,)),
        warmup=1,
        iters=2,
        cache_path=path,
    )
    if env.runtime_available:
        assert result.executed is True
        assert result.best_schedule["threads_per_block"] in (64, 128)
        assert result.best_latency_ms is not None
    else:
        assert result.executed is False


def test_autotuner_preserves_correctness():
    env = detect_cuda_environment()
    result = tune_blocked_fc_shape(
        out_features=5,
        in_features=7,
        search_space=CUDATuningSearchSpace(threads_per_block=(32,), unroll_factor=(1, 2)),
        warmup=1,
        iters=2,
        cache_path=_temp_cache_path(),
    )
    if env.runtime_available:
        assert result.max_abs_error == 0
        assert result.best_latency_ms is not None
    else:
        assert result.executed is False


def test_best_schedule_cache_roundtrip():
    path = _temp_cache_path()
    key = make_cache_key(5, 7, 16, "int4_i32", "cuda")
    cache = {"version": 1, "backend": "cuda_blocked_fc", "results": {}}
    cache["results"][key] = {"best_schedule": {"threads_per_block": 64, "unroll_factor": 2}}
    save_schedule_cache(cache, path)
    loaded = load_schedule_cache(path)
    assert loaded["results"][key]["best_schedule"]["threads_per_block"] == 64


def test_cost_model_pruning_selects_top_k_candidates():
    candidates = [c.to_dict() for c in CUDATuningSearchSpace().candidates()]
    target = {
        "name": "cuda",
        "cost_model_coefficients": {
            "intercept_us": 5.0,
            "memory_us_per_kib": 0.02,
            "cta_memory_us_per_kib": 0.3,
            "underoccupancy_penalty_us": 2.0,
            "tile_tail_penalty_us": 0.5,
        },
    }
    ranked = rank_candidates_by_cost_model(512, 512, 16, candidates, target=target)
    selected, pruned, meta = select_pruned_candidates(512, 512, 16, candidates, top_k=4, target=target)

    assert len(selected) >= 4
    assert len(pruned) == len(candidates) - len(selected)
    assert selected[:4] == [item["schedule"] for item in ranked[:4]]
    assert all("predicted_latency_us" in item for item in pruned)
    assert meta["selected_candidate_count"] == len(selected)
    assert meta["search_reduction_x"] == len(candidates) / len(selected)


def test_cost_model_ranking_is_schedule_aware_on_unroll():
    candidates = [
        {"threads_per_block": 128, "unroll_factor": 1},
        {"threads_per_block": 128, "unroll_factor": 8},
    ]
    ranked = rank_candidates_by_cost_model(1024, 1024, 16, candidates, target="cuda")
    assert ranked[0]["schedule"]["unroll_factor"] == 8
    assert ranked[0]["predicted_latency_us"] < ranked[1]["predicted_latency_us"]


def test_pruning_keeps_schedule_family_diversity():
    candidates = [c.to_dict() for c in CUDATuningSearchSpace().candidates()]
    selected, _pruned, _meta = select_pruned_candidates(512, 512, 16, candidates, top_k=2, target="cuda")
    unrolls = {item["unroll_factor"] for item in selected}
    assert len(unrolls) == 4


def test_pruning_keeps_thread_block_family_diversity():
    candidates = [c.to_dict() for c in CUDATuningSearchSpace().candidates()]
    selected, _pruned, _meta = select_pruned_candidates(16, 512, 16, candidates, top_k=4, target="cuda")
    thread_blocks = {item["threads_per_block"] for item in selected}
    assert thread_blocks == {32, 64, 128, 256}


def test_pruning_fallback_for_small_search_space():
    candidates = [c.to_dict() for c in CUDATuningSearchSpace(threads_per_block=(64, 128), unroll_factor=(1, 2)).candidates()]
    selected, pruned, meta = select_pruned_candidates(256, 256, 16, candidates, top_k=2, target="cuda")
    assert len(candidates) == 4
    assert selected == candidates
    assert pruned == []
    assert meta["used_policy"] is False


def test_pruning_policy_can_keep_more_than_top_k_on_ties():
    candidates = [c.to_dict() for c in CUDATuningSearchSpace().candidates()]
    target = {
        "name": "cuda",
        "cost_model_coefficients": {
            "intercept_us": 1.0,
            "memory_us_per_kib": 0.0,
            "cta_memory_us_per_kib": 0.0,
            "underoccupancy_penalty_us": 0.0,
            "tile_tail_penalty_us": 0.0,
            "unroll_gain_us": 0.0,
            "unroll_k_tail_penalty_us": 0.0,
            "unroll_shape_interaction_us": 0.0,
            "small_out_tpb_interaction_us": 0.0,
            "small_out_unroll_interaction_us": 0.0,
            "idle_thread_ratio_us": 0.0,
            "wave_tpb_interaction_us": 0.0,
            "small_out_idle_penalty_us": 0.0,
            "large_k_unroll_gain_us": 0.0,
            "small_out_unroll_penalty_us": 0.0,
        },
    }
    selected, _pruned, meta = select_pruned_candidates(128, 128, 16, candidates, top_k=2, target=target)
    assert len(selected) > 2
    assert meta["selected_candidate_count"] == len(selected)
    assert meta["tie_margin_kept_count"] > 0


def test_ranking_is_deterministic_under_prediction_ties():
    candidates = [c.to_dict() for c in CUDATuningSearchSpace().candidates()]
    target = {
        "name": "cuda",
        "cost_model_coefficients": {
            "intercept_us": 1.0,
            "memory_us_per_kib": 0.0,
            "cta_memory_us_per_kib": 0.0,
            "underoccupancy_penalty_us": 0.0,
            "tile_tail_penalty_us": 0.0,
            "unroll_gain_us": 0.0,
            "unroll_k_tail_penalty_us": 0.0,
            "unroll_shape_interaction_us": 0.0,
            "small_out_tpb_interaction_us": 0.0,
            "small_out_unroll_interaction_us": 0.0,
            "idle_thread_ratio_us": 0.0,
            "wave_tpb_interaction_us": 0.0,
            "small_out_idle_penalty_us": 0.0,
            "large_k_unroll_gain_us": 0.0,
            "small_out_unroll_penalty_us": 0.0,
        },
    }
    ranked = rank_candidates_by_cost_model(128, 128, 16, candidates, target=target)
    assert ranked[0]["schedule"] == {"threads_per_block": 32, "unroll_factor": 1}


def test_evaluation_reports_policy_vs_strict_topk_fields():
    import json

    temp_dir = Path(tempfile.mkdtemp(prefix="utpu_eval_"))
    calib_path = temp_dir / "calib.json"
    out_json = temp_dir / "report.json"
    out_md = temp_dir / "report.md"
    candidates = [c.to_dict() for c in CUDATuningSearchSpace().candidates()]
    per_point = []
    for sched in candidates:
        measured = 50.0 + float(sched["threads_per_block"]) * 0.01 + float(sched["unroll_factor"]) * 0.1
        if sched["threads_per_block"] == 256 and sched["unroll_factor"] == 1:
            measured = 1.0
        per_point.append(
            {
                "shape_used": {"in_features": 64, "out_features": 16, "array_size": 16},
                "schedule": dict(sched),
                "measured_latency_us": measured,
            }
        )
    coeffs = {
        "intercept_us": 1.0,
        "memory_us_per_kib": 0.0,
        "cta_memory_us_per_kib": 0.0,
        "underoccupancy_penalty_us": 0.0,
        "tile_tail_penalty_us": 0.0,
        "unroll_gain_us": 0.0,
        "unroll_k_tail_penalty_us": 0.0,
        "unroll_shape_interaction_us": 0.0,
        "small_out_tpb_interaction_us": 0.0,
        "small_out_unroll_interaction_us": 0.0,
        "idle_thread_ratio_us": 0.0,
        "wave_tpb_interaction_us": 0.0,
        "small_out_idle_penalty_us": 0.0,
        "large_k_unroll_gain_us": 0.0,
        "small_out_unroll_penalty_us": 0.0,
    }
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": "2026-01-01T00:00:00+00:00", "fitted_coefficients": coeffs, "per_point": per_point}, f)
    report = evaluate_pruned_search(calibration_json=calib_path, top_k=2, output_json=out_json, output_md=out_md)
    assert report["summary"]["profiled_candidate_count"] > 2.0
    assert report["summary"]["strict_top_k_profiled_candidate_count"] == 2
    assert report["summary"]["policy_contains_winner_accuracy"] >= report["summary"]["strict_topk_contains_winner_accuracy"]
    assert {
        "profiled_candidate_count",
        "search_reduction_x",
        "strict_top_k_profiled_candidate_count",
        "strict_top_k_search_reduction_x",
        "policy_contains_winner_accuracy",
        "strict_topk_contains_winner_accuracy",
        "max_quality_regression_pct",
        "strict_top_k_max_quality_regression_pct",
        "within_1pct_fraction",
    }.issubset(report["summary"].keys())
    assert {
        "profiled_candidate_count",
        "strict_top_k_profiled_candidate_count",
        "quality_regression_pct",
        "strict_top_k_quality_regression_pct",
        "top_k_schedules",
        "strict_top_k_schedules",
        "pruning_policy",
    }.issubset(report["per_shape"][0].keys())


def test_load_cost_model_target_reads_schedule_aware_coefficients():
    import json

    coeffs = {
        "intercept_us": 1.0,
        "memory_us_per_kib": 0.01,
        "cta_memory_us_per_kib": 0.2,
        "underoccupancy_penalty_us": 3.0,
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
    }
    path = _temp_cache_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fitted_coefficients": coeffs}, f)
    target = load_cost_model_target(path)
    loaded = target["cost_model_coefficients"]
    assert loaded["unroll_gain_us"] == coeffs["unroll_gain_us"]
    assert loaded["unroll_k_tail_penalty_us"] == coeffs["unroll_k_tail_penalty_us"]
    assert loaded["unroll_shape_interaction_us"] == coeffs["unroll_shape_interaction_us"]
    assert loaded["small_out_tpb_interaction_us"] == coeffs["small_out_tpb_interaction_us"]
    assert loaded["small_out_unroll_interaction_us"] == coeffs["small_out_unroll_interaction_us"]
    assert loaded["idle_thread_ratio_us"] == coeffs["idle_thread_ratio_us"]
    assert loaded["wave_tpb_interaction_us"] == coeffs["wave_tpb_interaction_us"]
    assert loaded["small_out_idle_penalty_us"] == coeffs["small_out_idle_penalty_us"]
    assert loaded["large_k_unroll_gain_us"] == coeffs["large_k_unroll_gain_us"]
    assert loaded["small_out_unroll_penalty_us"] == coeffs["small_out_unroll_penalty_us"]


def test_backward_compatible_coeff_loading_with_missing_new_terms():
    import json

    path = _temp_cache_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fitted_coefficients": {"intercept_us": 1.0}}, f)
    target = load_cost_model_target(path)
    ranked = rank_candidates_by_cost_model(64, 64, 16, [{"threads_per_block": 32, "unroll_factor": 1}], target=target)
    assert ranked[0]["predicted_latency_us"] > 0.0


def test_ranking_diagnostics_reports_per_shape_metrics():
    rows = [
        {"shape_used": {"in_features": 16, "out_features": 16}, "measured_latency_us": 1.0, "predicted_latency_us": 1.1},
        {"shape_used": {"in_features": 16, "out_features": 16}, "measured_latency_us": 2.0, "predicted_latency_us": 1.9},
        {"shape_used": {"in_features": 32, "out_features": 16}, "measured_latency_us": 1.0, "predicted_latency_us": 1.0},
        {"shape_used": {"in_features": 32, "out_features": 16}, "measured_latency_us": 3.0, "predicted_latency_us": 3.0},
    ]
    diag = _ranking_diagnostics(rows, top_k=1)
    assert diag["shape_count"] == 2
    assert "top1_winner_accuracy" in diag
    assert "mean_spearman_rho" in diag


def test_cost_model_fit_records_pairwise_ordering_objective():
    shape = {"in_features": 128, "out_features": 64, "batch": 1, "array_size": 16}
    schedules = [
        {"threads_per_block": 32, "unroll_factor": 1},
        {"threads_per_block": 64, "unroll_factor": 2},
        {"threads_per_block": 128, "unroll_factor": 4},
        {"threads_per_block": 256, "unroll_factor": 8},
    ]
    rows = []
    for idx, schedule in enumerate(schedules):
        rows.append(
            {
                "shape_used": dict(shape),
                "schedule": dict(schedule),
                "features": _feature_terms(shape, schedule),
                "measured_latency_us": float(10.0 + idx),
            }
        )
    fit = _fit_coefficients(rows)
    assert fit["fit_objective"] == "mean_log_latency_mse_plus_pairwise_ordering"
    assert fit["pairwise_ordering"]["pair_count"] == 3
    assert fit["pairwise_ordering"]["lambda"] == 0.1
    assert fit["pairwise_ordering"]["near_winner_rel"] == 0.05
    assert fit["pairwise_ordering"]["tie_rel"] == 0.01


def test_refit_report_schema_from_existing_measurements():
    import json

    shape = {"in_features": 128, "out_features": 64, "batch": 1, "array_size": 16}
    schedules = [
        {"threads_per_block": 32, "unroll_factor": 1},
        {"threads_per_block": 64, "unroll_factor": 2},
        {"threads_per_block": 128, "unroll_factor": 4},
        {"threads_per_block": 256, "unroll_factor": 8},
    ]
    rows = []
    for idx, schedule in enumerate(schedules):
        rows.append(
            {
                "shape_triplet": {"in_features": 128, "hidden_features": 64, "out_features": 64},
                "layer_name": "fc1",
                "shape_used": dict(shape),
                "schedule": dict(schedule),
                "measured_latency_us": float(10.0 + idx),
            }
        )
    path = Path(tempfile.mkdtemp(prefix="utpu_refit_")) / "calib.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "machine_info": {},
                "environment": {},
                "shape_grid": [],
                "schedule_grid": schedules,
                "methodology": {},
                "per_point": rows,
            },
            f,
        )
    report = run_refit_from_existing_measurements(path)
    assert report["refit_from_existing_measurements"] is True
    assert report["measurement_timestamp_utc"] == "2026-01-01T00:00:00+00:00"
    assert {
        "fitted_coefficients",
        "aggregate_metrics",
        "ranking_diagnostics",
        "fit_objective",
        "pairwise_ordering",
        "per_point",
    }.issubset(report.keys())
    assert report["fit_objective"] == "mean_log_latency_mse_plus_pairwise_ordering"
    assert report["pairwise_ordering"]["pair_count"] > 0


def test_compiled_runtime_can_use_tuned_schedule():
    path = _temp_cache_path()
    _write_tiny_cache(path)
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda", use_tuned_schedule=True, autotune_cache_path=path)

    with torch.no_grad():
        y_compiled = compiled(x)
        y_ref = model(x)
    report = compiled.execution_report()
    max_abs = float(torch.max(torch.abs(y_compiled.detach().cpu() - y_ref.detach().cpu())).item())

    assert max_abs == 0.0
    assert report["fallback_ops"] == []
    assert any(trace["engine"] == "nvrtc_cuda_blocked_fc_tuned" for trace in report["op_traces"])


def test_fixed_schedule_still_available():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda", use_tuned_schedule=False)

    y = compiled(x)
    report = compiled.execution_report()

    assert tuple(y.shape) == (1, 2)
    assert report["fallback_ops"] == []
    assert any(trace["engine"] == "nvrtc_cuda_blocked_fc" for trace in report["op_traces"])


def run_all():
    test_autotuner_search_space_nonempty()
    test_autotuner_runs_one_shape()
    test_autotuner_preserves_correctness()
    test_best_schedule_cache_roundtrip()
    test_cost_model_pruning_selects_top_k_candidates()
    test_compiled_runtime_can_use_tuned_schedule()
    test_fixed_schedule_still_available()
    print("test_cuda_autotuner: PASS")


if __name__ == "__main__":
    run_all()
