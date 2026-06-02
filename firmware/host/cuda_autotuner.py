import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from cost_model import predict_latency_us
from cuda_blocked_fc_backend import (
    CUDABlockedFCExecutor,
    DEFAULT_CUDA_SCHEDULE_PARAMS,
    detect_cuda_environment,
    normalize_cuda_schedule_params,
)
from lowering_types import BlockedFCLoweringRequest


DEFAULT_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "reports", "cuda_autotune_results.json")
)
DEFAULT_COST_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "build", "reports", "cost_model_calibration.json")
)
_RANK_TIE_REL_EPS = 1e-3
_RANK_TIE_ABS_EPS_US = 1e-3


@dataclass(frozen=True)
class CUDABlockedFCScheduleParams:
    threads_per_block: int = 128
    unroll_factor: int = 1
    use_smem: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return normalize_cuda_schedule_params(
            {
                "threads_per_block": self.threads_per_block,
                "unroll_factor": self.unroll_factor,
                "use_smem": self.use_smem,
            }
        )


@dataclass(frozen=True)
class CUDATuningSearchSpace:
    threads_per_block: Tuple[int, ...] = (32, 64, 128, 256)
    unroll_factor: Tuple[int, ...] = (1, 2, 4, 8)

    def candidates(self, *, include_smem_variants: bool = False) -> List[CUDABlockedFCScheduleParams]:
        smem_values = (False, True) if include_smem_variants else (False,)
        return [
            CUDABlockedFCScheduleParams(threads_per_block=t, unroll_factor=u, use_smem=use_smem)
            for t in self.threads_per_block
            for u in self.unroll_factor
            for use_smem in smem_values
        ]

    def schedule_candidates(self) -> List[CUDABlockedFCScheduleParams]:
        """Full autotuner grid including ``use_smem`` True/False for each schedule."""
        return self.candidates(include_smem_variants=True)

    def schema(self) -> Dict[str, Any]:
        return {
            "threads_per_block": {
                "type": "int",
                "values": list(self.threads_per_block),
                "description": "CUDA blockDim.x for one-output-row-per-thread blocked FC kernel.",
            },
            "unroll_factor": {
                "type": "int",
                "values": list(self.unroll_factor),
                "description": "Manual unroll factor for the inner K loop in the generated NVRTC kernel.",
            },
            "use_smem": {
                "type": "bool",
                "values": [False, True],
                "description": "Load input vector x into shared memory once per block (GEMV).",
            },
        }


@dataclass
class CUDATuningResult:
    shape: Dict[str, int]
    dtype_mode: str
    target: str
    fixed_schedule: Dict[str, int]
    best_schedule: Dict[str, int]
    fixed_latency_ms: Optional[float]
    best_latency_ms: Optional[float]
    fixed_end_to_end_ms: Optional[float]
    best_end_to_end_ms: Optional[float]
    improvement_pct: Optional[float]
    max_abs_error: int
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    pruned_candidates: List[Dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0
    profiled_candidate_count: int = 0
    pruning: Optional[Dict[str, Any]] = None
    executed: bool = True
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": dict(self.shape),
            "dtype_mode": self.dtype_mode,
            "target": self.target,
            "fixed_schedule": dict(self.fixed_schedule),
            "best_schedule": dict(self.best_schedule),
            "fixed_latency_ms": self.fixed_latency_ms,
            "best_latency_ms": self.best_latency_ms,
            "fixed_end_to_end_ms": self.fixed_end_to_end_ms,
            "best_end_to_end_ms": self.best_end_to_end_ms,
            "improvement_pct": self.improvement_pct,
            "max_abs_error": self.max_abs_error,
            "candidates": list(self.candidates),
            "pruned_candidates": list(self.pruned_candidates),
            "candidate_count": self.candidate_count,
            "profiled_candidate_count": self.profiled_candidate_count,
            "pruning": dict(self.pruning) if self.pruning is not None else None,
            "executed": self.executed,
            "reason": self.reason,
        }


def default_search_space() -> CUDATuningSearchSpace:
    return CUDATuningSearchSpace()


def make_cache_key(
    out_features: int,
    in_features: int,
    array_size: int = 16,
    dtype_mode: str = "int4_i32",
    target: str = "cuda",
) -> str:
    return (
        f"{target}|dtype={dtype_mode}|M={int(out_features)}|N=1|K={int(in_features)}|"
        f"array_size={int(array_size)}"
    )


def load_schedule_cache(path: str = DEFAULT_CACHE_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"version": 1, "backend": "cuda_blocked_fc", "results": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data.get("schedule_cache"), dict):
        data = data["schedule_cache"]
    data.setdefault("version", 1)
    data.setdefault("backend", "cuda_blocked_fc")
    if not isinstance(data.get("results"), dict):
        data["results"] = {}
    return data


def save_schedule_cache(cache: Dict[str, Any], path: str = DEFAULT_CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def lookup_best_schedule(
    out_features: int,
    in_features: int,
    array_size: int = 16,
    dtype_mode: str = "int4_i32",
    target: str = "cuda",
    path: str = DEFAULT_CACHE_PATH,
) -> Optional[Dict[str, int]]:
    cache = load_schedule_cache(path)
    key = make_cache_key(out_features, in_features, array_size, dtype_mode, target)
    entry = cache.get("results", {}).get(key)
    if not entry:
        return None
    return normalize_cuda_schedule_params(entry["best_schedule"])


def load_cost_model_target(path: str = DEFAULT_COST_MODEL_PATH) -> Any:
    if not path or not os.path.exists(path):
        return "cuda"
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    coeffs = report.get("fitted_coefficients")
    if not isinstance(coeffs, dict):
        return "cuda"
    return {"name": "cuda", "cost_model_coefficients": coeffs}


def rank_candidates_by_cost_model(
    out_features: int,
    in_features: int,
    array_size: int,
    candidates: Iterable[Dict[str, int]],
    target: Any = "cuda",
) -> List[Dict[str, Any]]:
    shape = {
        "out_features": int(out_features),
        "in_features": int(in_features),
        "batch": 1,
        "array_size": int(array_size),
        "apply_quant": True,
    }
    ranked: List[Dict[str, Any]] = []
    for candidate in candidates:
        schedule = normalize_cuda_schedule_params(candidate)
        predicted_us = predict_latency_us(shape, schedule, target=target)
        ranked.append(
            {
                "schedule": schedule,
                "predicted_latency_us": float(predicted_us),
            }
        )
    if ranked:
        best_us = min(float(item["predicted_latency_us"]) for item in ranked)
        tie_eps_us = max(_RANK_TIE_ABS_EPS_US, best_us * _RANK_TIE_REL_EPS)
    else:
        best_us = 0.0
        tie_eps_us = _RANK_TIE_ABS_EPS_US
    for item in ranked:
        predicted_us = float(item["predicted_latency_us"])
        item["tie_bucket"] = int((predicted_us - best_us) / tie_eps_us) if ranked else 0
    ranked.sort(
        key=lambda item: (
            int(item["tie_bucket"]),
            item["schedule"]["threads_per_block"],
            item["schedule"]["unroll_factor"],
            int(bool(item["schedule"].get("use_smem", False))),
            float(item["predicted_latency_us"]),
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["cost_model_rank"] = int(rank)
    return ranked


def select_pruned_candidates(
    out_features: int,
    in_features: int,
    array_size: int,
    candidates: Iterable[Dict[str, int]],
    top_k: Optional[int],
    target: Any = "cuda",
) -> Tuple[List[Dict[str, int]], List[Dict[str, Any]], Dict[str, Any]]:
    normalized = [normalize_cuda_schedule_params(c) for c in candidates]
    policy_meta = {
        "requested_top_k": int(top_k) if top_k is not None else None,
        "tie_margin_fraction": 0.01,
        "small_space_threshold": None,
        "best_predicted_latency_us": None,
        "tie_margin_kept_count": 0,
        "unroll_family_kept_count": 0,
        "thread_block_family_kept_count": 0,
        "thread_block_family_condition": "selected_count_le_top_k_and_min_shape_dim_le_16",
        "selected_candidate_count": len(normalized),
        "search_reduction_x": 1.0,
        "used_policy": False,
    }
    if top_k is None or int(top_k) <= 0 or int(top_k) >= len(normalized):
        return normalized, [], policy_meta
    smem_variants = len({bool(c.get("use_smem", False)) for c in normalized})
    small_space_threshold = max(6, int(top_k)) * max(1, smem_variants)
    policy_meta["small_space_threshold"] = int(small_space_threshold)
    if len(normalized) <= small_space_threshold:
        return normalized, [], policy_meta
    ranked = rank_candidates_by_cost_model(out_features, in_features, array_size, normalized, target=target)
    k = int(top_k)
    keep = ranked[:k]
    margin_threshold = 0.01
    policy_meta["used_policy"] = True
    tie_kept = 0
    if ranked:
        best_us = float(ranked[0]["predicted_latency_us"])
        policy_meta["best_predicted_latency_us"] = float(best_us)
        if best_us > 0.0:
            for item in ranked[k:]:
                delta = (float(item["predicted_latency_us"]) - best_us) / best_us
                if delta <= margin_threshold:
                    keep.append(item)
                    tie_kept += 1
                else:
                    break
    seen_unroll = {int(item["schedule"]["unroll_factor"]) for item in keep}
    unroll_family_kept = 0
    for item in ranked:
        unroll = int(item["schedule"]["unroll_factor"])
        if unroll not in seen_unroll:
            keep.append(item)
            seen_unroll.add(unroll)
            unroll_family_kept += 1
    thread_family_kept = 0
    unique_thread_blocks = {
        int(item["schedule"]["threads_per_block"]) for item in keep
    }
    keep_thread_diversity = (
        len(unique_thread_blocks) <= k and min(int(out_features), int(in_features)) <= 16
    )
    if keep_thread_diversity:
        seen_threads = {int(item["schedule"]["threads_per_block"]) for item in keep}
        for item in ranked:
            threads = int(item["schedule"]["threads_per_block"])
            if threads not in seen_threads:
                keep.append(item)
                seen_threads.add(threads)
                thread_family_kept += 1
    keep_schedules = []
    seen_schedule = set()
    for item in keep:
        key = (
            int(item["schedule"]["threads_per_block"]),
            int(item["schedule"]["unroll_factor"]),
            int(bool(item["schedule"].get("use_smem", False))),
        )
        if key in seen_schedule:
            continue
        seen_schedule.add(key)
        keep_schedules.append(dict(item["schedule"]))
    pruned = [
        item
        for item in ranked
        if (
            int(item["schedule"]["threads_per_block"]),
            int(item["schedule"]["unroll_factor"]),
            int(bool(item["schedule"].get("use_smem", False))),
        )
        not in seen_schedule
    ]
    policy_meta["tie_margin_kept_count"] = int(tie_kept)
    policy_meta["unroll_family_kept_count"] = int(unroll_family_kept)
    policy_meta["thread_block_family_kept_count"] = int(thread_family_kept)
    policy_meta["selected_candidate_count"] = int(len(keep_schedules))
    policy_meta["search_reduction_x"] = float(len(normalized) / max(1, len(keep_schedules)))
    return keep_schedules, pruned, policy_meta


def _target_id(executor: CUDABlockedFCExecutor) -> str:
    env = detect_cuda_environment()
    if not env.runtime_available:
        return "cuda:unavailable"
    try:
        cuda, _ = executor._load_cuda_bindings()
        executor._ensure_context()
        err, dev = cuda.cuDeviceGet(0)
        executor._check_cuda(err, "cuDeviceGet")
        name_buf = bytearray(128)
        err, = cuda.cuDeviceGetName(name_buf, len(name_buf), dev)
        executor._check_cuda(err, "cuDeviceGetName")
        name = bytes(name_buf).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        return f"cuda:{name}"
    except Exception:
        return "cuda"


def _make_request(out_features: int, in_features: int, array_size: int, seed: int = 0) -> BlockedFCLoweringRequest:
    rng = np.random.default_rng(seed)
    weights = rng.integers(-8, 8, size=(out_features, in_features), dtype=np.int8)
    activations = rng.integers(-8, 8, size=(in_features,), dtype=np.int8)
    return BlockedFCLoweringRequest(
        weights_int4=weights,
        activations_int4=activations,
        out_features=out_features,
        in_features=in_features,
        array_size=array_size,
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )


def _measure_candidate(
    executor: CUDABlockedFCExecutor,
    request: BlockedFCLoweringRequest,
    params: Dict[str, int],
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    for _ in range(warmup):
        warm = executor.execute(request, schedule_params=params)
        if not warm.get("executed", False):
            return {
                "schedule": dict(params),
                "executed": False,
                "reason": warm.get("reason", "unknown CUDA execution failure"),
            }

    samples = []
    end_to_end = []
    max_abs = 0
    for _ in range(iters):
        result = executor.execute(request, schedule_params=params)
        if not result.get("executed", False):
            return {
                "schedule": dict(params),
                "executed": False,
                "reason": result.get("reason", "unknown CUDA execution failure"),
            }
        max_abs = max(max_abs, int(result.get("max_abs_diff_vs_numpy_reference", 0)))
        samples.append(float(result["kernel_time_ms"]))
        end_to_end.append(float(result["end_to_end_time_ms"]))

    return {
        "schedule": dict(params),
        "executed": True,
        "kernel_median_ms": float(statistics.median(samples)),
        "kernel_mean_ms": float(statistics.fmean(samples)),
        "end_to_end_median_ms": float(statistics.median(end_to_end)),
        "end_to_end_mean_ms": float(statistics.fmean(end_to_end)),
        "max_abs_error": int(max_abs),
    }


def tune_blocked_fc_shape(
    out_features: int,
    in_features: int,
    array_size: int = 16,
    search_space: Optional[CUDATuningSearchSpace] = None,
    warmup: int = 2,
    iters: int = 5,
    cache_path: str = DEFAULT_CACHE_PATH,
    seed: int = 0,
    max_candidates: Optional[int] = None,
    prune_top_k: Optional[int] = None,
    cost_model_target: Any = None,
    cost_model_path: str = DEFAULT_COST_MODEL_PATH,
) -> CUDATuningResult:
    env = detect_cuda_environment()
    fixed = normalize_cuda_schedule_params(DEFAULT_CUDA_SCHEDULE_PARAMS)
    shape = {
        "M": int(out_features),
        "N": 1,
        "K": int(in_features),
        "array_size": int(array_size),
    }
    if not env.runtime_available:
        return CUDATuningResult(
            shape=shape,
            dtype_mode="int4_i32",
            target="cuda:unavailable",
            fixed_schedule=fixed,
            best_schedule=fixed,
            fixed_latency_ms=None,
            best_latency_ms=None,
            fixed_end_to_end_ms=None,
            best_end_to_end_ms=None,
            improvement_pct=None,
            max_abs_error=0,
            candidate_count=0,
            profiled_candidate_count=0,
            executed=False,
            reason=env.reason,
        )

    executor = CUDABlockedFCExecutor(verbose=False)
    request = _make_request(out_features, in_features, array_size, seed=seed)
    target = _target_id(executor)
    space = search_space or default_search_space()
    all_candidates = [c.to_dict() for c in space.schedule_candidates()]
    if max_candidates is not None:
        all_candidates = all_candidates[: int(max_candidates)]
    target_model = cost_model_target if cost_model_target is not None else load_cost_model_target(cost_model_path)
    candidates, pruned_candidates, policy_meta = select_pruned_candidates(
        out_features=out_features,
        in_features=in_features,
        array_size=array_size,
        candidates=all_candidates,
        top_k=prune_top_k,
        target=target_model,
    )
    pruning = None
    if pruned_candidates:
        pruning = {
            "method": "cost_model_top_k_margin_diversity",
            "top_k": int(prune_top_k),
            "candidate_count": int(len(all_candidates)),
            "profiled_candidate_count": int(len(candidates)),
            "search_reduction_x": float(len(all_candidates) / max(1, len(candidates))),
            "cost_model_path": cost_model_path,
            "fallback_small_space_threshold": int(max(6, int(prune_top_k))),
            "tie_margin_fraction": 0.01,
            "family_diversity": "keep_best_per_unroll_factor_and_threads_per_block",
            "best_predicted_latency_us": policy_meta.get("best_predicted_latency_us"),
            "tie_margin_kept_count": int(policy_meta.get("tie_margin_kept_count", 0)),
            "unroll_family_kept_count": int(policy_meta.get("unroll_family_kept_count", 0)),
            "thread_block_family_kept_count": int(policy_meta.get("thread_block_family_kept_count", 0)),
            "selected_candidate_count": int(policy_meta.get("selected_candidate_count", len(candidates))),
            "actual_search_reduction_x": float(policy_meta.get("search_reduction_x", float(len(all_candidates) / max(1, len(candidates))))),
        }

    measured = []
    for params in candidates:
        measured.append(_measure_candidate(executor, request, params, warmup=warmup, iters=iters))

    valid = [m for m in measured if m.get("executed") and int(m.get("max_abs_error", 1)) == 0]
    fixed_result = next((m for m in measured if m.get("schedule") == fixed and m.get("executed")), None)
    if fixed_result is None:
        fixed_result = _measure_candidate(executor, request, fixed, warmup=warmup, iters=iters)
        measured.append(fixed_result)
        if fixed_result.get("executed") and int(fixed_result.get("max_abs_error", 1)) == 0:
            valid.append(fixed_result)

    if not valid:
        reason = "; ".join(str(m.get("reason", "invalid candidate")) for m in measured if not m.get("executed"))
        return CUDATuningResult(
            shape=shape,
            dtype_mode="int4_i32",
            target=target,
            fixed_schedule=fixed,
            best_schedule=fixed,
            fixed_latency_ms=None,
            best_latency_ms=None,
            fixed_end_to_end_ms=None,
            best_end_to_end_ms=None,
            improvement_pct=None,
            max_abs_error=0,
            candidates=measured,
            pruned_candidates=pruned_candidates,
            candidate_count=len(all_candidates),
            profiled_candidate_count=len(measured),
            pruning=pruning,
            executed=False,
            reason=reason or "no correct tuning candidate found",
        )

    best = min(valid, key=lambda m: float(m["kernel_median_ms"]))
    fixed_latency = float(fixed_result["kernel_median_ms"]) if fixed_result and fixed_result.get("executed") else None
    best_latency = float(best["kernel_median_ms"])
    improvement = None
    if fixed_latency is not None and fixed_latency > 0.0:
        improvement = float(((fixed_latency - best_latency) / fixed_latency) * 100.0)

    result = CUDATuningResult(
        shape=shape,
        dtype_mode="int4_i32",
        target=target,
        fixed_schedule=fixed,
        best_schedule=normalize_cuda_schedule_params(best["schedule"]),
        fixed_latency_ms=fixed_latency,
        best_latency_ms=best_latency,
        fixed_end_to_end_ms=float(fixed_result["end_to_end_median_ms"]) if fixed_result and fixed_result.get("executed") else None,
        best_end_to_end_ms=float(best["end_to_end_median_ms"]),
        improvement_pct=improvement,
        max_abs_error=int(best.get("max_abs_error", 0)),
        candidates=measured,
        pruned_candidates=pruned_candidates,
        candidate_count=len(all_candidates),
        profiled_candidate_count=len(measured),
        pruning=pruning,
        executed=True,
    )

    cache = load_schedule_cache(cache_path)
    key = make_cache_key(out_features, in_features, array_size, result.dtype_mode, "cuda")
    cache["results"][key] = result.to_dict()
    save_schedule_cache(cache, cache_path)
    return result


def tune_many_shapes(
    shapes: Iterable[Tuple[str, int, int]],
    array_size: int = 16,
    warmup: int = 2,
    iters: int = 5,
    cache_path: str = DEFAULT_CACHE_PATH,
    max_candidates: Optional[int] = None,
    prune_top_k: Optional[int] = None,
    cost_model_path: str = DEFAULT_COST_MODEL_PATH,
) -> Dict[str, Any]:
    space = default_search_space()
    results = []
    for idx, (name, out_features, in_features) in enumerate(shapes):
        tuned = tune_blocked_fc_shape(
            out_features=out_features,
            in_features=in_features,
            array_size=array_size,
            search_space=space,
            warmup=warmup,
            iters=iters,
            cache_path=cache_path,
            seed=idx,
            max_candidates=max_candidates,
            prune_top_k=prune_top_k,
            cost_model_path=cost_model_path,
        )
        entry = tuned.to_dict()
        entry["shape_name"] = name
        results.append(entry)
    report = {
        "search_space": space.schema(),
        "cache_path": cache_path,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prune_top_k": prune_top_k,
        "cost_model_path": cost_model_path if prune_top_k is not None else None,
        "results": results,
        "schedule_cache": {
            "version": 1,
            "backend": "cuda_blocked_fc",
            "results": {},
        },
    }
    for item in results:
        shape = item["shape"]
        key = make_cache_key(shape["M"], shape["K"], shape["array_size"], item["dtype_mode"], "cuda")
        report["schedule_cache"]["results"][key] = item
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report
