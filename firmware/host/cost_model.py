import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


_DEFAULT_COEFFICIENTS_CUDA: Dict[str, float] = {
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
    "large_out_small_k_wave_tpb_efficiency_us": 0.25,
}


def _as_int(value: Any, name: str) -> int:
    v = int(value)
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def _shape_dims(shape: Dict[str, Any]) -> tuple[int, int, int, int]:
    out_features = shape.get("out_features", shape.get("M"))
    in_features = shape.get("in_features", shape.get("K"))
    batch = shape.get("batch", shape.get("N", 1))
    array_size = shape.get("array_size", 16)
    if out_features is None or in_features is None:
        raise ValueError("shape must provide out_features/in_features or M/K")
    return (
        _as_int(out_features, "out_features"),
        _as_int(in_features, "in_features"),
        _as_int(batch, "batch"),
        _as_int(array_size, "array_size"),
    )


def _schedule_params(schedule: Dict[str, Any]) -> tuple[int, int]:
    threads = int(schedule.get("threads_per_block", 128))
    unroll = int(schedule.get("unroll_factor", 1))
    if threads <= 0 or threads > 1024:
        raise ValueError(f"threads_per_block must be in 1..1024, got {threads}")
    if unroll not in (1, 2, 4, 8):
        raise ValueError(f"unroll_factor must be one of 1,2,4,8, got {unroll}")
    return threads, unroll


def _target_name(target: Any) -> str:
    if isinstance(target, dict):
        return str(target.get("name", "cuda")).strip().lower()
    return str(target or "cuda").strip().lower()


def _target_coefficients(target: Any) -> Dict[str, float]:
    name = _target_name(target)
    if name != "cuda":
        raise ValueError(f"predict_latency_us currently supports target='cuda', got '{name}'")
    coeffs = dict(_DEFAULT_COEFFICIENTS_CUDA)
    if isinstance(target, dict) and isinstance(target.get("cost_model_coefficients"), dict):
        for key, value in target["cost_model_coefficients"].items():
            k = str(key)
            if k in coeffs:
                coeffs[k] = float(value)
    return coeffs


def predict_latency_us(shape: Dict[str, Any], schedule: Dict[str, Any], target: Any = "cuda") -> float:
    """Analytical CUDA blocked-FC latency proxy calibrated for one-thread-per-output-row kernel geometry."""
    out_features, in_features, batch, array_size = _shape_dims(shape)
    threads_per_block, unroll_factor = _schedule_params(schedule)
    coeffs = _target_coefficients(target)

    out_padded = int(math.ceil(out_features / array_size) * array_size)
    in_padded = int(math.ceil(in_features / array_size) * array_size)

    weights_bytes = float(out_padded * in_padded)
    activations_bytes = float(batch * in_padded)
    outputs_bytes = float(batch * out_padded * 4)
    memory_kib = (weights_bytes + activations_bytes + outputs_bytes) / 1024.0
    cta_rows = min(float(out_padded), float(threads_per_block))
    cta_memory_kib = ((cta_rows * float(in_padded)) + activations_bytes + (float(batch) * cta_rows * 4.0)) / 1024.0

    warp_width = 32.0
    warps_per_block = max(1.0, threads_per_block / warp_width)
    active_warps = (out_padded / threads_per_block) * warps_per_block
    occupancy_proxy = min(1.0, active_warps / 8.0)
    underoccupancy_penalty = (1.0 - occupancy_proxy) ** 2

    out_tail = float(out_padded - out_features) / float(out_padded)
    in_tail = float(in_padded - in_features) / float(in_padded)
    tail_ratio = 0.5 * (out_tail + in_tail)
    unroll_log2 = math.log2(float(unroll_factor))
    max_unroll_log2 = math.log2(8.0)
    unroll_norm = unroll_log2 / max_unroll_log2
    unroll_gain = coeffs["unroll_gain_us"] * unroll_norm
    # May be inactive on aligned padded-K grids where this remainder is always zero.
    k_unroll_tail = float(in_padded % unroll_factor) / float(unroll_factor)
    unroll_k_tail_penalty = coeffs["unroll_k_tail_penalty_us"] * k_unroll_tail
    shape_compute_scale = math.log2(1.0 + (float(out_padded) * float(in_padded) / 1024.0))
    unroll_shape_interaction = coeffs["unroll_shape_interaction_us"] * unroll_norm * shape_compute_scale
    tpb_norm = math.log2(float(threads_per_block) / 32.0) / math.log2(8.0)
    small_out_ratio = max(0.0, 64.0 - float(out_padded)) / 64.0
    small_out_tpb_interaction = coeffs["small_out_tpb_interaction_us"] * small_out_ratio * tpb_norm
    small_out_unroll_interaction = coeffs["small_out_unroll_interaction_us"] * small_out_ratio * unroll_norm
    idle_thread_ratio = max(0.0, float(threads_per_block) - float(out_padded)) / float(threads_per_block)
    waves = float(math.ceil(float(out_padded) / float(threads_per_block)))
    wave_tpb_interaction = waves * tpb_norm
    small_out_idle_penalty = small_out_ratio * idle_thread_ratio
    large_k_unroll_gain = unroll_log2 * math.log2(1.0 + (float(in_padded) / 128.0))
    small_out_unroll_penalty = small_out_ratio * unroll_log2
    small_k_ratio = max(0.0, 1.0 - (float(in_padded) / 256.0))
    large_out_small_k_wave_tpb_efficiency = (
        tpb_norm * math.log2(1.0 + (float(out_padded) / 128.0)) / max(1.0, waves) * small_k_ratio
    )

    memory_term = coeffs["memory_us_per_kib"] * memory_kib
    cta_memory_term = coeffs["cta_memory_us_per_kib"] * cta_memory_kib
    latency = (
        coeffs["intercept_us"]
        + cta_memory_term
        + memory_term
        + coeffs["underoccupancy_penalty_us"] * underoccupancy_penalty
        + coeffs["tile_tail_penalty_us"] * tail_ratio
        - unroll_gain
        + unroll_k_tail_penalty
        - unroll_shape_interaction
        + small_out_tpb_interaction
        + small_out_unroll_interaction
        + coeffs["idle_thread_ratio_us"] * idle_thread_ratio
        + coeffs["wave_tpb_interaction_us"] * wave_tpb_interaction
        + coeffs["small_out_idle_penalty_us"] * small_out_idle_penalty
        - coeffs["large_k_unroll_gain_us"] * large_k_unroll_gain
        + coeffs["small_out_unroll_penalty_us"] * small_out_unroll_penalty
        - coeffs["large_out_small_k_wave_tpb_efficiency_us"] * large_out_small_k_wave_tpb_efficiency
    )
    return float(max(latency, 1e-6))


@dataclass(frozen=True)
class CostModelChoice:
    """The cost model's predicted-best candidate plus selection metadata.

    `select` returns one of these per call. The fields are designed so a
    compiler can both *use* the choice (`schedule`) and *reason about its
    confidence* (`margin_pct`, `confidence`) without re-running the model.

    `score` is `-predicted_latency_us` (higher is better) so callers can
    compare choices across calls; `predicted_latency_us` is the raw
    cost-model output in microseconds.

    `confidence` is a deterministic margin-based heuristic, **not** a
    calibrated probability. It is `1 - exp(-margin_pct / 5)` clipped to
    `[0, 1]`. A margin of 5% gives ~0.63, 10% gives ~0.86, 1% gives ~0.18,
    and an exact tie gives 0.0. Use it to gate fallbacks, not as a
    probability of being correct.
    """

    schedule: Dict[str, int]
    predicted_latency_us: float
    score: float
    rank: int
    candidates_considered: int
    runner_up_schedule: Optional[Dict[str, int]]
    runner_up_predicted_latency_us: Optional[float]
    margin_us: float
    margin_pct: float
    confidence: float
    target_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule": dict(self.schedule),
            "predicted_latency_us": float(self.predicted_latency_us),
            "score": float(self.score),
            "rank": int(self.rank),
            "candidates_considered": int(self.candidates_considered),
            "runner_up_schedule": dict(self.runner_up_schedule) if self.runner_up_schedule is not None else None,
            "runner_up_predicted_latency_us": (
                float(self.runner_up_predicted_latency_us)
                if self.runner_up_predicted_latency_us is not None
                else None
            ),
            "margin_us": float(self.margin_us),
            "margin_pct": float(self.margin_pct),
            "confidence": float(self.confidence),
            "target_name": self.target_name,
        }


def _confidence_from_margin_pct(margin_pct: float) -> float:
    if margin_pct <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - math.exp(-margin_pct / 5.0))))


def select(
    shape: Dict[str, Any],
    candidates: Iterable[Dict[str, Any]],
    target: Any = "cuda",
) -> CostModelChoice:
    """Choose one candidate plan via the calibrated cost model.

    Promotes the cost model from a pruning filter into a real compilation
    decision component: scores every input candidate with
    `predict_latency_us`, sorts by predicted latency, and commits to the
    minimum. Tie-break is stable on insertion order.

    The caller is responsible for normalizing candidate dicts (the schedule
    keys / value types must already match what `predict_latency_us` expects).
    This keeps `cost_model.py` free of backend imports.

    Raises `ValueError` if `candidates` is empty or `shape` is missing the
    keys `predict_latency_us` requires.
    """
    candidate_list: List[Dict[str, Any]] = [dict(c) for c in candidates]
    if not candidate_list:
        raise ValueError("select() requires at least one candidate")

    scored: List[Dict[str, Any]] = []
    for index, schedule in enumerate(candidate_list):
        predicted_us = float(predict_latency_us(shape, schedule, target=target))
        scored.append(
            {
                "index": index,
                "schedule": schedule,
                "predicted_latency_us": predicted_us,
            }
        )

    scored.sort(key=lambda item: (item["predicted_latency_us"], item["index"]))
    chosen = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None

    predicted_us = float(chosen["predicted_latency_us"])
    if runner_up is not None:
        runner_us = float(runner_up["predicted_latency_us"])
        margin_us = runner_us - predicted_us
        margin_pct = (margin_us / predicted_us) * 100.0 if predicted_us > 0.0 else 0.0
    else:
        runner_us = None
        margin_us = 0.0
        margin_pct = 0.0

    return CostModelChoice(
        schedule=dict(chosen["schedule"]),
        predicted_latency_us=predicted_us,
        score=-predicted_us,
        rank=1,
        candidates_considered=len(candidate_list),
        runner_up_schedule=dict(runner_up["schedule"]) if runner_up is not None else None,
        runner_up_predicted_latency_us=runner_us,
        margin_us=float(margin_us),
        margin_pct=float(margin_pct),
        confidence=_confidence_from_margin_pct(float(margin_pct)),
        target_name=_target_name(target),
    )
