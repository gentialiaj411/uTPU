import math
from typing import Any, Dict


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
    )
    return float(max(latency, 1e-6))
