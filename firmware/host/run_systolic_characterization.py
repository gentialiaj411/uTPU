from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from generate_batched_gemm_rtl_vectors import ARRAY_SIZE, generate_vectors
from run_rtl_batched_gemm_sim import _iverilog_run, _parse_perf_counter, _resolve_iverilog_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "systolic_characterization.json"
SCHEMA_VERSION = 1

# Single-tile streaming curve plus representative multi-tile control-bound
# points. The 16x16 family is the flagship streaming result; larger shapes are
# carried along to show the current top-level control limitation honestly.
CASES: Tuple[Tuple[int, int, int], ...] = (
    (16, 16, 1),
    (16, 16, 4),
    (16, 16, 16),
    (16, 16, 32),
    (16, 16, 64),
    (32, 32, 1),
    (32, 32, 16),
    (64, 64, 1),
    (64, 64, 16),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _items_per_word() -> int:
    # Shipping batched path is INT4 on a 16-bit buffer word.
    return 4


def _case_stem(out_features: int, in_features: int, batch_size: int) -> str:
    return f"systolic_char_o{out_features}_i{in_features}_b{batch_size}"


def _compute_model(case: Dict[str, Any]) -> Dict[str, Any]:
    out_blocks = int(case["out_blocks"])
    in_blocks = int(case["in_blocks"])
    batch_size = int(case["batch_size"])
    useful_macs = int(case["useful_macs"])
    pes = ARRAY_SIZE * ARRAY_SIZE

    # Secondary explanatory model only. No fitted parameters.
    per_tile_busy_cycles = int((2 * ARRAY_SIZE) + batch_size - 2)
    total_busy_cycles = int(out_blocks * in_blocks * per_tile_busy_cycles)
    pe_occupancy = useful_macs / float(pes * total_busy_cycles)
    return {
        "label": "secondary_zero_fit_weight_stationary_model",
        "per_tile_busy_cycles": per_tile_busy_cycles,
        "total_busy_cycles": total_busy_cycles,
        "pe_occupancy": pe_occupancy,
    }


def _characterize_case(
    out_features: int,
    in_features: int,
    batch_size: int,
    *,
    run_rtl: bool,
) -> Dict[str, Any]:
    stem = _case_stem(out_features, in_features, batch_size)
    vectors = generate_vectors(
        out_features=out_features,
        in_features=in_features,
        batch_size=batch_size,
        stem=stem,
        output_json=os.path.join("build", "test_vectors", f"{stem}.json"),
    )
    useful_macs = int(vectors["useful_macs"])
    pes = ARRAY_SIZE * ARRAY_SIZE
    measured: Dict[str, Optional[float]] = {
        "rtl_cycle_counter": None,
        "rtl_busy_counter": None,
        "busy_fraction": None,
        "pe_occupancy": None,
    }
    rtl_passed = None
    simulator_log_tail = None
    if run_rtl:
        ok, log = _iverilog_run(str(REPO_ROOT))
        rtl_passed = bool(ok)
        simulator_log_tail = "\n".join((log or "").splitlines()[-12:])
        measured["rtl_cycle_counter"] = _parse_perf_counter(log, "PERF_CYCLE_COUNTER")
        measured["rtl_busy_counter"] = _parse_perf_counter(log, "PERF_BUSY_COUNTER")
        if measured["rtl_cycle_counter"] and measured["rtl_busy_counter"]:
            measured["busy_fraction"] = (
                float(measured["rtl_busy_counter"]) / float(measured["rtl_cycle_counter"])
            )
            measured["pe_occupancy"] = (
                float(useful_macs) / float(pes * float(measured["rtl_busy_counter"]))
            )

    return {
        "shape": {"out_features": out_features, "in_features": in_features},
        "batch_size": batch_size,
        "array_size": ARRAY_SIZE,
        "items_per_word": _items_per_word(),
        "out_blocks": int(vectors["out_blocks"]),
        "in_blocks": int(vectors["in_blocks"]),
        "useful_macs": useful_macs,
        "fetch_bytes": len(vectors["expected_fetch_bytes"]),
        "rtl_sim_passed": rtl_passed,
        "measured": measured,
        "model": _compute_model(vectors),
        "control_flow": {
            "accumulate_runs_total": int(vectors["out_blocks"]) * int(vectors["in_blocks"]),
            "compute_en_windows": int(vectors["out_blocks"]),
            "accumulate_runs_per_compute_window": int(vectors["in_blocks"]),
            "note": (
                "compute_en is asserted by the accumulate RUN decode and stays high until the finalize "
                "RUN clears it; shapes with in_blocks > 1 therefore count multiple accumulate runs plus "
                "their intervening load/fetch/decode steps inside each compute_en window."
            ),
        },
        "simulator_log_tail": simulator_log_tail,
    }


def _shape_summary(shape_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(shape_rows, key=lambda row: int(row["batch_size"]))
    first = ordered[0]
    last = ordered[-1]
    first_busy = first["measured"]["rtl_busy_counter"]
    last_busy = last["measured"]["rtl_busy_counter"]
    first_occ = first["measured"]["pe_occupancy"]
    last_occ = last["measured"]["pe_occupancy"]
    first_frac = first["measured"]["busy_fraction"]
    last_frac = last["measured"]["busy_fraction"]
    batch_curve = [
        {
            "batch_size": int(row["batch_size"]),
            "rtl_busy_counter": row["measured"]["rtl_busy_counter"],
            "pe_occupancy": row["measured"]["pe_occupancy"],
            "busy_fraction": row["measured"]["busy_fraction"],
            "streaming_ceiling": (
                float(row["batch_size"]) / float((2 * ARRAY_SIZE) + int(row["batch_size"]))
                if int(row["out_blocks"]) == 1 and int(row["in_blocks"]) == 1
                else None
            ),
        }
        for row in ordered
    ]
    marginal_busy_curve = []
    for prev, curr in zip(ordered, ordered[1:]):
        prev_b = int(prev["batch_size"])
        curr_b = int(curr["batch_size"])
        prev_busy = prev["measured"]["rtl_busy_counter"]
        curr_busy = curr["measured"]["rtl_busy_counter"]
        marginal_busy_curve.append(
            {
                "from_batch_size": prev_b,
                "to_batch_size": curr_b,
                "delta_busy_cycles": (
                    int(curr_busy) - int(prev_busy)
                    if prev_busy is not None and curr_busy is not None
                    else None
                ),
                "delta_batch": curr_b - prev_b,
                "marginal_busy_cycles_per_added_batch": (
                    (float(curr_busy) - float(prev_busy)) / float(curr_b - prev_b)
                    if prev_busy is not None and curr_busy is not None and curr_b != prev_b
                    else None
                ),
            }
        )
    return {
        "shape": first["shape"],
        "out_blocks": first["out_blocks"],
        "in_blocks": first["in_blocks"],
        "batch_curve": batch_curve,
        "marginal_busy_curve": marginal_busy_curve,
        "b1_busy_cycles": first_busy,
        "max_b_busy_cycles": last_busy,
        "busy_cycles_growth_vs_b1": (
            float(last_busy) / float(first_busy)
            if first_busy not in (None, 0) and last_busy is not None
            else None
        ),
        "b1_pe_occupancy": first_occ,
        "max_b_pe_occupancy": last_occ,
        "pe_occupancy_growth_vs_b1": (
            float(last_occ) / float(first_occ)
            if first_occ not in (None, 0) and last_occ is not None
            else None
        ),
        "b1_busy_fraction": first_frac,
        "max_b_busy_fraction": last_frac,
        "asymmetry_explanation": (
            "This shape has in_blocks > 1, so each out-block keeps compute_en high across multiple "
            "accumulate RUNs and the intervening weight/input reload + fetch/decode steps until the finalize RUN. "
            "That inflates the B=1 busy baseline relative to the single-tile 16x16 case, so the B=1 -> B=16 "
            "busy-cycle multiplier is expected to be much smaller than 16x even though the per-RUN batch subruns "
            "do scale with B."
            if int(first["in_blocks"]) > 1
            else
            "This shape is a single blocked-FC tile (out_blocks=1, in_blocks=1), so B=1 contains one "
            "accumulate RUN before finalize. The B sweep is therefore closer to the number of batch subruns "
            "itself, and the busy-cycle multiplier stays near-linear."
        ),
    }


def build_artifact(*, skip_iverilog: bool = False) -> Dict[str, Any]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    rtl_available = bool((not skip_iverilog) and iv_bin and vv_bin)
    rows = [
        _characterize_case(out_features, in_features, batch_size, run_rtl=rtl_available)
        for (out_features, in_features, batch_size) in CASES
    ]

    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["shape"]["out_features"]), int(row["shape"]["in_features"]))
        grouped.setdefault(key, []).append(row)
    shape_summaries = [_shape_summary(grouped[key]) for key in sorted(grouped)]

    status = "ok" if rtl_available and all(row["rtl_sim_passed"] for row in rows) else "iverilog_unavailable"
    if rtl_available and any(row["rtl_sim_passed"] is False for row in rows):
        status = "failed"

    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_iso(),
        "status": status,
        "array_size": ARRAY_SIZE,
        "methodology": {
            "headline_metrics": {
                "pe_occupancy": "useful_macs / (ARRAY_SIZE^2 * rtl_busy_counter)",
                "busy_fraction": "rtl_busy_counter / rtl_cycle_counter",
            },
            "primary_source": (
                "RTL perf counters from rtl/top/top.sv: perf_busy_counter increments on each cycle with compute_en=1; "
                "perf_cycle_counter increments every cycle."
            ),
            "secondary_model": (
                "Zero-fit explanatory model only: per_tile_busy_cycles = 2*ARRAY_SIZE + B - 2. "
                "Reported to explain the trend, not to replace the RTL-measured numbers."
            ),
            "counter_provenance": (
                "For B>1, top.sv captures the real PE-array streaming accumulation under Icarus by sampling the "
                "bottom-row PE accumulators during the live valid window, with a terminal sample on compute_done. "
                "Additional busy cycles therefore come from genuine per-cycle PE-array/controller advancement under "
                "asserted compute_en, not from a behavioral dot-product shortcut."
            ),
        },
        "flagship_scope_note": (
            "The single-tile 16x16 family is the flagship streaming result. Multi-tile 32x32 and 64x64 remain "
            "control-bound by top-level blocked-FC orchestration and are reported as measured current behavior, "
            "not as a fully streamed multi-tile efficiency claim."
        ),
        "asymmetry_note": (
            "The 16x16 family is single-tile (out_blocks=1, in_blocks=1), so its B=1 busy baseline is one accumulate RUN "
            "before finalize and the B sweep stays near-linear. The 32x32 and 64x64 families have in_blocks > 1, so B=1 already "
            "counts multiple accumulate RUNs plus their intervening load/fetch/decode steps inside each compute_en window; their "
            "busy-cycle growth is therefore expected to be materially smaller than the 16x16 multiplier."
        ),
        "cases": rows,
        "shape_summaries": shape_summaries,
        "aggregate": {
            "rtl_available": rtl_available,
            "all_cases_passed": rtl_available and all(row["rtl_sim_passed"] for row in rows),
            "all_large_b_pe_occupancy_exceeds_b1": all(
                (summary["max_b_pe_occupancy"] is not None)
                and (summary["b1_pe_occupancy"] is not None)
                and (float(summary["max_b_pe_occupancy"]) > float(summary["b1_pe_occupancy"]))
                for summary in shape_summaries
            ) if rtl_available else None,
            "single_tile_streaming_curve": next(
                (
                    summary for summary in shape_summaries
                    if int(summary["shape"]["out_features"]) == 16 and int(summary["shape"]["in_features"]) == 16
                ),
                None,
            ),
        },
    }

    if not rtl_available:
        artifact["instructions"] = (
            "Install iverilog locally (Windows: bleyer.org/icarus; Linux/WSL2: apt-get install -y iverilog) "
            "and rerun `python firmware/host/run_systolic_characterization.py` to populate the RTL-measured metrics."
        )
    return artifact


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Phase 1 systolic characterization artifact")
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--skip-iverilog", action="store_true")
    args = parser.parse_args(argv)

    artifact = build_artifact(skip_iverilog=bool(args.skip_iverilog))
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"[run_systolic_characterization] status={artifact['status']} "
        f"rtl_available={artifact['aggregate']['rtl_available']} -> {out_path}"
    )
    return 0 if artifact["status"] in {"ok", "iverilog_unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
