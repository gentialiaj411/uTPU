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

# Single-tile streaming curve plus multi-tile points. The 16x16 family remains
# the flagship anchor. Larger shapes are reported with an opt-in hoisted-tile
# payload path so compute-span duty-cycle can be compared honestly against the
# serialized baseline that still refills buffer words between accumulate runs.
CASES: Tuple[Tuple[int, int, int], ...] = (
    (16, 16, 1),
    (16, 16, 4),
    (16, 16, 16),
    (16, 16, 32),
    (16, 16, 64),
    (32, 32, 1),
    (32, 32, 4),
    (32, 32, 16),
    (32, 32, 32),
    (64, 64, 1),
    (64, 64, 4),
    (64, 64, 16),
    (64, 64, 32),
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
    hoist_tile_payloads: bool,
) -> Dict[str, Any]:
    stem = _case_stem(out_features, in_features, batch_size)
    if hoist_tile_payloads:
        stem = f"{stem}_hoisted"
    vectors = generate_vectors(
        out_features=out_features,
        in_features=in_features,
        batch_size=batch_size,
        stem=stem,
        output_json=os.path.join("build", "test_vectors", f"{stem}.json"),
        hoist_tile_payloads=hoist_tile_payloads,
    )
    useful_macs = int(vectors["useful_macs"])
    pes = ARRAY_SIZE * ARRAY_SIZE
    measured: Dict[str, Optional[float]] = {
        "rtl_cycle_counter": None,
        "rtl_busy_counter": None,
        "total_program_cycles": None,
        "busy_fraction": None,
        "pe_occupancy": None,
        "compute_busy_cycles": None,
        "compute_span_cycles": None,
        "compute_span_duty_cycle": None,
    }
    rtl_passed = None
    simulator_log_tail = None
    if run_rtl:
        ok, log = _iverilog_run(str(REPO_ROOT))
        rtl_passed = bool(ok)
        simulator_log_tail = "\n".join((log or "").splitlines()[-12:])
        measured["rtl_cycle_counter"] = _parse_perf_counter(log, "PERF_CYCLE_COUNTER")
        measured["rtl_busy_counter"] = _parse_perf_counter(log, "PERF_BUSY_COUNTER")
        measured["total_program_cycles"] = _parse_perf_counter(log, "TOTAL_PROGRAM_CYCLES")
        measured["compute_busy_cycles"] = _parse_perf_counter(log, "COMPUTE_BUSY_CYCLES")
        measured["compute_span_cycles"] = _parse_perf_counter(log, "COMPUTE_SPAN_CYCLES")
        if measured["rtl_cycle_counter"] and measured["rtl_busy_counter"]:
            measured["busy_fraction"] = (
                float(measured["rtl_busy_counter"]) / float(measured["rtl_cycle_counter"])
            )
            measured["pe_occupancy"] = (
                float(useful_macs) / float(pes * float(measured["rtl_busy_counter"]))
            )
        if measured["compute_busy_cycles"] and measured["compute_span_cycles"]:
            measured["compute_span_duty_cycle"] = (
                float(measured["compute_busy_cycles"]) / float(measured["compute_span_cycles"])
            )

    serial_baseline = None
    if run_rtl and hoist_tile_payloads:
        baseline_stem = f"{_case_stem(out_features, in_features, batch_size)}_serial_baseline"
        baseline_vectors = generate_vectors(
            out_features=out_features,
            in_features=in_features,
            batch_size=batch_size,
            stem=baseline_stem,
            output_json=os.path.join("build", "test_vectors", f"{baseline_stem}.json"),
            hoist_tile_payloads=False,
        )
        ok, log = _iverilog_run(str(REPO_ROOT))
        serial_baseline = {
            "hoist_tile_payloads": False,
            "rtl_sim_passed": bool(ok),
            "program_words": int(baseline_vectors["program_words"]),
            "rtl_cycle_counter": _parse_perf_counter(log, "PERF_CYCLE_COUNTER"),
            "rtl_busy_counter": _parse_perf_counter(log, "PERF_BUSY_COUNTER"),
            "total_program_cycles": _parse_perf_counter(log, "TOTAL_PROGRAM_CYCLES"),
            "compute_busy_cycles": _parse_perf_counter(log, "COMPUTE_BUSY_CYCLES"),
            "compute_span_cycles": _parse_perf_counter(log, "COMPUTE_SPAN_CYCLES"),
        }
        if serial_baseline["compute_busy_cycles"] and serial_baseline["compute_span_cycles"]:
            serial_baseline["compute_span_duty_cycle"] = (
                float(serial_baseline["compute_busy_cycles"])
                / float(serial_baseline["compute_span_cycles"])
            )
        else:
            serial_baseline["compute_span_duty_cycle"] = None

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
        "hoist_tile_payloads": bool(vectors.get("hoist_tile_payloads", False)),
        "measured": measured,
        "serial_baseline": serial_baseline,
        "model": _compute_model(vectors),
        "control_flow": {
            "accumulate_runs_total": int(vectors["out_blocks"]) * int(vectors["in_blocks"]),
            "out_blocks": int(vectors["out_blocks"]),
            "in_blocks": int(vectors["in_blocks"]),
            "note": (
                "Single-tile 16x16 rows keep the legacy serialized payload path. Multi-tile rows may hoist "
                "weight/input tile payloads into distinct buffer slots before the first compute window so the "
                "compute-span duty metric can isolate inter-tile refill cost honestly without redefining the "
                "full rtl_cycle_counter denominator."
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
            "total_program_cycles": row["measured"]["total_program_cycles"],
            "pe_occupancy": row["measured"]["pe_occupancy"],
            "busy_fraction": row["measured"]["busy_fraction"],
            "compute_span_duty_cycle": row["measured"]["compute_span_duty_cycle"],
            "serial_baseline_compute_span_duty_cycle": (
                row["serial_baseline"]["compute_span_duty_cycle"]
                if row.get("serial_baseline") is not None
                else None
            ),
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
        "b1_compute_span_duty_cycle": first["measured"]["compute_span_duty_cycle"],
        "max_b_compute_span_duty_cycle": last["measured"]["compute_span_duty_cycle"],
        "b1_serial_baseline_compute_span_duty_cycle": (
            first["serial_baseline"]["compute_span_duty_cycle"]
            if first.get("serial_baseline") is not None
            else None
        ),
        "max_b_serial_baseline_compute_span_duty_cycle": (
            last["serial_baseline"]["compute_span_duty_cycle"]
            if last.get("serial_baseline") is not None
            else None
        ),
        "asymmetry_explanation": (
            "This shape has multiple accumulate RUNs per full GEMM. The optimized rows therefore focus on "
            "compute-span duty-cycle before/after hoisting tile payloads out of the inner loop, while the "
            "rtl_cycle_counter still reports the full end-to-end serialized control cost honestly."
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
        _characterize_case(
            out_features,
            in_features,
            batch_size,
            run_rtl=rtl_available,
            hoist_tile_payloads=bool(out_features > ARRAY_SIZE or in_features > ARRAY_SIZE),
        )
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
                "total_program_cycles": (
                    "MAGIC_START to HALT wall-clock cycles from top.sv::perf_program_cycle_counter, "
                    "exported as the 4th 64-bit word of MAGIC_READ_PERF (0xA4). Includes accumulate, "
                    "inter-tile LOAD gaps, and finalize wait_clear. Excludes UART upload and post-HALT idle."
                ),
                "pe_occupancy": (
                    "useful_macs / (ARRAY_SIZE^2 * rtl_busy_counter). Array-utilization metric only: "
                    "rtl_busy_counter / compute span exclude finalize wait_clear, so pe_occupancy is NOT "
                    "a throughput or wall-clock metric."
                ),
                "busy_fraction": "rtl_busy_counter / rtl_cycle_counter (free-running cycle includes upload)",
                "compute_span_duty_cycle": (
                    "compute_busy_cycles / compute_span_cycles, where compute_span runs from the first "
                    "accumulate RUN start to the last accumulate RUN done and therefore includes inter-tile "
                    "LOAD/refill gaps but excludes finalize wait_clear and UART upload/fetch traffic. "
                    "Array-utilization / duty metric, not end-to-end throughput."
                ),
            },
            "primary_source": (
                "RTL perf counters from rtl/top/top.sv via MAGIC_READ_PERF (0xA4): free-running "
                "perf_cycle_counter, busy-window perf_busy_counter, halt-count perf_program_count, and "
                "START->HALT perf_program_cycle_counter (total_program_cycles)."
            ),
            "secondary_model": (
                "Zero-fit explanatory model only: per_tile_busy_cycles = 2*ARRAY_SIZE + B - 2. "
                "Reported to explain the trend, not to replace the RTL-measured numbers."
            ),
            "counter_provenance": (
                "For B>1, top.sv captures the real PE-array streaming accumulation under Icarus by sampling the "
                "bottom-row PE accumulators during the live valid window, with a terminal sample on compute_done. "
                "Additional busy cycles therefore come from genuine per-cycle PE-array/controller advancement inside "
                "the active accumulate/finalize window, not from a behavioral dot-product shortcut or a counter-only "
                "annotation of inter-tile refill control."
            ),
            "measurement_integrity_note": (
                "Step 2b pipelined requant added measurable wall-clock cost (+60 cycles at B=32 on the "
                "QUANTIZER_LANES axis) that pe_occupancy / compute_span_duty_cycle / rtl_busy_counter all "
                "reported as zero delta because those gauges exclude finalize wait_clear. Prefer "
                "total_program_cycles for throughput / wall-clock steering; keep span metrics for array "
                "utilization only."
            ),
        },
        "flagship_scope_note": (
            "The single-tile 16x16 family remains the flagship streaming anchor. Multi-tile rows are scoped "
            "more narrowly: they show that per-tile streaming generalizes across shapes, while compute-span "
            "duty-cycle captures whether the array still idles across tile boundaries."
        ),
        "asymmetry_note": (
            "The 16x16 family is single-tile (out_blocks=1, in_blocks=1), so its B=1 busy baseline is one accumulate RUN "
            "before finalize and the B sweep stays near-linear. The 32x32 and 64x64 families execute multiple accumulate RUNs per "
            "full GEMM, so their honest end-to-end question is not just the busy window but the duty-cycle across the entire "
            "compute span between the first and last accumulate tiles."
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
            "all_multitile_duty_cycles_improve_vs_serial": all(
                (
                    summary["max_b_serial_baseline_compute_span_duty_cycle"] is None
                    or (
                        summary["max_b_compute_span_duty_cycle"] is not None
                        and summary["max_b_serial_baseline_compute_span_duty_cycle"] is not None
                        and summary["max_b_compute_span_duty_cycle"]
                        > summary["max_b_serial_baseline_compute_span_duty_cycle"]
                    )
                )
                for summary in shape_summaries
                if int(summary["out_blocks"]) > 1 or int(summary["in_blocks"]) > 1
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
