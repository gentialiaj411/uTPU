"""Measure QUANTIZER_PIPE_DEPTH in {0, 3} cycle cost at fixed QUANTIZER_LANES.

Uses the INT8 requant path (same contract as run_requant_rightsizing_measure.py).
Wall-clock compares cycles/Fmax using closed synth Fmax:
  depth=0: 50 MHz (combo Step1+2 closed_config)
  depth=3: 100 MHz (requant_fmax_mb4_clk10_pd3)
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from isa_encoder import IsaConfig
from requantization import RequantParams
from run_rtl_batched_gemm_sim import run_rtl_batched_gemm_sim

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "bench" / "results" / "quantizer_pipe_depth_cost.json"

ARRAY_SIZE = 16
FIXED_LANES = ARRAY_SIZE  # narrow column-stream; fixed across depths
BATCHES = (1, 4, 16, 32)
DEPTHS = (0, 3)
INT8_CFG = IsaConfig(address_width=13, compute_data_width=8)
INT8_REQUANT = RequantParams(multiplier=11, right_shift=6, enable=True)
FMAX_MHZ = {0: 50.0, 3: 100.0}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def _wall_ns(cycles: int | None, fmax_mhz: float) -> float | None:
    if cycles is None:
        return None
    return round(float(cycles) / fmax_mhz * 1000.0, 3)


def main() -> int:
    rows: List[Dict[str, Any]] = []
    for batch in BATCHES:
        for depth in DEPTHS:
            print(f"[pipe_depth_cost] B={batch} DEPTH={depth} LANES={FIXED_LANES}", flush=True)
            m = run_rtl_batched_gemm_sim(
                str(REPO_ROOT / "build" / "reports" / f"pipe_depth_b{batch}_pd{depth}.json"),
                out_features=ARRAY_SIZE,
                in_features=ARRAY_SIZE,
                batch_size=batch,
                stem=f"pipe_depth_cost_b{batch}_pd{depth}",
                cfg=INT8_CFG,
                accumulator_data_width=32,
                requant_params=INT8_REQUANT,
                quantizer_lanes=FIXED_LANES,
                relu_lanes=FIXED_LANES,
                quantizer_pipe_depth=depth,
            )
            prog = m.get("total_program_cycles")
            fin = m.get("finalize_requant_cycles")
            rows.append(
                {
                    "batch_size": batch,
                    "quantizer_pipe_depth": depth,
                    "quantizer_lanes": FIXED_LANES,
                    "rtl_sim_passed": m.get("rtl_sim_passed"),
                    "total_program_cycles": prog,
                    "finalize_requant_cycles": fin,
                    "rtl_busy_counter": m.get("perf_busy_counter"),
                    "compute_span_cycles": m.get("compute_span_cycles"),
                    "fmax_mhz_closed": FMAX_MHZ[depth],
                    "wall_clock_ns_from_total_program_cycles": _wall_ns(prog, FMAX_MHZ[depth]),
                }
            )

    by_b: Dict[str, Any] = {}
    for batch in BATCHES:
        d0 = next(r for r in rows if r["batch_size"] == batch and r["quantizer_pipe_depth"] == 0)
        d3 = next(r for r in rows if r["batch_size"] == batch and r["quantizer_pipe_depth"] == 3)
        d_fin = (
            None
            if d0["finalize_requant_cycles"] is None or d3["finalize_requant_cycles"] is None
            else int(d3["finalize_requant_cycles"]) - int(d0["finalize_requant_cycles"])
        )
        d_prog = (
            None
            if d0["total_program_cycles"] is None or d3["total_program_cycles"] is None
            else int(d3["total_program_cycles"]) - int(d0["total_program_cycles"])
        )
        # Classify: burst-fill (~constant, ~PIPE_DEPTH) vs per-column (~k*B).
        by_b[str(batch)] = {
            "finalize_cycles_depth0": d0["finalize_requant_cycles"],
            "finalize_cycles_depth3": d3["finalize_requant_cycles"],
            "delta_finalize_cycles": d_fin,
            "total_program_cycles_depth0": d0["total_program_cycles"],
            "total_program_cycles_depth3": d3["total_program_cycles"],
            "delta_total_program_cycles": d_prog,
            "wall_clock_ns_depth0": d0["wall_clock_ns_from_total_program_cycles"],
            "wall_clock_ns_depth3": d3["wall_clock_ns_from_total_program_cycles"],
            "wall_clock_speedup_depth3_vs_0": (
                None
                if not d0["wall_clock_ns_from_total_program_cycles"]
                or not d3["wall_clock_ns_from_total_program_cycles"]
                else round(
                    float(d0["wall_clock_ns_from_total_program_cycles"])
                    / float(d3["wall_clock_ns_from_total_program_cycles"]),
                    4,
                )
            ),
        }

    deltas = [by_b[str(b)]["delta_finalize_cycles"] for b in BATCHES]
    # ~2-cycle fill per finalize burst => roughly constant across B.
    # ~2 cycles/column => delta grows ~linear with B (or with columns streamed).
    burst_like = (
        all(d is not None for d in deltas)
        and max(deltas) - min(deltas) <= 2
        and max(deltas) <= 8
    )
    per_column_like = (
        all(d is not None for d in deltas)
        and deltas[-1] is not None
        and deltas[-1] >= 16
    )
    if burst_like:
        verdict = "ship_depth_3_default"
        action = "Set QUANTIZER_PIPE_DEPTH default to 3; 2x Fmax dominates a small fill cost."
    elif per_column_like:
        verdict = "report_both_human_decision"
        action = (
            "Depth=3 pays ~per-column pipe fill under the current finalize FSM "
            "(fill re-armed each column). Report wall-clock both ways; do not auto-flip default."
        )
    else:
        verdict = "report_both_human_decision"
        action = "Ambiguous cost shape; keep default=0 until human decides."

    artifact = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "status": "ok" if all(r["rtl_sim_passed"] for r in rows) else "failed",
        "shape": {"out_features": ARRAY_SIZE, "in_features": ARRAY_SIZE},
        "fixed_quantizer_lanes": FIXED_LANES,
        "requant": INT8_REQUANT.as_dict(),
        "pipe_depths": list(DEPTHS),
        "batches": list(BATCHES),
        "fmax_mhz_closed": FMAX_MHZ,
        "fmax_provenance": {
            "0": "combo PIPE_DEPTH=0 Step1+2 closed_config @ 20 ns (50 MHz)",
            "3": "timing_closure_sweep.json requant_fmax_mb4_clk10_pd3 closed @ 10 ns (100 MHz)",
        },
        "cases": rows,
        "deltas_by_batch": by_b,
        "verdict": verdict,
        "action": action,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "verdict": verdict, "deltas_by_batch": by_b}, indent=2))
    print(f"-> {OUT}")
    return 0 if artifact["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
