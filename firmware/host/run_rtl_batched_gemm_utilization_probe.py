from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from generate_batched_gemm_rtl_vectors import ARRAY_SIZE, generate_vectors
from run_rtl_batched_gemm_sim import _iverilog_run


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "build" / "reports" / "rtl_batched_gemm_utilization_probe.json"

# Minimal probe set before Phase 1 harness work.
CASES: Tuple[Tuple[int, int, int], ...] = (
    (16, 16, 1),
    (16, 16, 4),
    (16, 16, 16),
    (16, 16, 32),
    (16, 16, 64),
    (32, 32, 1),
    (32, 32, 16),
)


def _parse_counter(log: str, label: str) -> int:
    needle = f"{label}="
    for line in (log or "").splitlines():
        if needle in line:
            return int(line.split(needle, 1)[1].strip())
    raise ValueError(f"missing {label} in RTL log")


def _compute_model(case: Dict[str, Any]) -> Dict[str, Any]:
    out_blocks = int(case["out_blocks"])
    in_blocks = int(case["in_blocks"])
    batch_size = int(case["batch_size"])
    useful_macs = int(case["useful_macs"])
    pes = ARRAY_SIZE * ARRAY_SIZE

    # Zero-fit first-order weight-stationary systolic model:
    # per tile busy cycles = K_tile + ARRAY_SIZE + B - 2
    # with K_tile == ARRAY_SIZE for the current blocked path.
    per_tile_busy_cycles = int((2 * ARRAY_SIZE) + batch_size - 2)
    total_busy_cycles = int(out_blocks * in_blocks * per_tile_busy_cycles)
    pe_util_busy = useful_macs / float(pes * total_busy_cycles)
    return {
        "per_tile_busy_cycles": per_tile_busy_cycles,
        "total_busy_cycles": total_busy_cycles,
        "pe_utilization_busy_window": pe_util_busy,
    }


def run_probe(output_json: str = str(OUTPUT_JSON)) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for out_features, in_features, batch_size in CASES:
        stem = f"batched_probe_o{out_features}_i{in_features}_b{batch_size}"
        vectors = generate_vectors(
            out_features=out_features,
            in_features=in_features,
            batch_size=batch_size,
            stem=stem,
            output_json=os.path.join("build", "test_vectors", f"{stem}.json"),
        )
        ok, log = _iverilog_run(str(REPO_ROOT))
        if not ok:
            raise RuntimeError(f"RTL probe failed for {(out_features, in_features, batch_size)}\n{log}")
        cycle_ctr = _parse_counter(log, "PERF_CYCLE_COUNTER")
        busy_ctr = _parse_counter(log, "PERF_BUSY_COUNTER")
        program_ctr = _parse_counter(log, "PERF_PROGRAM_COUNT")
        useful_macs = int(vectors["useful_macs"])
        pes = ARRAY_SIZE * ARRAY_SIZE
        measured = {
            "rtl_cycle_counter": cycle_ctr,
            "rtl_busy_counter": busy_ctr,
            "rtl_program_counter": program_ctr,
            "array_busy_fraction": (busy_ctr / cycle_ctr) if cycle_ctr > 0 else 0.0,
            "pe_utilization_busy_window": (useful_macs / float(pes * busy_ctr)) if busy_ctr > 0 else 0.0,
            "pe_utilization_total_window": (useful_macs / float(pes * cycle_ctr)) if cycle_ctr > 0 else 0.0,
        }
        model = _compute_model(vectors)
        residual = {
            "busy_cycles_delta": int(model["total_busy_cycles"]) - int(measured["rtl_busy_counter"]),
            "busy_cycles_rel_error_pct": (
                100.0 * (int(model["total_busy_cycles"]) - int(measured["rtl_busy_counter"])) / float(measured["rtl_busy_counter"])
                if measured["rtl_busy_counter"] > 0
                else 0.0
            ),
            "pe_utilization_busy_window_delta_pct_points": 100.0
            * (float(model["pe_utilization_busy_window"]) - float(measured["pe_utilization_busy_window"])),
        }
        rows.append(
            {
                "shape": {"out_features": out_features, "in_features": in_features},
                "batch_size": batch_size,
                "out_blocks": int(vectors["out_blocks"]),
                "in_blocks": int(vectors["in_blocks"]),
                "useful_macs": useful_macs,
                "measured": measured,
                "model": model,
                "residual": residual,
            }
        )

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "array_size": ARRAY_SIZE,
        "methodology": {
            "headline_metric_primary": "RTL perf counters: busy/cycle and useful_macs/(ARRAY_SIZE^2 * busy)",
            "model_secondary": "Zero-fit first-order weight-stationary busy-cycle model: per_tile_busy_cycles = 2*ARRAY_SIZE + B - 2",
            "scope_note": "Probe only, before the full Phase 1 harness. No free parameters fitted to a single point.",
        },
        "cases": rows,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    payload = run_probe()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
