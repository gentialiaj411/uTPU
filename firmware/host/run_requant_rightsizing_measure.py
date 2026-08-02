"""Step-3 measurement for requant rightsizing: finalize-cycle A/B + duty snapshot.

Writes/updates bench/results/requant_rightsizing_synth.json with:
  - before/after compute_span_duty_cycle from systolic_characterization
  - finalize requant cycles per chunk at B=1/4/16/32 (wide N^2 vs narrow N)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from isa_encoder import IsaConfig
from requantization import RequantParams
from run_rtl_batched_gemm_sim import run_rtl_batched_gemm_sim
from run_systolic_characterization import OUTPUT_JSON as SYSTOLIC_JSON
from run_systolic_characterization import build_artifact as build_systolic_artifact


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "requant_rightsizing_synth.json"
INT8_CFG = IsaConfig(address_width=13, compute_data_width=8)
INT8_REQUANT = RequantParams(multiplier=11, right_shift=6, enable=True)
BATCHES = (1, 4, 16, 32)
ARRAY_SIZE = 16


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def _duty_snapshot(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for case in data.get("cases", []):
        shape = case["shape"]
        b = int(case["batch_size"])
        if int(shape["out_features"]) not in (16, 32):
            continue
        if b not in BATCHES:
            continue
        measured = case["measured"]
        serial = case.get("serial_baseline") or {}
        rows.append(
            {
                "shape": shape,
                "batch_size": b,
                "hoist_tile_payloads": bool(case.get("hoist_tile_payloads")),
                "rtl_busy_counter": measured.get("rtl_busy_counter"),
                "compute_span_duty_cycle": measured.get("compute_span_duty_cycle"),
                "serial_baseline_compute_span_duty_cycle": serial.get("compute_span_duty_cycle"),
            }
        )
    return {
        "source": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "generated_at_utc": data.get("generated_at_utc"),
        "status": data.get("status"),
        "rows": rows,
    }


def _expected_finalize_cycles(batch_size: int, *, narrow: bool) -> int:
    """Analytical requant wait_clear cycles for a single-tile (16x16) GEMM.

    Registered quantizer needs one fill bubble per presented input vector.
    Narrow streams one column at a time (fill+capture each); wide does one
    fill+capture per ARRAY_SIZE-wide writeback chunk.
    """
    chunks = (batch_size + ARRAY_SIZE - 1) // ARRAY_SIZE
    if not narrow:
        return 2 * chunks
    return 2 * batch_size


def _measure_finalize_ab() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for batch_size in BATCHES:
        wide_lanes = ARRAY_SIZE * ARRAY_SIZE
        narrow_lanes = ARRAY_SIZE
        wide = run_rtl_batched_gemm_sim(
            str(REPO_ROOT / "build" / "reports" / f"requant_finalize_wide_b{batch_size}.json"),
            out_features=ARRAY_SIZE,
            in_features=ARRAY_SIZE,
            batch_size=batch_size,
            stem=f"requant_finalize_ab_wide_b{batch_size}",
            cfg=INT8_CFG,
            accumulator_data_width=32,
            requant_params=INT8_REQUANT,
            quantizer_lanes=wide_lanes,
            relu_lanes=wide_lanes,
        )
        narrow = run_rtl_batched_gemm_sim(
            str(REPO_ROOT / "build" / "reports" / f"requant_finalize_narrow_b{batch_size}.json"),
            out_features=ARRAY_SIZE,
            in_features=ARRAY_SIZE,
            batch_size=batch_size,
            stem=f"requant_finalize_ab_narrow_b{batch_size}",
            cfg=INT8_CFG,
            accumulator_data_width=32,
            requant_params=INT8_REQUANT,
            quantizer_lanes=narrow_lanes,
            relu_lanes=narrow_lanes,
        )
        rows.append(
            {
                "shape": {"out_features": ARRAY_SIZE, "in_features": ARRAY_SIZE},
                "batch_size": batch_size,
                "wide": {
                    "quantizer_lanes": wide_lanes,
                    "rtl_sim_passed": wide.get("rtl_sim_passed"),
                    "rtl_busy_counter": wide.get("perf_busy_counter"),
                    "compute_span_duty_cycle": wide.get("compute_span_duty_cycle"),
                    "finalize_requant_cycles": wide.get("finalize_requant_cycles"),
                    "expected_finalize_requant_cycles": _expected_finalize_cycles(
                        batch_size, narrow=False
                    ),
                },
                "narrow": {
                    "quantizer_lanes": narrow_lanes,
                    "rtl_sim_passed": narrow.get("rtl_sim_passed"),
                    "rtl_busy_counter": narrow.get("perf_busy_counter"),
                    "compute_span_duty_cycle": narrow.get("compute_span_duty_cycle"),
                    "finalize_requant_cycles": narrow.get("finalize_requant_cycles"),
                    "expected_finalize_requant_cycles": _expected_finalize_cycles(
                        batch_size, narrow=True
                    ),
                },
                "delta_busy_cycles": (
                    None
                    if wide.get("perf_busy_counter") is None or narrow.get("perf_busy_counter") is None
                    else int(narrow["perf_busy_counter"]) - int(wide["perf_busy_counter"])
                ),
                "delta_finalize_requant_cycles": (
                    None
                    if wide.get("finalize_requant_cycles") is None
                    or narrow.get("finalize_requant_cycles") is None
                    else int(narrow["finalize_requant_cycles"]) - int(wide["finalize_requant_cycles"])
                ),
            }
        )
    all_pass = all(
        row["wide"]["rtl_sim_passed"] and row["narrow"]["rtl_sim_passed"] for row in rows
    )
    return {
        "status": "ok" if all_pass else "failed",
        "methodology": (
            "A/B iverilog on INT8 requant 16x16 single-tile programs. Wide = "
            "QUANTIZER_LANES=N^2 one-shot tile finalize; narrow = default "
            "QUANTIZER_LANES=N column stream. FINALIZE_REQUANT_CYCLES counts TB "
            "cycles with requant_finalize_enable && writeback_wait_clear."
        ),
        "rows": rows,
    }


def _load_or_empty() -> Dict[str, Any]:
    if OUTPUT_JSON.exists():
        return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    return {"version": 1, "steps": {}}


def build_artifact(*, skip_systolic: bool = False, skip_finalize_ab: bool = False) -> Dict[str, Any]:
    before_path = REPO_ROOT / "bench" / "results" / "systolic_characterization.json"
    before_snapshot = _duty_snapshot(before_path) if before_path.exists() else None

    if not skip_systolic:
        systolic = build_systolic_artifact(skip_iverilog=False)
        SYSTOLIC_JSON.write_text(json.dumps(systolic, indent=2) + "\n", encoding="utf-8")
        after_snapshot = _duty_snapshot(SYSTOLIC_JSON)
    else:
        after_snapshot = _duty_snapshot(before_path) if before_path.exists() else None

    artifact = _load_or_empty()
    if skip_finalize_ab:
        finalize_ab = artifact.get("step3_measure", {}).get("finalize_cycle_ab")
    else:
        finalize_ab = _measure_finalize_ab()

    artifact.update(
        {
            "version": 1,
            "generated_at_utc": _now_iso(),
            "git_sha": _git_sha(),
            "part": "xc7a100tcsg324-1",
            "step3_measure": {
                "status": "partial",
                "systolic_characterization_before": before_snapshot,
                "systolic_characterization_after": after_snapshot,
                "finalize_cycle_ab": finalize_ab,
                "synth_note": (
                    "DSP/WNS after-numbers are filled by Vivado batch runs; see steps.step1/step2 "
                    "after blocks and step3_measure.synth_runs when present."
                ),
            },
        }
    )
    # Preserve existing step1/step2 blocks if present.
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-systolic", action="store_true")
    parser.add_argument("--skip-finalize-ab", action="store_true")
    args = parser.parse_args()
    art = build_artifact(skip_systolic=args.skip_systolic, skip_finalize_ab=args.skip_finalize_ab)
    print(f"Wrote {OUTPUT_JSON}")
    finalize = art.get("step3_measure", {}).get("finalize_cycle_ab") or {}
    print(f"finalize_ab status={finalize.get('status')}")
    return 0 if finalize.get("status", "ok") in {"ok", None} else 1


if __name__ == "__main__":
    sys.exit(main())
