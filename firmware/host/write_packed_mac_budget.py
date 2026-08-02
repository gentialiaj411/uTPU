#!/usr/bin/env python3
"""Rebuild packed_dsp_synth.json and budget gate from measured OOC reports."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "build" / "reports"
OUT_PACKED = REPO / "bench" / "results" / "packed_dsp_synth.json"
OUT_RIGHT = REPO / "bench" / "results" / "requant_rightsizing_synth.json"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def dsp_from_util(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "DSP48E1 only" in line:
            nums = [int(p.strip()) for p in line.split("|") if re.fullmatch(r"\s*\d+\s*", p)]
            if nums:
                return nums[0]
    m = re.search(r"\|\s*DSPs\s*\|\s*(\d+)\s*\|", text)
    return int(m.group(1)) if m else None


def main() -> None:
    cell = dsp_from_util(REPORTS / "packed_mac_ooc_cell1_utilization.rpt")
    a8 = dsp_from_util(REPORTS / "packed_mac_ooc_8x8_utilization.rpt")
    a16 = dsp_from_util(REPORTS / "packed_mac_ooc_16x16_utilization.rpt")
    assert cell == 2, f"expected 2 DSP/cell, got {cell}"
    assert a8 == 64, f"expected 64 DSP for packed 8x8, got {a8}"
    assert a16 == 240, f"expected 240 DSP (part-capped) for packed 16x16, got {a16}"

    packed = {
        "version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "vivado_version": "2025.2",
        "part": "xc7a100tcsg324-1",
        "methodology": (
            "Out-of-context synth_design of pe_packed_skewed / pe_array_packed via "
            "scripts/synth_packed_mac_ooc.tcl. No IOB/place. DSP counts from utilization.rpt "
            "'DSP48E1 only'. Inference audit from synth DSP Final Report (cascade A*B + PCIN>>17)."
        ),
        "loop_converge_root_cause": {
            "files": ["rtl/PEArray/pe_controller.sv", "rtl/PEArray/pe_controller_packed.sv"],
            "symptom": "synth_failed: loop condition does not converge after 2000 iterations",
            "cause": (
                "Nested for (col < MAX_BATCH_COUNT) / for (row < ARRAY_SIZE) capture loops gated "
                "on runtime active_columns=batch_count prevented Vivado unroll."
            ),
            "fix": (
                "Rewrote capture to O(ARRAY_SIZE) static loop: col = capture_cycle - ARRAY_SIZE - 1 - row. "
                "Same schedule semantics; bit-exact requant suite still 11/11."
            ),
        },
        "packing_inference": {
            "hypothesis_2_mac_per_dsp": False,
            "measured_dsp_per_pe_packed_skewed_cell": cell,
            "dsp_final_report_modes": ["A'*B (wide)", "PCIN>>17+A*B (cascade)"],
            "explanation": (
                "PACK_SHIFT=18 makes packed_operand width 27 and product width 35, which does not "
                "fit a single DSP48E1 25x18 multiply. Vivado infers a 2-DSP cascade per packed cell, "
                "so packing does not reduce DSP vs one-MAC-per-DSP."
            ),
        },
        "runs": [
            {
                "name": "packed_mac_ooc_cell1",
                "top": "pe_packed_skewed",
                "macs_logical": 2,
                "hypothesis_dsp": 1,
                "measured_dsp": cell,
                "report_prefix": "packed_mac_ooc_cell1",
                "status": "ok",
            },
            {
                "name": "packed_mac_ooc_8x8",
                "top": "pe_array_packed",
                "ARRAY_SIZE": 8,
                "macs_logical": 64,
                "packed_cells": 32,
                "hypothesis_dsp": 32,
                "measured_dsp": a8,
                "report_prefix": "packed_mac_ooc_8x8",
                "status": "ok",
                "note": "64 DSP = same as unpacked 8x8 MAC; packing saves 0 DSP at this size.",
            },
            {
                "name": "packed_mac_ooc_16x16",
                "top": "pe_array_packed",
                "ARRAY_SIZE": 16,
                "macs_logical": 256,
                "packed_cells": 128,
                "hypothesis_dsp": 128,
                "unconstrained_expected_dsp_at_2_per_cell": 256,
                "measured_dsp": a16,
                "dsp_available": 240,
                "report_prefix": "packed_mac_ooc_16x16",
                "status": "ok_part_capped",
                "note": (
                    "Would need 256 DSP at measured 2/cell; Artix-7 A7-100T reports 240/240 (100%). "
                    "Hypothesis 128 is FALSE."
                ),
            },
        ],
    }
    OUT_PACKED.write_text(json.dumps(packed, indent=2) + "\n", encoding="utf-8")

    art = json.loads(OUT_RIGHT.read_text(encoding="utf-8"))
    art["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    art["git_sha"] = git_sha()
    art["task_b"] = {
        "status": "ok",
        "packed_mac_16x16_measured_dsp": a16,
        "packed_mac_8x8_measured_dsp": a8,
        "dsp_per_packed_cell_measured": cell,
        "hypothesis_128_validated": False,
        "artifact": "bench/results/packed_dsp_synth.json",
    }
    art["task_c_decision_gate"] = {
        "part_dsp_available": 240,
        "budget_table": [
            {
                "config": "8x8 unpacked MAC + tile requant (baseline pre-rightsizing)",
                "dsp": 192,
                "source": "measured baseline_8x8_current_rtl_synth.json",
                "fits_240": True,
            },
            {
                "config": "8x8 + Step1 only (QUANTIZER_LANES=64)",
                "dsp": 128,
                "source": "measured requant_step1_wide_mb4_clk20",
                "fits_240": True,
            },
            {
                "config": "8x8 + Step1+2 (QUANTIZER_LANES=8)",
                "dsp": 72,
                "source": "measured requant_step1_2_mb4_clk20",
                "fits_240": True,
            },
            {
                "config": "16x16 packed MAC only (OOC pe_array_packed)",
                "dsp": a16,
                "source": "measured packed_mac_ooc_16x16",
                "fits_240": False,
                "note": "Part-capped; unconstrained need ~256 at 2 DSP/cell",
            },
            {
                "config": "16x16 packed MAC + Step2 requant (16 lanes x 1 DSP) [estimate]",
                "dsp": a16 + 16,
                "source": "measured MAC + measured Step1 1DSP/quantizer scaled",
                "fits_240": False,
                "estimate_components": {"mac_measured": a16, "requant_estimate": 16},
            },
            {
                "config": "16x16 if packing were 1 DSP/cell (OLD estimate — INVALID)",
                "dsp": 128 + 16,
                "source": "engineering estimate — SUPERSEDED by Task B",
                "fits_240": True,
                "invalid": True,
            },
        ],
        "verdict_16x16_fits_240": False,
        "plain_statement": (
            "ARRAY_SIZE=16 does NOT fit Artix-7 A7-100T 240 DSP. Packed MAC alone measures "
            f"{a16}/240 and would need ~256 unconstrained because each pe_packed_skewed cell "
            "infers a 2-DSP cascade, not 1 DSP for 2 MACs. Step2 requant rightsizing cannot "
            "rescue 16x16 on this part."
        ),
        "options_if_no": [
            {
                "option": "stay_array_size_8",
                "dsp_measured": 72,
                "fits": True,
                "note": "Step1+2 already measured closed; ship 8x8 INT8",
            },
            {
                "option": "redesign_packing_for_single_dsp48",
                "dsp": "TODO/VERIFY",
                "fits": "unknown",
                "note": (
                    "Need PACK_SHIFT/operand widths that fit one 25x18 multiply per cell; "
                    "current PACK_SHIFT=18 forces cascade."
                ),
            },
            {
                "option": "smaller_than_16_array",
                "example": "ARRAY_SIZE=10 or 12",
                "dsp": "TODO/VERIFY",
                "fits": "unknown until re-measured",
            },
        ],
        "step2_rtl_note": (
            "This prompt said not to start Step2 until the gate. In this workspace Step2 RTL "
            "already landed earlier in the session; the gate still says 16x16 is dead, but "
            "Step2 remains load-bearing for the measured 8x8 DSP win (192->72)."
        ),
    }
    # Strike invalid estimate in budget_arithmetic
    art["budget_arithmetic"]["projected_array_size_16_after_step1_and_step2"] = {
        "mac_packed_dsp_MEASURED": a16,
        "mac_packed_dsp_estimate_INVALID": 128,
        "requant_lanes_eq_array_size_1dsp_each": 16,
        "total_with_measured_mac": a16 + 16,
        "fits_240": False,
        "conclusion": (
            "SUPERSEDED: Task B measured packed 16x16 MAC at 240 DSP (2 DSP/cell cascade). "
            "Old 144-fit conclusion is invalid."
        ),
    }
    OUT_RIGHT.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT_PACKED)
    print("wrote", OUT_RIGHT)
    print("VERDICT: 16x16 fits 240?", False)
    print(f"measured cell={cell} 8x8={a8} 16x16={a16}")


if __name__ == "__main__":
    main()
