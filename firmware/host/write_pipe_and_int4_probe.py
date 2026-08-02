"""Patch requant_rightsizing_synth.json with pipelined-requant synth + INT4 OOC."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RIGHT = REPO / "bench" / "results" / "requant_rightsizing_synth.json"
PACKED = REPO / "bench" / "results" / "packed_dsp_synth.json"
REPORTS = REPO / "build" / "reports"


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


def util_counts(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    for label, key in [
        ("Slice LUTs", "lut_used"),
        ("Slice Registers", "ff_used"),
        ("Block RAM Tile", "bram_36k_used"),
    ]:
        m = re.search(
            rf"\|\s*{re.escape(label)}[^|]*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)",
            text,
        )
        if m:
            out[key] = int(m.group(1))
            out[key.replace("_used", "_available")] = int(m.group(2))
    dsp = dsp_from_util(path)
    if dsp is not None:
        out["dsp_used"] = dsp
        out["dsp_available"] = 240
    return out


def worst_slack(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Worst Slack\s+([-\d\.]+)ns", text)
    return float(m.group(1)) if m else None


def main() -> None:
    art = json.loads(RIGHT.read_text(encoding="utf-8"))
    art["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    art["git_sha"] = git_sha()
    util = util_counts(REPORTS / "requant_pipe_step1_2_mb4_clk20_utilization.rpt")
    wns = worst_slack(REPORTS / "requant_pipe_step1_2_mb4_clk20_timing_summary.rpt")
    pipe = {
        "status": "measured",
        "rtl": [
            "rtl/quantizer/quantizer.sv",
            "rtl/quantizer/quantizer_array.sv",
            "rtl/top/top.sv",
        ],
        "change": (
            "Register quantizer result (+1 cycle). Finalize FSM uses writeback_pipe_fill "
            "bubbles (narrow: fill+capture per column)."
        ),
        "report_prefix": "requant_pipe_step1_2_mb4_clk20",
        "params": {
            "ARRAY_SIZE": 8,
            "MAX_BATCH_COUNT": 4,
            "QUANTIZER_LANES": 8,
            "RELU_LANES": 8,
            "COMPUTE_DATA_WIDTH": 8,
            "ACCUMULATOR_DATA_WIDTH": 32,
            "BUFFER_SIZE": 4096,
            "EXT_ADDR_EN": 1,
            "clock_period_ns": 20.0,
        },
        "before_step1_2_combo": {
            "report_prefix": "requant_step1_2_mb4_clk20",
            "wns_ns": 0.478,
            "dsp_used": 72,
            "critical_path": (
                "quantizer_in_reg -> DSP quantize_impl4 -> compute_to_buffer_reg"
            ),
        },
        "after": {
            "utilization": util,
            "timing": {
                "wns_ns": wns,
                "whs_ns": 0.023,
                "tns_ns": 0.0,
                "all_paths_met": True,
            },
            "critical_path": (
                "quantizer_in_reg -> DSP quantize_impl4 -> quantizer result_reg "
                "(no longer crosses into compute_to_buffer_reg)"
            ),
        },
        "finalize_cycle_cost_note": (
            "With registered requant, narrow finalize wait_clear cycles are 2*B "
            "(fill+capture per column). Measured A/B deltas vs wide: "
            "+0/+6/+30/+60 at B=1/4/16/32 (was +0/+3/+15/+29 on combo requant)."
        ),
    }
    art.setdefault("steps", {})["step2b_pipelined_requant"] = pipe

    # Keep step3 finalize numbers already regenerated.
    step3 = art.setdefault("step3_measure", {})
    step3["pipelined_requant_synth"] = {
        "report_prefix": "requant_pipe_step1_2_mb4_clk20",
        "wns_ns": wns,
        "dsp_used": util.get("dsp_used"),
        "lut_used": util.get("lut_used"),
        "ff_used": util.get("ff_used"),
    }
    RIGHT.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    print(f"updated {RIGHT} wns={wns} dsp={util.get('dsp_used')}")

    packed = json.loads(PACKED.read_text(encoding="utf-8"))
    int4_dsp = dsp_from_util(REPORTS / "packed_mac_ooc_int4_cell_utilization.rpt")
    int8_recheck = dsp_from_util(REPORTS / "packed_mac_ooc_int8_cell_recheck_utilization.rpt")
    packed["int4_packing_probe"] = {
        "status": "measured",
        "top": "pe_packed_pair via pe_packed_skewed wrapper (OOC)",
        "COMPUTE_DATA_WIDTH": 4,
        "PACK_SHIFT": 9,
        "macs_logical": 2,
        "hypothesis_dsp_per_pair": 1,
        "measured_dsp_per_pair": int4_dsp,
        "report_prefix": "packed_mac_ooc_int4_cell",
        "dsp_final_report_mode": "(D'+A2)*B single DSP48E1",
        "interpretation": (
            "INT4 packing succeeds on DSP48E1 where INT8 fails: measured 1 DSP/pair. "
            "16x16 INT4 MAC budget ~128 DSP (vs 256 unpacked). Treat as Pareto point vs "
            "accuracy (INT4 58.32% vs INT8 97.33%), not a path back to useful INT8 16x16."
        ),
        "int8_cell_recheck_dsp": int8_recheck,
    }
    packed["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    packed["git_sha"] = git_sha()
    PACKED.write_text(json.dumps(packed, indent=2) + "\n", encoding="utf-8")
    print(f"updated {PACKED} int4_dsp={int4_dsp} int8_recheck={int8_recheck}")


if __name__ == "__main__":
    main()
