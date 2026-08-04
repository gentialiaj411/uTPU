#!/usr/bin/env python3
"""BSTORE path measure: pre-widen baseline + post-widen smoke linkage.

Pre-widen workload identity comes from fused MNIST case1 attribution
(`cycle_attribution_mnist.json`) + program.mem — historically 4.0 cyc/word.
Post-widen functional evidence is `bstore_wide_smoke.json` (BSTORE_WIDTH=8).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from isa_encoder import OPCODE_BSTORE, OPCODE_HALT  # noqa: E402

OUT = REPO / "bench" / "results" / "bstore_path_measure.json"
MNIST_ATTR = REPO / "bench" / "results" / "cycle_attribution_mnist.json"
MNIST_MEM = REPO / "build" / "test_vectors" / "mnist_case1_program.mem"
WIDE_SMOKE = REPO / "bench" / "results" / "bstore_wide_smoke.json"
LUT_EST = REPO / "bench" / "results" / "bstore_widen_lut_estimate.json"

# Frozen pre-widen fused-MNIST identity (BSTORE_WIDTH=1). Do not recompute from
# live cycle_attribution_mnist.json after the widen lands.
PRE_WIDEN_BSTORE_CYCLES = 5197
PRE_WIDEN_TOTAL_CYCLES = 6523
PRE_WIDEN_COMPUTE_CYCLES = 709


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def parse_bstore_payloads(mem_path: Path) -> Tuple[int, int, List[Dict[str, int]]]:
    """EXT_ADDR BSTORE: header, addr, count, then ``count`` payload words."""
    words = [
        int(line.strip(), 16)
        for line in mem_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    i = 0
    bursts: List[Dict[str, int]] = []
    payload_total = 0
    while i < len(words):
        w = words[i]
        op = w & 0x7
        if op == OPCODE_HALT:
            break
        if op == OPCODE_BSTORE:
            addr = words[i + 1]
            count = words[i + 2]
            bursts.append({"pc": i, "addr": addr, "count": count})
            payload_total += count
            i += 3 + count
            continue
        if op in (3, 2, 1):  # LOAD, RUN, FETCH
            i += 2
        else:
            i += 1
    return payload_total, len(bursts), bursts


def buffer_geometry(array_size: int, compute_data_width: int) -> Dict[str, Any]:
    buffer_word_size = 16
    items = buffer_word_size // compute_data_width
    lanes = array_size * array_size
    banks = lanes // items
    return {
        "array_size": array_size,
        "compute_data_width": compute_data_width,
        "buffer_word_bits": buffer_word_size,
        "items_in_slot": items,
        "num_compute_lanes": lanes,
        "banks": banks,
        "compute_port_write_bits_per_cycle": banks * buffer_word_size,
        "compute_port_write_words_per_cycle": banks,
        "store_port_write_words_per_cycle": 1,
        "store_port_write_words_per_cycle_shipping": 8,
        "note": (
            "compute_en path asserts bank_we on all BANKS (tile-wide). "
            "Shipping BSTORE_WIDTH=8 writes up to 8 consecutive words/banks per we-beat."
        ),
    }


def main() -> int:
    if not MNIST_MEM.exists():
        print("MISSING mnist program.mem", flush=True)
        return 2

    attr = None
    if MNIST_ATTR.exists():
        attr = json.loads(MNIST_ATTR.read_text(encoding="utf-8"))

    payload, n_bursts, bursts = parse_bstore_payloads(MNIST_MEM)

    # Pre-widen baseline (frozen).
    bstore_cyc = PRE_WIDEN_BSTORE_CYCLES
    total = PRE_WIDEN_TOTAL_CYCLES
    identity = payload * 4 + n_bursts
    cyc_per_word = bstore_cyc / payload if payload else None

    geo16 = buffer_geometry(16, 4)
    geo8 = buffer_geometry(8, 8)
    measured_words_per_cyc = (1.0 / cyc_per_word) if cyc_per_word else None
    widen_n16 = (
        geo16["compute_port_write_words_per_cycle"] / measured_words_per_cyc
        if measured_words_per_cyc
        else None
    )

    bstore_frac = bstore_cyc / total

    def e2e(speedup: float) -> float:
        return 1.0 / (1.0 - bstore_frac + bstore_frac / speedup)

    removable = total - bstore_cyc - PRE_WIDEN_COMPUTE_CYCLES
    part_a_amdahl = total / (total - removable)

    wide = None
    if WIDE_SMOKE.exists():
        wide = json.loads(WIDE_SMOKE.read_text(encoding="utf-8"))
    lut = None
    if LUT_EST.exists():
        lut = json.loads(LUT_EST.read_text(encoding="utf-8"))

    post_attr = None
    if attr is not None:
        post_attr = {
            "total_program_cycles": attr.get("total_program_cycles"),
            "bstore_cycles": (attr.get("groups") or {}).get("bstore", {}).get("cycles"),
            "compute_cycles": (attr.get("groups") or {}).get("compute", {}).get("cycles"),
            "bstore_pct": (attr.get("groups") or {}).get("bstore", {}).get(
                "pct_of_total_program_cycles"
            ),
            "amdahl_part_a_multilayer": attr.get("amdahl_part_a_multilayer"),
            "e2e_speedup_vs_pre_widen": (
                PRE_WIDEN_TOTAL_CYCLES / attr["total_program_cycles"]
                if attr.get("total_program_cycles")
                else None
            ),
            "bstore_cyc_per_word": (
                (attr.get("groups") or {}).get("bstore", {}).get("cycles") / payload
                if payload and (attr.get("groups") or {}).get("bstore", {}).get("cycles")
                else None
            ),
        }

    report: Dict[str, Any] = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "implementation": {
            "BSTORE_WIDTH_shipping": 8,
            "status": "landed",
            "sources": [
                "rtl/top/top.sv",
                "rtl/unified_buffer/unified_buffer.sv",
                "firmware/host/run_bstore_wide_smoke.py",
            ],
            "post_widen_smoke": {
                "status": (wide or {}).get("status"),
                "attr_bstore": (wide or {}).get("attr_bstore"),
                "cycles_per_payload_word": (wide or {}).get("cycles_per_payload_word"),
                "payload_words": (wide or {}).get("payload_words"),
            }
            if wide
            else None,
            "post_widen_mnist_attr": post_attr,
            "ooc_lut_estimate": {
                "recommended_width": (lut or {}).get("recommendation", {}).get(
                    "recommended_width"
                ),
                "rows": (lut or {}).get("recommendation", {}).get("rows"),
            }
            if lut
            else None,
        },
        "workload": {
            "name": "fused_mnist_case1",
            "source_attr_live": str(MNIST_ATTR.relative_to(REPO)).replace("\\", "/")
            if MNIST_ATTR.exists()
            else None,
            "program_mem": str(MNIST_MEM.relative_to(REPO)).replace("\\", "/"),
            "pre_widen_total_program_cycles": total,
            "pre_widen_bstore_cycles": bstore_cyc,
            "pre_widen_bstore_pct": bstore_frac,
            "array_size": 16,
            "uart_baud_hz_testbench": 100000000,
        },
        "measured": {
            "bstore_bursts": n_bursts,
            "bstore_payload_words": payload,
            "bstore_cycles": bstore_cyc,
            "cycles_per_payload_word": cyc_per_word,
            "identity_check": {
                "formula": "payload_words * 4 + bursts",
                "value": identity,
                "matches_attr_bstore_cycles": identity == bstore_cyc,
            },
            "interpretation": (
                "Pre-widen baseline (frozen): exactly 4 on-chip cycles per 16-bit "
                "payload word plus 1 COUNT cycle per burst (5197 = 1296*4 + 13)."
            ),
        },
        "unified_buffer_write_width": {
            "sim_attr_path_n16_int4": geo16,
            "synth_path_n8_int8": geo8,
            "burst_achieved_words_per_cycle": measured_words_per_cyc,
            "bandwidth_ratio_compute_port_vs_burst_n16": widen_n16,
            "ordering_note": (
                "BSTORE_WIDTH=8 landed. Pre-widen sketches retained for Amdahl "
                "comparison; live post-widen MNIST attr is under "
                "implementation.post_widen_mnist_attr."
            ),
        },
        "amdahl_sketches_same_workload": {
            "part_a_if_bstore_and_compute_survive": {
                "removable_cycles": removable,
                "removable_fraction": removable / total,
                "ceiling_x": part_a_amdahl,
            },
            "bstore_pipeline_2x_e2e": e2e(2.0),
            "bstore_widen_8x_e2e": e2e(8.0),
            "bstore_widen_16x_e2e": e2e(16.0),
        },
        "bursts_head": bursts[:5],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    smoke_cpw = (wide or {}).get("cycles_per_payload_word")
    print(
        json.dumps(
            {
                "pre_widen_cyc_per_word": cyc_per_word,
                "post_widen_smoke_cyc_per_word": smoke_cpw,
                "post_widen_mnist_bstore_cyc": (post_attr or {}).get("bstore_cycles"),
                "post_widen_e2e_x": (post_attr or {}).get("e2e_speedup_vs_pre_widen"),
                "identity_ok": identity == bstore_cyc,
                "recommended_width": 8,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
