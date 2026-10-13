#!/usr/bin/env python3
"""RTL FSM state-group cycle attribution for blocked-FC programs.

Emits bench/results/cycle_attribution.json for
(16,16)/(32,32)/(64,64) x B in {1,4,16,48} via iverilog + MAGIC_READ_PERF.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from generate_batched_gemm_rtl_vectors import ARRAY_SIZE, generate_vectors  # noqa: E402
from isa_encoder import IsaConfig  # noqa: E402
from run_rtl_batched_gemm_sim import _iverilog_run, _parse_perf_counter  # noqa: E402

OUT = REPO / "bench" / "results" / "cycle_attribution.json"
SHAPES = [(16, 16), (32, 32), (64, 64)]
BATCHES = [1, 4, 16, 48]
ATTR_KEYS = [
    "fetch_decode",
    "store",
    "load",
    "compute",
    "writeback",
    "bstore",
    "ext_addr",
    "requant",
    "result_fetch",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def _pct(part: Optional[int], total: Optional[int]) -> Optional[float]:
    if part is None or total is None or total <= 0:
        return None
    return float(part) / float(total)


def parse_attr(log: str) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    for key in ATTR_KEYS:
        out[key] = _parse_perf_counter(log, f"ATTR_{key.upper()}")
    return out


def run_case(out_f: int, in_f: int, batch: int) -> Dict[str, Any]:
    cfg = IsaConfig(address_width=12, compute_data_width=4) if batch > 1 else None
    hoist = bool(out_f > ARRAY_SIZE or in_f > ARRAY_SIZE)
    gen = generate_vectors(
        out_features=out_f,
        in_features=in_f,
        batch_size=batch,
        stem=f"cycle_attr_{out_f}x{in_f}_b{batch}",
        hoist_tile_payloads=hoist,
        cfg=cfg,
    )
    ok, log = _iverilog_run(
        str(REPO),
        quantizer_pipe_depth=3,
        out_vvp_name=f"tb_cycle_attr_{out_f}x{in_f}_b{batch}.out",
    )
    total = _parse_perf_counter(log, "TOTAL_PROGRAM_CYCLES")
    attrs = parse_attr(log)
    attributed = sum(int(v or 0) for v in attrs.values())
    program_words = int(len(gen["program"]) // 2) if isinstance(gen.get("program"), (bytes, bytearray)) else None
    if program_words is None:
        # generate_vectors returns program in payload under nested keys
        program_words = int(gen.get("program_words") or 0) or None
    # generate_vectors stores program length via mem; recover from file words in svh if needed
    if not program_words:
        m = re.search(r"`define BG_WORDS (\d+)", (REPO / "build" / "test_vectors" / "batched_gemm_expected.svh").read_text(encoding="utf-8"))
        program_words = int(m.group(1)) if m else None

    n = int(ARRAY_SIZE)
    tiles = max(1, (out_f // n) * (in_f // n))
    compute_model = (2 * n + batch - 1) * tiles
    groups = {}
    for key in ATTR_KEYS:
        cyc = attrs[key]
        groups[key] = {
            "cycles": cyc,
            "pct_of_total_program_cycles": _pct(cyc, total),
        }

    return {
        "shape": {"out_features": out_f, "in_features": in_f},
        "batch_size": batch,
        "array_size": n,
        "hoist_tile_payloads": hoist,
        "program_words": program_words,
        "rtl_sim_passed": bool(ok),
        "total_program_cycles": total,
        "attr_sum_cycles": attributed,
        "attr_sum_vs_total_delta": (
            None if total is None else int(attributed) - int(total)
        ),
        "groups": groups,
        "hypothesis_metrics": {
            "fetch_decode_cycles": attrs["fetch_decode"],
            "load_cycles": attrs["load"],
            "compute_cycles": attrs["compute"],
            "fetch_decode_over_compute": (
                None
                if not attrs["compute"] or not attrs["fetch_decode"]
                else float(attrs["fetch_decode"]) / float(attrs["compute"])
            ),
            "load_over_total": _pct(attrs["load"], total),
            "fetch_decode_over_total": _pct(attrs["fetch_decode"], total),
            "compute_over_total": _pct(attrs["compute"], total),
            "analytical_compute_cycles_2N_B_1_times_tiles": compute_model,
            "tiles": tiles,
        },
        "log_tail": "\n".join((log or "").splitlines()[-20:]),
    }


def main() -> int:
    cases: List[Dict[str, Any]] = []
    for shape in SHAPES:
        for batch in BATCHES:
            print(f"=== cycle attribution {shape} B={batch} ===", flush=True)
            cases.append(run_case(shape[0], shape[1], batch))

    # Headline verdict: weight LOAD vs instruction-stream cost.
    flagship = next(
        (c for c in cases if c["shape"]["out_features"] == 64 and c["batch_size"] == 48),
        cases[-1] if cases else None,
    )
    verdict = "inconclusive"
    reason = "no cases"
    if flagship and flagship.get("rtl_sim_passed") and flagship.get("total_program_cycles"):
        g = flagship["groups"]

        def gp(name: str) -> float:
            return float((g.get(name) or {}).get("pct_of_total_program_cycles") or 0.0)

        ld = gp("load")
        cmp_ = gp("compute")
        fd = gp("fetch_decode")
        stream = gp("fetch_decode") + gp("store") + gp("bstore") + gp("ext_addr") + gp("requant")
        # Annotate flagship with stream aggregate for the artifact consumers.
        flagship["hypothesis_metrics"]["instruction_stream_pct"] = stream
        flagship["hypothesis_metrics"]["instruction_stream_cycles"] = int(
            sum(
                int((g[k]["cycles"] or 0))
                for k in ("fetch_decode", "store", "bstore", "ext_addr", "requant")
            )
        )
        if ld <= 0.05 and stream >= 0.50:
            verdict = "confirmed_fetch_stream_dominates_load_negligible"
            reason = (
                f"On 64x64 B=48 N={ARRAY_SIZE} (fast-UART TB): instruction-stream "
                f"(fetch_decode+store+bstore+ext_addr+requant)={stream:.1%}, "
                f"load={ld:.1%}, compute={cmp_:.1%}, result_fetch={gp('result_fetch'):.1%} "
                f"of on-chip core cycles. PE-level concurrent LOAD+COMPUTE cannot "
                f"materially help. Redirect to descriptor ISA. Part A Amdahl ceiling "
                f"strips the stream share only; post-cut compute share stays drain-limited."
            )
        elif ld >= 0.20:
            verdict = "rejected_weight_load_material"
            reason = (
                f"load={ld:.1%} of total on 64x64 B=48; concurrent LOAD+COMPUTE "
                f"remains justified."
            )
        else:
            verdict = "mixed"
            reason = (
                f"fetch_decode={fd:.1%} stream={stream:.1%} load={ld:.1%} "
                f"compute={cmp_:.1%}; see cases for detail."
            )

    artifact: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "methodology": {
            "source": (
                "RTL FSM state-group counters in rtl/top/top.sv, streamed via "
                "MAGIC_READ_PERF (0xA4) after legacy 4x64 words"
            ),
            "window": (
                "Same START->HALT window as total_program_cycles / "
                "perf_program_cycle_counter (on-chip core cycles; fast-UART TB)"
            ),
            "array_size": ARRAY_SIZE,
            "uart_baud_hz_testbench": 100_000_000,
            "uart_baud_hz_shipping_default": 115_200,
            "measurement_scope": (
                "On-chip core cycles in rtl/tb/tb_batched_gemm.sv with "
                "UART_BAUD=100000000 (~868x shipping 115200). NOT board wall-clock."
            ),
            "note": (
                "isa_simulator.py cycle_count_sequential is NOT absolute-RTL-faithful "
                "(1-cycle-per-op); attribution is measured in iverilog RTL."
            ),
            "groups": ATTR_KEYS,
        },
        "hypothesis": {
            "claim": (
                "Instruction stream dominates on-chip core cycles (fast-UART TB); "
                "buffer->PE weight LOAD cannot materially move duty cycle via "
                "PE-level double-buffering alone."
            ),
            "analytical_sketch": (
                "FETCH_BRAM+WAIT+DECODE >= 3 cycles/instruction; ~217 words/tile ⇒ "
                ">=650 fetch/decode cycles/tile vs compute 2N+B-1."
            ),
        },
        "verdict": verdict,
        "verdict_reason": reason,
        "cases": cases,
    }
    if flagship and flagship.get("total_program_cycles"):
        stream_pct = float(flagship["hypothesis_metrics"].get("instruction_stream_pct") or 0.0)
        total = int(flagship["total_program_cycles"])
        remain = int(round(total * (1.0 - stream_pct)))
        n = int(ARRAY_SIZE)
        b = int(flagship["batch_size"])
        compute_cyc = int((flagship["groups"]["compute"]["cycles"] or 0))
        result_cyc = int((flagship["groups"]["result_fetch"]["cycles"] or 0))
        artifact["amdahl_part_a"] = {
            "flagship": f"{flagship['shape']['out_features']}x{flagship['shape']['in_features']}_B{b}_N{n}",
            "total_program_cycles": total,
            "instruction_stream_pct": stream_pct,
            "cycles_if_instruction_stream_removed": remain,
            "speedup_ceiling": (float(total) / float(remain)) if remain else None,
            "post_cut_compute_share_approx": (float(compute_cyc) / float(remain)) if remain else None,
            "post_cut_result_fetch_share_approx": (float(result_cyc) / float(remain)) if remain else None,
            "array_streaming_ceiling_B_over_2N_B": {
                "N": n,
                "B": b,
                "value": float(b) / float(2 * n + b),
            },
            "note": (
                f"Array streaming ceiling is B/(2N+B) at measured N={n}, not shipping N=8. "
                "Post-descriptor end-to-end compute share remains drain-limited; "
                "do not conflate with the isolated array ceiling."
            ),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"VERDICT: {verdict}")
    print(reason)
    return 0 if all(c.get("rtl_sim_passed") for c in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
