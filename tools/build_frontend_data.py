#!/usr/bin/env python3
"""Aggregate committed bench artifacts into frontend/public/data bundles.

Single source of truth for the Evidence Explorer UI. Every surfaced metric
must carry tier + source_artifact; the build fails otherwise.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST_DIR = REPO_ROOT / "firmware" / "host"
BENCH_RESULTS = REPO_ROOT / "bench" / "results"
OUT_DIR = REPO_ROOT / "frontend" / "public" / "data"

VALID_TIERS = frozenset({"sim", "ci", "synth", "silicon"})
GITHUB_REPO = "gentialiaj411/uTPU"

# Artifacts whose headline numbers are regenerated in .github/workflows/ci.yml
CI_ARTIFACTS: Set[str] = {
    "fusion_payoff.json",
    "tiling_correctness.json",
    "scheduler_cycles.json",
    "cost_model_heldout.json",
    "cublas_baseline.json",
    "selection_ab.json",
    "board_fit_audit.json",
    "scheduler_rtl_crosscheck.json",
    "scheduler_rtl_crosscheck_bigmlp.json",
    "megakernel_payoff.json",
    "superopt_payoff.json",
    "batched_gemm_correctness.json",
    "systolic_characterization.json",
    "real_model_end_to_end.json",
    "cost_model_regression.json",
    "cost_model_selection.json",
    "multi_pe_sim.json",
    "packed_dsp_synth.json",
    "pe_packed_pair_sim.json",
    "pe_array_packed_sim.json",
    "pe_array_packed_hardened.json",
    "packed_array_cycle_compare.json",
    "top_packed_smoke.json",
    "uart_preboard_roundtrip.json",
    "utpu_conv2d_validation.json",
    "utpu_small_cnn_validation.json",
    "utpu_attention_hybrid.json",
    "real_model_accelerator.json",
}

SYNTH_ARTIFACTS: Set[str] = {
    "baseline_8x8_current_rtl_synth.json",
    "p4_2_vivado_reports.json",
    "packed_dsp_synth.json",
    "requant_rightsizing_synth.json",
    "prog_depth_sweep.json",
    "design_space_sweep.json",
}

Claim = Dict[str, Any]
Extractor = Callable[[str, Dict[str, Any]], List[Claim]]


def _rel_artifact(name: str) -> str:
    return f"bench/results/{name}"


def _claim(
    *,
    claim_id: str,
    category: str,
    headline: str,
    value: Any,
    unit: str,
    tier: str,
    source_name: str,
    fence_note: str,
    raw: Any,
    tags: Optional[List[str]] = None,
) -> Claim:
    return {
        "id": claim_id,
        "category": category,
        "headline": headline,
        "value": value,
        "unit": unit,
        "tier": tier,
        "source_artifact": _rel_artifact(source_name),
        "fence_note": fence_note,
        "tags": tags or [],
        "raw": raw,
    }


def infer_tier(source_name: str, blob: Dict[str, Any]) -> str:
    if source_name in SYNTH_ARTIFACTS:
        return "synth"
    if source_name in CI_ARTIFACTS:
        return "ci"
    # sim-only replay artifacts
    methodology = blob.get("methodology")
    claims_scope = ""
    if isinstance(methodology, dict):
        claims_scope = str(methodology.get("claims_scope", "") or "")
    scope = str(blob.get("scope", "") or claims_scope)
    if "sim" in scope.lower() or "replay" in scope.lower():
        return "sim"
    return "sim"


def _extract_baseline_8x8(name: str, blob: Dict[str, Any]) -> List[Claim]:
    closed = blob.get("closed_config") or {}
    timing = closed.get("timing") or {}
    util = closed.get("utilization") or {}
    return [
        _claim(
            claim_id="rtl_synth_8x8_timing_closed",
            category="FPGA synthesis",
            headline="Current RTL 8×8 INT8 timing closure",
            value=closed.get("frequency_mhz"),
            unit="MHz",
            tier="synth",
            source_name=name,
            fence_note="Vivado P&R on current RTL; supersedes stale p4_2 100 MHz numbers.",
            raw={"wns_ns": timing.get("wns_ns"), "clock_period_ns": closed.get("clock_period_ns")},
        ),
        _claim(
            claim_id="rtl_synth_8x8_dsp",
            category="FPGA synthesis",
            headline="DSP utilization (current RTL 8×8)",
            value=f"{util.get('dsp_used')}/{util.get('dsp_available')}",
            unit="DSP",
            tier="synth",
            source_name=name,
            fence_note="Post-route utilization from baseline_8x8_current_rtl_synth.json.",
            raw=util,
        ),
        _claim(
            claim_id="rtl_synth_8x8_bit",
            category="FPGA synthesis",
            headline="Bitstream generated",
            value=bool(closed.get("bit_generated")),
            unit="bool",
            tier="synth",
            source_name=name,
            fence_note="Synthesized bit exists; on-board execution not captured (P0 open).",
            raw={"bit_generated": closed.get("bit_generated"), "route_status": closed.get("route_status")},
        ),
    ]


def _extract_p4_2_vivado(name: str, blob: Dict[str, Any]) -> List[Claim]:
    claims: List[Claim] = []
    for run in blob.get("runs", []):
        timing = run.get("timing") or {}
        if not timing.get("all_paths_met"):
            continue
        claims.append(
            _claim(
                claim_id=f"p4_2_{run.get('name')}_wns",
                category="FPGA synthesis",
                headline=f"P4.2 Vivado WNS ({run.get('name')})",
                value=timing.get("wns_ns"),
                unit="ns",
                tier="synth",
                source_name=name,
                fence_note="Legacy p4_2 Vivado reports; prefer baseline_8x8_current_rtl_synth when present.",
                raw=run,
            )
        )
    return claims


def _extract_packed_dsp_synth(name: str, blob: Dict[str, Any]) -> List[Claim]:
    failed = sum(1 for r in blob.get("runs", []) if "synth_failed" in str(r.get("route_status", "")))
    total = len(blob.get("runs", []))
    return [
        _claim(
            claim_id="packed_dsp_synth_failed",
            category="FPGA synthesis",
            headline="Packed-DSP synthesis attempts",
            value=f"{failed}/{total} synth_failed",
            unit="runs",
            tier="synth",
            source_name=name,
            fence_note="All runs failed: pe_controller loop does not converge. Sim-only for packed path.",
            raw={"runs": [{"name": r.get("name"), "route_status": r.get("route_status")} for r in blob.get("runs", [])]},
            tags=["synth_failed"],
        )
    ]


def _extract_megakernel(name: str, blob: Dict[str, Any]) -> List[Claim]:
    agg = blob.get("aggregate") or {}
    return [
        _claim(
            claim_id="megakernel_launch_reduction_pooled",
            category="CUDA megakernel",
            headline="Pooled CUDA launch reduction vs op-by-op",
            value=agg.get("launch_reduction_vs_op_by_op_pct_pooled"),
            unit="%",
            tier="ci",
            source_name=name,
            fence_note="Count-based stat; regen-stable. Integer launch counts per workload.",
            raw={"launch_reduction_vs_op_by_op_pct_pooled": agg.get("launch_reduction_vs_op_by_op_pct_pooled")},
        ),
        _claim(
            claim_id="megakernel_all_correct",
            category="CUDA megakernel",
            headline="All workloads bit-exact correct",
            value=agg.get("all_workloads_correct"),
            unit="bool",
            tier="ci",
            source_name=name,
            fence_note="Differential oracle gate on fused-region kernels.",
            raw={"all_workloads_correct": agg.get("all_workloads_correct")},
        ),
        _claim(
            claim_id="megakernel_latency_reduction_median",
            category="CUDA megakernel",
            headline="Latency reduction vs op-by-op (median)",
            value=agg.get("latency_reduction_vs_op_by_op_pct_median"),
            unit="%",
            tier="ci",
            source_name=name,
            fence_note="[needs-locked-clock-artifact] — not regen-stable on unlocked GPU clocks.",
            raw={"latency_reduction_vs_op_by_op_pct_median": agg.get("latency_reduction_vs_op_by_op_pct_median")},
            tags=["unstable_latency"],
        ),
    ]


def _extract_cublas(name: str, blob: Dict[str, Any]) -> List[Claim]:
    agg = blob.get("aggregate") or {}
    return [
        _claim(
            claim_id="cublas_gap_median",
            category="CUDA baseline",
            headline="Gap vs cuBLAS GEMV (median of shapes)",
            value=agg.get("cublas_gap_pct_median_of_shapes"),
            unit="%",
            tier="ci",
            source_name=name,
            fence_note="[needs-locked-clock-artifact] — latency medians unstable on unlocked clocks.",
            raw={"cublas_gap_pct_median_of_shapes": agg.get("cublas_gap_pct_median_of_shapes")},
            tags=["unstable_latency"],
        ),
        _claim(
            claim_id="cublaslt_int8_gap_median",
            category="CUDA baseline",
            headline="Gap vs cuBLASLt IMMA INT8 (median)",
            value=agg.get("cublaslt_int8_gap_pct_median_of_shapes"),
            unit="%",
            tier="ci",
            source_name=name,
            fence_note="Dtype-matched INT8 comparison; [needs-locked-clock-artifact] for latency.",
            raw={"cublaslt_int8_gap_pct_median_of_shapes": agg.get("cublaslt_int8_gap_pct_median_of_shapes")},
            tags=["unstable_latency"],
        ),
    ]


def _extract_real_model_e2e(name: str, blob: Dict[str, Any]) -> List[Claim]:
    return [
        _claim(
            claim_id="resnet18_eager_parity",
            category="Real model (CUDA)",
            headline="ResNet-18 eager parity (all seeds)",
            value=blob.get("all_cases_within_tolerance_vs_eager"),
            unit="bool",
            tier="ci",
            source_name=name,
            fence_note="CUDA graph backend vs eager PyTorch; 3 seeds in artifact.",
            raw={"model": blob.get("model"), "graph_op_count": blob.get("graph_op_count")},
        ),
        _claim(
            claim_id="resnet18_inductor_parity",
            category="Real model (CUDA)",
            headline="ResNet-18 Inductor parity (all seeds)",
            value=blob.get("all_cases_within_tolerance_vs_inductor"),
            unit="bool",
            tier="ci",
            source_name=name,
            fence_note="TorchInductor oracle in isolated subprocess.",
            raw={"execution_backend": blob.get("execution_backend")},
        ),
    ]


def _extract_cost_model_heldout(name: str, blob: Dict[str, Any]) -> List[Claim]:
    test = (blob.get("latency_prediction") or {}).get("test_metrics") or {}
    sel_summary = (blob.get("selection_quality") or {}).get("summary") or {}
    return [
        _claim(
            claim_id="cost_model_heldout_log_r2",
            category="Cost model",
            headline="Held-out log R²",
            value=test.get("log_r2"),
            unit="ratio",
            tier="ci",
            source_name=name,
            fence_note="Replay-only; no live CUDA in this artifact.",
            raw=test,
        ),
        _claim(
            claim_id="cost_model_heldout_mape",
            category="Cost model",
            headline="Held-out latency MAPE",
            value=test.get("mape_pct"),
            unit="%",
            tier="ci",
            source_name=name,
            fence_note="Generalization to unseen layer shapes.",
            raw=test,
        ),
        _claim(
            claim_id="cost_model_heldout_regret_median",
            category="Cost model",
            headline="Held-out selection regret (median)",
            value=sel_summary.get("median_regret_pct"),
            unit="%",
            tier="ci",
            source_name=name,
            fence_note="Measured regret vs autotuner-best on held-out shapes.",
            raw=sel_summary,
        ),
    ]


def _extract_scheduler_cycles(name: str, blob: Dict[str, Any]) -> List[Claim]:
    agg = blob.get("aggregate") or {}
    return [
        _claim(
            claim_id="scheduler_cycle_reduction",
            category="Scheduler",
            headline="Aggregate ISA-sim cycle reduction",
            value=agg.get("cycle_reduction_pct"),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Python ISA simulator cycles; RTL absolute counts differ.",
            raw=agg,
        ),
        _claim(
            claim_id="scheduler_fetch_bytes_match",
            category="Scheduler",
            headline="All shapes fetch_bytes match naive",
            value=agg.get("all_fetch_bytes_match"),
            unit="bool",
            tier="sim",
            source_name=name,
            fence_note="Bit-exact fetch_bytes invariant vs naive lowering.",
            raw={"all_fetch_bytes_match": agg.get("all_fetch_bytes_match")},
        ),
    ]


def _extract_scheduler_rtl_crosscheck(name: str, blob: Dict[str, Any]) -> List[Claim]:
    prefix = name.removesuffix(".json")
    expected = blob.get("expected") or {}
    rtl = blob.get("rtl") or {}
    claims: List[Claim] = []
    if expected.get("reduction_permille") is not None:
        claims.append(
            _claim(
                claim_id=f"{prefix}_rtl_reduction_permille",
                category="ISA vs RTL",
                headline="RTL cycle reduction (32×32)",
                value=expected.get("reduction_permille"),
                unit="permille",
                tier="sim",
                source_name=name,
                fence_note="iverilog RTL sim; percentage reduction within ±20 permille of ISA sim.",
                raw={"expected": expected, "rtl": rtl, "status": blob.get("status")},
            )
        )
    if expected.get("fetch_bytes_invariant_simulator") is not None:
        claims.append(
            _claim(
                claim_id=f"{prefix}_rtl_fetch_invariant",
                category="ISA vs RTL",
                headline="RTL fetch_bytes invariant",
                value=expected.get("fetch_bytes_invariant_simulator"),
                unit="bool",
                tier="sim",
                source_name=name,
                fence_note="RTL_naive_bytes === RTL_scheduled_bytes across fetch stream.",
                raw={"fetch_bytes_n": expected.get("fetch_bytes_n")},
            )
        )
    if not claims:
        status = blob.get("status") or rtl.get("status")
        if status:
            claims.append(
                _claim(
                    claim_id=f"{prefix}_status",
                    category="ISA vs RTL",
                    headline="RTL cross-check status",
                    value=status,
                    unit="status",
                    tier="sim",
                    source_name=name,
                    fence_note="iverilog scheduler RTL cross-check artifact.",
                    raw=blob,
                )
            )
    return claims


def _extract_scheduler_rtl_crosscheck_bigmlp(name: str, blob: Dict[str, Any]) -> List[Claim]:
    agg = blob.get("aggregate") or {}
    return [
        _claim(
            claim_id="scheduler_rtl_bigmlp_all_ok",
            category="ISA vs RTL",
            headline="BigMLP scheduler RTL cases all OK",
            value=agg.get("all_cases_ok"),
            unit="bool",
            tier="sim",
            source_name=name,
            fence_note="5 blocked-FC shapes; RTL byte-exact vs ISA-scheduled programs.",
            raw=agg,
        ),
        _claim(
            claim_id="scheduler_rtl_bigmlp_byte_exact",
            category="ISA vs RTL",
            headline="All cases RTL byte-exact",
            value=agg.get("all_cases_rtl_byte_exact"),
            unit="bool",
            tier="sim",
            source_name=name,
            fence_note="iverilog cross-check across multiple MLP shapes.",
            raw={"ok_case_count": agg.get("ok_case_count"), "case_count": agg.get("case_count")},
        ),
    ]


def _extract_tiling(name: str, blob: Dict[str, Any]) -> List[Claim]:
    summary = blob.get("summary") or {}
    return [
        _claim(
            claim_id="tiling_all_bit_identical",
            category="Tiling",
            headline="All workloads bit-identical to oracle",
            value=summary.get("all_bit_identical_to_oracle"),
            unit="bool",
            tier="ci",
            source_name=name,
            fence_note="ISA sim vs NumPy int32 oracle; sim-only.",
            raw=summary,
        ),
    ]


def _extract_fusion(name: str, blob: Dict[str, Any]) -> List[Claim]:
    claims: List[Claim] = []
    summary = (blob.get("numpy_reference") or {}).get("summary") or blob.get("summary") or {}
    if summary:
        claims.append(
            _claim(
                claim_id="fusion_numpy_median_delta",
                category="Fusion",
                headline="NumPy-reference fusion throughput delta (median)",
                value=summary.get("median_throughput_delta_pct"),
                unit="%",
                tier="ci",
                source_name=name,
                fence_note="CPU NumPy reference path; positive => fusion wins.",
                raw=summary,
            )
        )
    cuda = (blob.get("cuda_fusion") or {}).get("result") or {}
    if cuda:
        claims.append(
            _claim(
                claim_id="fusion_cuda_all_correct",
                category="Fusion",
                headline="CUDA fusion correctness (all workloads)",
                value=cuda.get("all_correctness_within_tolerance"),
                unit="bool",
                tier="ci",
                source_name=name,
                fence_note="Eager vs Inductor FP32; GPU fusion speedup not claimed at tiny shapes.",
                raw={"status": cuda.get("status")},
            )
        )
    return claims


def _extract_superopt(name: str, blob: Dict[str, Any]) -> List[Claim]:
    agg = blob.get("aggregate") or {}
    return [
        _claim(
            claim_id="superopt_median_cost_reduction",
            category="Superoptimizer",
            headline="Median modeled ISA cost reduction",
            value=agg.get("cost_reduction_pct_median"),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Equality-saturation e-graph; sim-only isa_cycle_model.",
            raw=agg,
        ),
        _claim(
            claim_id="superopt_max_cost_reduction",
            category="Superoptimizer",
            headline="Max modeled ISA cost reduction",
            value=agg.get("cost_reduction_pct_max"),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Random-corpus baseline is often 0%; max shows upper bound.",
            raw={"pct_graphs_with_any_win": agg.get("pct_graphs_with_any_win")},
        ),
    ]


def _extract_multi_pe(name: str, blob: Dict[str, Any]) -> List[Claim]:
    claims: List[Claim] = []
    for case in blob.get("per_case", []):
        two = case.get("two_pe") or {}
        one = case.get("one_pe") or {}
        if two.get("schedule_emitted") and one.get("cycle_count_sequential"):
            seq = int(one["cycle_count_sequential"])
            par = int(two.get("cycle_count_parallel_estimate", seq))
            reduction = round((1 - par / seq) * 100, 2) if seq else 0
            claims.append(
                _claim(
                    claim_id=f"multi_pe_{case.get('case_name')}_reduction",
                    category="Multi-PE ISA",
                    headline=f"2-PE cycle reduction ({case.get('case_name')})",
                    value=reduction,
                    unit="%",
                    tier="sim",
                    source_name=name,
                    fence_note="ISA simulator parallel estimate; no RTL multi-PE.",
                    raw={"case": case.get("case_name"), "one_pe_cycles": seq, "two_pe_cycles": par},
                )
            )
    return claims


def _extract_board_fit(name: str, blob: Dict[str, Any]) -> List[Claim]:
    agg = blob.get("aggregate") or {}
    per_board = agg.get("per_board") or {}
    bram = per_board.get("pynqz2_bram_max") or {}
    baseline = per_board.get("pynqz2_baseline") or {}
    artix = per_board.get("artix_a7100t_bram_max") or {}
    claims = [
        _claim(
            claim_id="board_fit_bram_max_fraction",
            category="Board fit",
            headline="Shapes fitting pynqz2_bram_max (PROG_DEPTH=8192)",
            value=bram.get("shapes_fit_fraction"),
            unit="fraction",
            tier="ci",
            source_name=name,
            fence_note="Host lowering audit vs instruction BRAM depth; not on-silicon.",
            raw=bram,
        ),
    ]
    if baseline.get("shapes_fit_count") is not None:
        claims.append(
            _claim(
                claim_id="board_fit_baseline_shapes",
                category="Board fit",
                headline="Shapes fitting pynqz2_baseline (PROG_DEPTH=1024)",
                value=baseline.get("shapes_fit_count"),
                unit="shapes",
                tier="ci",
                source_name=name,
                fence_note="Historical shipping PROG_DEPTH=1024 admits 4/14 shapes.",
                raw=baseline,
            )
        )
    if artix.get("shapes_fit_count") is not None:
        claims.append(
            _claim(
                claim_id="board_fit_artix_bram_max_shapes",
                category="Board fit",
                headline="Shapes fitting artix_a7100t_bram_max (PROG_DEPTH=65536)",
                value=artix.get("shapes_fit_count"),
                unit="shapes",
                tier="ci",
                source_name=name,
                fence_note="Shipping Artix PROG_DEPTH=65536 admits 10/14 shapes; host audit only.",
                raw=artix,
            )
        )
    return claims


def _extract_selection_ab(name: str, blob: Dict[str, Any]) -> List[Claim]:
    agg = blob.get("aggregate") or {}
    return [
        _claim(
            claim_id="selection_ab_regret_median",
            category="Schedule selection",
            headline="Realized regret median (cost-model vs oracle)",
            value=agg.get("realized_regret_pct_median"),
            unit="%",
            tier="ci",
            source_name=name,
            fence_note="Wall-clock NVRTC A/B; schedule_source=cost_model consumed at runtime.",
            raw=agg,
            tags=["unstable_latency"],
        ),
        _claim(
            claim_id="selection_ab_within_5pct",
            category="Schedule selection",
            headline="Fraction within 5% regret",
            value=agg.get("realized_within_5pct_fraction"),
            unit="fraction",
            tier="ci",
            source_name=name,
            fence_note="Count-based quality metric on 24 shapes.",
            raw={"realized_within_5pct_fraction": agg.get("realized_within_5pct_fraction")},
        ),
    ]


def _extract_cost_model_regression(name: str, blob: Dict[str, Any]) -> List[Claim]:
    baseline = blob.get("baseline") or {}
    return [
        _claim(
            claim_id="cost_model_replay_median_error",
            category="Cost model",
            headline="Replay median abs % error",
            value=baseline.get("median_abs_percent_error"),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Measured-data replay; frozen baseline coefficients.",
            raw=baseline,
        ),
    ]


def _extract_batched_gemm(name: str, blob: Dict[str, Any]) -> List[Claim]:
    claims: List[Claim] = []
    for key in ("all_cases_bit_exact_vs_oracle", "all_b1_programs_byte_identical"):
        if key in blob:
            claims.append(
                _claim(
                    claim_id=f"batched_gemm_{key}",
                    category="Batched GEMM",
                    headline=key.replace("_", " "),
                    value=blob[key],
                    unit="bool",
                    tier="ci",
                    source_name=name,
                    fence_note="ISA sim correctness + B=1 byte-identical legacy programs.",
                    raw={key: blob[key]},
                )
            )
    return claims


def _extract_systolic_char(name: str, blob: Dict[str, Any]) -> List[Claim]:
    claims: List[Claim] = []
    flagship = [
        c
        for c in blob.get("cases", [])
        if c.get("shape", {}).get("out_features") == 16
        and c.get("shape", {}).get("in_features") == 16
        and c.get("out_blocks") == 1
    ]
    for case in flagship:
        b = case.get("batch_size")
        measured = case.get("measured") or {}
        claims.append(
            _claim(
                claim_id=f"systolic_pe_occupancy_b{b}",
                category="Systolic array",
                headline=f"PE occupancy (16×16, B={b})",
                value=round(float(measured.get("pe_occupancy", 0)) * 100, 2),
                unit="%",
                tier="sim",
                source_name=name,
                fence_note="RTL perf counters via iverilog; simulated tier.",
                raw=case,
            )
        )
    return claims


def _extract_sim_pass(name: str, blob: Dict[str, Any], *, category: str, headline_field: str = "result") -> List[Claim]:
    if headline_field not in blob and blob.get("status") not in ("ok", "ran", "PASS"):
        return []
    value = blob.get(headline_field) or blob.get("status") or blob.get("result")
    return [
        _claim(
            claim_id=f"{name.removesuffix('.json')}_status",
            category=category,
            headline=f"{category} status",
            value=value,
            unit="status",
            tier="sim",
            source_name=name,
            fence_note="iverilog / ISA-sim artifact; not hardware-executed.",
            raw={"status": blob.get("status"), "result": blob.get("result")},
        )
    ]


def _extract_uart(name: str, blob: Dict[str, Any]) -> List[Claim]:
    rtl = blob.get("rtl_simulation") or blob.get("simulation") or {}
    roundtrip = blob.get("uart_roundtrip") or {}
    passed = rtl.get("rtl_sim_passed")
    if passed is None:
        passed = rtl.get("iverilog_passed")
    if passed is None:
        passed = roundtrip.get("captured_uart_matches_isa_sim")
    return [
        _claim(
            claim_id="uart_preboard_roundtrip",
            category="UART pre-board",
            headline="UART replay sim passed",
            value=passed,
            unit="bool",
            tier="sim",
            source_name=name,
            fence_note="Simulation-only UART roundtrip; not on-board silicon.",
            raw={"rtl_simulation": rtl, "uart_roundtrip": roundtrip},
        ),
    ]


def _extract_cycle_attribution(name: str, blob: Dict[str, Any]) -> List[Claim]:
    """Flagship isolated GEMM: 64x64 B=48 N=16 instruction-stream / load shares."""
    flagship = None
    for case in blob.get("cases") or []:
        shape = case.get("shape") or {}
        if (
            shape.get("out_features") == 64
            and shape.get("in_features") == 64
            and case.get("batch_size") == 48
        ):
            flagship = case
            break
    if flagship is None and (blob.get("cases") or []):
        flagship = blob["cases"][0]
    if not flagship:
        return []
    groups = flagship.get("groups") or {}
    fd = (groups.get("fetch_decode") or {}).get("pct_of_total_program_cycles") or 0.0
    store = (groups.get("store") or {}).get("pct_of_total_program_cycles") or 0.0
    bstore = (groups.get("bstore") or {}).get("pct_of_total_program_cycles") or 0.0
    # Prefer artifact verdict text fraction when present; else reconstruct stream share.
    stream_pct = round((fd + store + bstore) * 100.0, 1)
    reason = str(blob.get("verdict_reason") or "")
    if "instruction-stream=" in reason:
        try:
            stream_pct = float(reason.split("instruction-stream=")[1].split("%")[0])
        except (IndexError, ValueError):
            pass
    load_pct = ((groups.get("load") or {}).get("pct_of_total_program_cycles") or 0.0) * 100.0
    return [
        _claim(
            claim_id="cycle_attr_instruction_stream_pct",
            category="Cycle attribution",
            headline="Instruction-stream share of on-chip cycles (64x64 B=48)",
            value=stream_pct,
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Fast-UART TB on-chip core cycles; isolated GEMM harness, not multi-layer.",
            raw={"groups": groups, "verdict_reason": blob.get("verdict_reason")},
        ),
        _claim(
            claim_id="cycle_attr_load_pct",
            category="Cycle attribution",
            headline="LOAD share of on-chip cycles (64x64 B=48)",
            value=round(load_pct, 1),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Buffer→PE weight LOAD is negligible vs instruction stream.",
            raw=groups.get("load"),
        ),
    ]


def _extract_cycle_attribution_mnist(name: str, blob: Dict[str, Any]) -> List[Claim]:
    groups = blob.get("groups") or {}
    total = blob.get("total_program_cycles")
    compute_pct = ((groups.get("compute") or {}).get("pct_of_total_program_cycles") or 0.0) * 100.0
    claims = [
        _claim(
            claim_id="cycle_attr_mnist_total_cycles",
            category="Cycle attribution (MNIST)",
            headline="Fused MNIST total program cycles (post BSTORE widen)",
            value=total,
            unit="cycles",
            tier="sim",
            source_name=name,
            fence_note="Fused MNIST case1, fast-UART TB, N=16; post BSTORE_WIDTH=8.",
            raw={"total_program_cycles": total, "groups": groups},
        ),
        _claim(
            claim_id="cycle_attr_mnist_compute_pct",
            category="Cycle attribution (MNIST)",
            headline="Cold fused-MNIST compute share",
            value=round(compute_pct, 1),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Cold path with program-embedded weights; contrast steady_state_attribution.json.",
            raw=groups.get("compute"),
        ),
    ]
    return claims


def _extract_bstore_path_measure(name: str, blob: Dict[str, Any]) -> List[Claim]:
    impl = blob.get("implementation") or {}
    post = impl.get("post_widen_mnist_attr") or {}
    speedup = post.get("e2e_speedup_vs_pre_widen")
    claims = []
    if speedup is not None:
        claims.append(
            _claim(
                claim_id="bstore_widen_e2e_speedup",
                category="BSTORE widen",
                headline="BSTORE_WIDTH=8 end-to-end speedup vs pre-widen (fused MNIST)",
                value=round(float(speedup), 2),
                unit="x",
                tier="sim",
                source_name=name,
                fence_note="Multi-layer fused MNIST case1; iverilog attr, not silicon.",
                raw=post,
            )
        )
    measured = blob.get("measured") or {}
    if measured.get("cycles_per_payload_word") is not None:
        claims.append(
            _claim(
                claim_id="bstore_pre_widen_cyc_per_word",
                category="BSTORE widen",
                headline="Pre-widen BSTORE cycles per payload word",
                value=measured.get("cycles_per_payload_word"),
                unit="cyc/word",
                tier="sim",
                source_name=name,
                fence_note="Frozen pre-widen baseline identity: payload*4 + bursts.",
                raw=measured,
            )
        )
    return claims


def _extract_steady_state_attribution(name: str, blob: Dict[str, Any]) -> List[Claim]:
    cold = blob.get("cold") or {}
    steady = blob.get("steady_state") or {}
    cold_share = cold.get("compute_share")
    steady_share = steady.get("mean_compute_share")
    claims = []
    if cold_share is not None:
        claims.append(
            _claim(
                claim_id="steady_state_cold_compute_share",
                category="Buffer-resident weights",
                headline="Cold-path compute share (fused MNIST)",
                value=round(float(cold_share) * 100.0, 1),
                unit="%",
                tier="sim",
                source_name=name,
                fence_note="Program-embedded weights; A5 protocol cold run.",
                raw=cold,
            )
        )
    if steady_share is not None:
        claims.append(
            _claim(
                claim_id="steady_state_compute_share",
                category="Buffer-resident weights",
                headline="Steady-state compute share with buffer-resident weights",
                value=round(float(steady_share) * 100.0, 1),
                unit="%",
                tier="sim",
                source_name=name,
                fence_note="Bit-exact remapped fused MNIST; A5 once + control-only; iverilog.",
                raw=steady,
            )
        )
    return claims


def _extract_program_word_composition(name: str, blob: Dict[str, Any]) -> List[Claim]:
    headline = blob.get("headline") or {}
    frac = headline.get("mnist_fused_embedded_payload_fraction")
    if frac is None:
        return _generic_aggregate(name, blob)
    return [
        _claim(
            claim_id="program_word_mnist_payload_fraction",
            category="Program word composition",
            headline="Fused MNIST embedded BSTORE payload fraction",
            value=round(float(frac) * 100.0, 1),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Host opcode split; weights in program image dominate instruction BRAM.",
            raw=headline,
        ),
        _claim(
            claim_id="program_word_board_fit_median_payload",
            category="Program word composition",
            headline="Board-fit median embedded payload fraction",
            value=round(float(headline.get("board_fit_median_embedded_payload_fraction") or 0) * 100.0, 1),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Median across board-fit audit shapes.",
            raw=headline,
        ),
    ]


def _extract_prog_depth_sweep(name: str, blob: Dict[str, Any]) -> List[Claim]:
    summary = blob.get("summary") or {}
    point = summary.get("prog_depth_65536_point") or {}
    util = point.get("utilization") or {}
    timing = point.get("timing") or {}
    claims = [
        _claim(
            claim_id="prog_depth_65536_closes",
            category="PROG_DEPTH sweep",
            headline="PROG_DEPTH=65536 closes on Artix-7 A7-100T",
            value=summary.get("prog_depth_65536_closes"),
            unit="bool",
            tier="synth",
            source_name=name,
            fence_note="Post-route Vivado; shipping instruction capacity for board-fit 10/14.",
            raw={"summary_status": summary.get("prog_depth_65536_status"), "point": point},
        ),
    ]
    if util.get("dsp_used") is not None:
        claims.append(
            _claim(
                claim_id="prog_depth_65536_dsp",
                category="PROG_DEPTH sweep",
                headline="DSP used at PROG_DEPTH=65536 shipping close",
                value=util.get("dsp_used"),
                unit="DSP",
                tier="synth",
                source_name=name,
                fence_note=f"WNS={timing.get('wns_ns')} ns @ {point.get('clock_period_ns')} ns period.",
                raw=util,
            )
        )
    return claims


def _extract_utpu_cycle_model_heldout(name: str, blob: Dict[str, Any]) -> List[Claim]:
    lat = (blob.get("latency_prediction") or {}).get("test_metrics") or {}
    sel = (blob.get("selection_quality") or {}).get("summary") or {}
    return [
        _claim(
            claim_id="utpu_cycle_model_heldout_log_r2",
            category="uTPU cycle cost model",
            headline="Held-out log R² (RTL total_program_cycles)",
            value=round(float(lat.get("log_r2") or 0), 3),
            unit="",
            tier="sim",
            source_name=name,
            fence_note="80/20 shape split; 5-candidate (batch,hoist) menu — not CUDA's 16-candidate space.",
            raw=lat,
        ),
        _claim(
            claim_id="utpu_cycle_model_heldout_mape",
            category="uTPU cycle cost model",
            headline="Held-out MAPE",
            value=round(float(lat.get("mape_pct") or 0), 2),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Predicts RTL cycles; absolute MAPE is residual analytical error.",
            raw=lat,
        ),
        _claim(
            claim_id="utpu_cycle_model_heldout_regret",
            category="uTPU cycle cost model",
            headline="Held-out selection regret mean (5-candidate menu)",
            value=sel.get("mean_regret_pct"),
            unit="%",
            tier="sim",
            source_name=name,
            fence_note="Zero regret on 5-candidate menu; CUDA held-out mean regret 5.21% is 16-candidate.",
            raw=sel,
        ),
    ]


def _extract_latency_determinism_vs_gpu(name: str, blob: Dict[str, Any]) -> List[Claim]:
    comparison = blob.get("comparison") or {}
    loss = comparison.get("median_latency_loss") or {}
    return [
        _claim(
            claim_id="latency_fpga_cycle_variance",
            category="Latency determinism",
            headline="FPGA/RTL cycle variance across adversarial+random inputs",
            value=comparison.get("fpga_rtl_cycle_variance"),
            unit="cycles",
            tier="sim",
            source_name=name,
            fence_note="iverilog RTL sim; not on-board silicon.",
            raw={"fpga_arm": blob.get("fpga_arm"), "shape": blob.get("shape")},
        ),
        _claim(
            claim_id="latency_median_loss_shipping_83mhz",
            category="Latency determinism",
            headline="Median latency loss vs GPU at shipping ~83 MHz",
            value=round(float(loss.get("median_latency_loss_factor") or 0), 2),
            unit="x",
            tier="sim",
            source_name=name,
            fence_note="Shipping 12 ns close (WNS=+0.271); 100 MHz is ceiling only (WNS=+0.012).",
            raw=loss,
        ),
    ]


def _extract_design_space_sweep(name: str, blob: Dict[str, Any]) -> List[Claim]:
    shipping = blob.get("shipping_point") or {}
    timing = shipping.get("timing") or {}
    ceiling = blob.get("demonstrated_fmax_ceiling") or {}
    ceil_wns = (ceiling.get("timing") or {}).get("wns_ns", ceiling.get("wns_ns"))
    claims = [
        _claim(
            claim_id="design_space_shipping_mhz",
            category="Design-space sweep",
            headline="Shipping close frequency (N=8 INT8 mb48)",
            value=shipping.get("frequency_mhz_constraint") or shipping.get("frequency_mhz"),
            unit="MHz",
            tier="synth",
            source_name=name,
            fence_note=(
                f"Post-route on {blob.get('part', 'xc7a100tcsg324-1')}; "
                f"period={shipping.get('clock_period_ns')} ns; WNS={timing.get('wns_ns')} ns."
            ),
            raw=shipping,
        ),
        _claim(
            claim_id="design_space_shipping_wns",
            category="Design-space sweep",
            headline="Shipping close WNS",
            value=timing.get("wns_ns"),
            unit="ns",
            tier="synth",
            source_name=name,
            fence_note="Thin positive slack at 12 ns / ~83.3 MHz.",
            raw=timing,
        ),
    ]
    if ceiling.get("frequency_mhz_constraint") is not None or ceiling.get("clock_period_ns") == 10.0:
        claims.append(
            _claim(
                claim_id="design_space_fmax_ceiling_mhz",
                category="Design-space sweep",
                headline="Demonstrated Fmax ceiling",
                value=ceiling.get("frequency_mhz_constraint") or 100.0,
                unit="MHz",
                tier="synth",
                source_name=name,
                fence_note=f"Ceiling only; WNS={ceil_wns} ns — do not ship at this margin.",
                raw=ceiling,
            )
        )
    return claims


def _extract_requant_rightsizing(name: str, blob: Dict[str, Any]) -> List[Claim]:
    gate = blob.get("task_c_decision_gate") or {}
    table = gate.get("budget_table") or []
    baseline_dsp = None
    step12_dsp = None
    for row in table:
        if row.get("invalid"):
            continue
        cfg = str(row.get("config") or "")
        if "baseline" in cfg.lower():
            baseline_dsp = row.get("dsp")
        if "Step1+2" in cfg or "step1+2" in cfg.lower():
            step12_dsp = row.get("dsp")
    claims = []
    if baseline_dsp is not None and step12_dsp is not None:
        claims.append(
            _claim(
                claim_id="requant_dsp_192_to_72",
                category="Requant rightsizing",
                headline="DSP after Step1+2 rightsizing (was 192)",
                value=step12_dsp,
                unit="DSP",
                tier="synth",
                source_name=name,
                fence_note=f"Artix-7 A7-100T 8x8 INT8; baseline {baseline_dsp} → Step1+2 {step12_dsp}.",
                raw={"baseline_dsp": baseline_dsp, "step12_dsp": step12_dsp, "gate": gate},
            )
        )
    return claims


def _generic_aggregate(name: str, blob: Dict[str, Any]) -> List[Claim]:
    tier = infer_tier(name, blob)
    claims: List[Claim] = []
    for section in ("aggregate", "summary"):
        data = blob.get(section)
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                continue
            if value is None:
                continue
            claims.append(
                _claim(
                    claim_id=f"{name.removesuffix('.json')}_{section}_{key}",
                    category=name.removesuffix(".json").replace("_", " ").title(),
                    headline=key.replace("_", " "),
                    value=value,
                    unit="",
                    tier=tier,
                    source_name=name,
                    fence_note=f"Auto-extracted from {section} block.",
                    raw={key: value},
                )
            )
    return claims[:6]


EXTRACTORS: Dict[str, Extractor] = {
    "baseline_8x8_current_rtl_synth.json": _extract_baseline_8x8,
    "p4_2_vivado_reports.json": _extract_p4_2_vivado,
    "packed_dsp_synth.json": _extract_packed_dsp_synth,
    "megakernel_payoff.json": _extract_megakernel,
    "cublas_baseline.json": _extract_cublas,
    "real_model_end_to_end.json": _extract_real_model_e2e,
    "cost_model_heldout.json": _extract_cost_model_heldout,
    "scheduler_cycles.json": _extract_scheduler_cycles,
    "scheduler_rtl_crosscheck.json": _extract_scheduler_rtl_crosscheck,
    "scheduler_rtl_crosscheck_bigmlp.json": _extract_scheduler_rtl_crosscheck_bigmlp,
    "tiling_correctness.json": _extract_tiling,
    "fusion_payoff.json": _extract_fusion,
    "superopt_payoff.json": _extract_superopt,
    "multi_pe_sim.json": _extract_multi_pe,
    "board_fit_audit.json": _extract_board_fit,
    "selection_ab.json": _extract_selection_ab,
    "cost_model_regression.json": _extract_cost_model_regression,
    "batched_gemm_correctness.json": _extract_batched_gemm,
    "systolic_characterization.json": _extract_systolic_char,
    "uart_preboard_roundtrip.json": _extract_uart,
    "pe_packed_pair_sim.json": lambda n, b: _extract_sim_pass(n, b, category="Packed PE pair"),
    "pe_array_packed_sim.json": lambda n, b: _extract_sim_pass(n, b, category="Packed PE array"),
    "pe_array_packed_hardened.json": lambda n, b: _extract_sim_pass(n, b, category="Packed PE hardened"),
    "top_packed_smoke.json": lambda n, b: _extract_sim_pass(n, b, category="Top packed smoke"),
    "packed_array_cycle_compare.json": lambda n, b: _extract_sim_pass(n, b, category="Packed cycle compare"),
    "cycle_attribution.json": _extract_cycle_attribution,
    "cycle_attribution_mnist.json": _extract_cycle_attribution_mnist,
    "bstore_path_measure.json": _extract_bstore_path_measure,
    "steady_state_attribution.json": _extract_steady_state_attribution,
    "program_word_composition.json": _extract_program_word_composition,
    "prog_depth_sweep.json": _extract_prog_depth_sweep,
    "utpu_cycle_model_heldout.json": _extract_utpu_cycle_model_heldout,
    "latency_determinism_vs_gpu.json": _extract_latency_determinism_vs_gpu,
    "design_space_sweep.json": _extract_design_space_sweep,
    "requant_rightsizing_synth.json": _extract_requant_rightsizing,
}


def collect_claims() -> Tuple[List[Claim], List[str]]:
    claims: List[Claim] = []
    artifacts_read: List[str] = []
    for path in sorted(BENCH_RESULTS.glob("*.json")):
        name = path.name
        artifacts_read.append(name)
        blob = json.loads(path.read_text(encoding="utf-8"))
        extractor = EXTRACTORS.get(name, _generic_aggregate)
        extracted = extractor(name, blob)
        if not extracted:
            extracted = _generic_aggregate(name, blob)
        claims.extend(extracted)
    return claims, artifacts_read


def validate_claims(claims: Sequence[Claim]) -> None:
    errors: List[str] = []
    seen_ids: Set[str] = set()
    for claim in claims:
        cid = claim.get("id", "")
        if cid in seen_ids:
            errors.append(f"duplicate claim id: {cid}")
        seen_ids.add(cid)
        tier = claim.get("tier", "")
        if tier not in VALID_TIERS:
            errors.append(f"claim {cid}: invalid tier {tier!r}")
        if not claim.get("source_artifact"):
            errors.append(f"claim {cid}: missing source_artifact")
        else:
            artifact_path = REPO_ROOT / claim["source_artifact"]
            if not artifact_path.is_file():
                errors.append(f"claim {cid}: source_artifact missing on disk: {claim['source_artifact']}")
        if claim.get("value") is None and "synth_failed" not in (claim.get("tags") or []):
            errors.append(f"claim {cid}: null value")
    if errors:
        raise SystemExit("Schema lock failed:\n  " + "\n  ".join(errors))


def _shape_of(value: Any) -> Optional[List[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _lowering_summary(lowering: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "kernel_name",
        "mode",
        "program_instruction_words",
        "fits_instruction_bram",
        "instruction_bram_words",
        "array_size",
        "out_blocks",
        "in_blocks",
        "executable_on_current_cuda_path",
        "executable_on_current_fpga_path",
    ):
        if key in lowering:
            out[key] = lowering[key]
    if lowering.get("block_ops") is not None:
        bo = lowering["block_ops"]
        out["block_op_count"] = len(bo) if isinstance(bo, (list, tuple)) else bo
    return out


def _layout_viz_graph(
    ops: Sequence[Dict[str, Any]],
    *,
    graph_inputs: Sequence[str],
    graph_outputs: Sequence[str],
) -> Dict[str, Any]:
    """Layered layout for walkthrough SVG — positions derived from op dependency order."""
    op_names = {op["name"] for op in ops}
    depth: Dict[str, int] = {}
    for name in graph_inputs:
        depth[name] = 0
    changed = True
    while changed:
        changed = False
        for op in ops:
            inputs = [i for i in op.get("inputs", []) if i in op_names or i in graph_inputs]
            d = max([depth.get(i, 0) for i in inputs], default=0) + (1 if inputs else 1)
            if d > depth.get(op["name"], 0):
                depth[op["name"]] = d
                changed = True
    by_depth: Dict[int, List[str]] = {}
    for name in list(graph_inputs):
        by_depth.setdefault(0, []).append(name)
    for op in ops:
        d = depth.get(op["name"], 1)
        by_depth.setdefault(d, []).append(op["name"])

    nodes: List[Dict[str, Any]] = []
    for d, names in sorted(by_depth.items()):
        for i, name in enumerate(names):
            op = next((o for o in ops if o["name"] == name), None)
            nodes.append(
                {
                    "id": name,
                    "label": name,
                    "op": op["op"] if op else ("input" if name in graph_inputs else "output"),
                    "x": 40 + d * 160,
                    "y": 40 + i * 90,
                }
            )

    edges: List[Dict[str, str]] = []
    for op in ops:
        for inp in op.get("inputs", []):
            edges.append({"from": inp, "to": op["name"]})
    for out_name in graph_outputs:
        producers = [op["name"] for op in ops if out_name in op.get("outputs", [])]
        for prod in producers:
            edges.append({"from": prod, "to": out_name})
            nodes.append(
                {
                    "id": out_name,
                    "label": out_name,
                    "op": "output",
                    "x": 40 + (max(by_depth) + 1) * 160,
                    "y": 40,
                }
            )

    seen = {n["id"] for n in nodes}
    nodes = [n for n in nodes if n["id"] in seen]
    unique: List[Dict[str, Any]] = []
    done: Set[str] = set()
    for n in nodes:
        if n["id"] not in done:
            unique.append(n)
            done.add(n["id"])
    return {"nodes": unique, "edges": edges}


def _fx_viz_graph(fx_graph: Any) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []
    y = 40
    for i, node in enumerate(fx_graph.graph.nodes):
        if node.op == "placeholder":
            kind = "input"
        elif node.op == "output":
            kind = "output"
        elif node.op == "call_module":
            kind = str(node.target)
        else:
            kind = node.op
        nodes.append(
            {
                "id": node.name,
                "label": node.name,
                "op": kind,
                "x": 40 + i * 130,
                "y": y + (i % 2) * 50,
            }
        )
    for node in fx_graph.graph.nodes:
        for arg in node.args:
            src = getattr(arg, "name", None)
            if src:
                edges.append({"from": src, "to": node.name})
    return {"nodes": nodes, "edges": edges}


def _pass_op_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before_ops = before.get("ops", [])
    after_ops = after.get("ops", [])
    before_names = {op["name"] for op in before_ops}
    after_names = {op["name"] for op in after_ops}
    return {
        "removed": sorted(before_names - after_names),
        "added": sorted(after_names - before_names),
        "before_kinds": [op.get("op") for op in before_ops],
        "after_kinds": [op.get("op") for op in after_ops],
    }


def _build_walkthrough_frames(cuda_result: Any, utpu_result: Any, *, input_shape: Optional[List[int]]) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    frames.append(
        {
            "id": "pytorch",
            "stage": "pytorch",
            "title": "PyTorch module",
            "caption": f"TinyVisualMLP traced with example input shape {input_shape}. This is the scoped demo model — not a general compiler claim.",
            "effect": "enter",
            "graph": {
                "nodes": [
                    {"id": "x", "label": "input x", "op": "input", "x": 40, "y": 80},
                    {"id": "fc1", "label": "fc1", "op": "Linear(4→3)", "x": 200, "y": 40},
                    {"id": "relu", "label": "relu", "op": "ReLU", "x": 360, "y": 80},
                    {"id": "fc2", "label": "fc2", "op": "Linear(3→2)", "x": 520, "y": 40},
                    {"id": "out", "label": "output", "op": "output", "x": 680, "y": 80},
                ],
                "edges": [
                    {"from": "x", "to": "fc1"},
                    {"from": "fc1", "to": "relu"},
                    {"from": "relu", "to": "fc2"},
                    {"from": "fc2", "to": "out"},
                ],
            },
        }
    )

    if cuda_result.fx_graph is not None:
        frames.append(
            {
                "id": "fx",
                "stage": "fx",
                "title": "torch.fx trace",
                "caption": "symbolic_trace captures call_module nodes — the compiler's entry point.",
                "effect": "trace",
                "graph": _fx_viz_graph(cuda_result.fx_graph),
            }
        )

    imported = cuda_result.pass_records[0].before if cuda_result.pass_records else {"ops": [], "inputs": [], "outputs": []}
    frames.append(
        {
            "id": "ir_import",
            "stage": "ir",
            "title": "Graph IR import",
            "caption": "FX graph lowered to Graph IR op nodes with shape metadata.",
            "effect": "import",
            "graph": _layout_viz_graph(
                imported.get("ops", []),
                graph_inputs=imported.get("inputs", []),
                graph_outputs=imported.get("outputs", []),
            ),
        }
    )

    for record in cuda_result.pass_records:
        delta = _pass_op_delta(record.before, record.after)
        kinds_changed = delta["before_kinds"] != delta["after_kinds"]
        meta_changed = record.before.get("metadata") != record.after.get("metadata")
        if not kinds_changed and not meta_changed:
            continue
        effect = "fuse" if record.pass_name == "linear_relu_fusion" else "transform"
        caption = f"Pass `{record.pass_name}`"
        if kinds_changed:
            caption += f": {delta['before_kinds']} → {delta['after_kinds']}"
        else:
            caption += " updates planner metadata (buffers / legality)."
        frames.append(
            {
                "id": f"pass_{record.pass_name}",
                "stage": "pass",
                "title": record.pass_name.replace("_", " "),
                "caption": caption,
                "effect": effect,
                "pass_delta": delta,
                "graph": _layout_viz_graph(
                    record.after.get("ops", []),
                    graph_inputs=record.after.get("inputs", []),
                    graph_outputs=record.after.get("outputs", []),
                ),
            }
        )

    final_ops = []
    if cuda_result.graph_ir is not None:
        final_ops = [
            {"name": op.name, "op": op.op, "inputs": list(op.inputs), "outputs": list(op.outputs)}
            for op in cuda_result.graph_ir.ops
        ]
    frames.append(
        {
            "id": "ir_final",
            "stage": "ir",
            "title": "Final Graph IR",
            "caption": f"Ready for dual-backend lowering — {len(final_ops)} ops after pass pipeline.",
            "effect": "settle",
            "graph": _layout_viz_graph(
                final_ops,
                graph_inputs=list(cuda_result.graph_ir.inputs) if cuda_result.graph_ir else [],
                graph_outputs=list(cuda_result.graph_ir.outputs) if cuda_result.graph_ir else [],
            ),
        }
    )

    cuda_cards = [
        {
            "graph_op": op.graph_op,
            "op": op.op,
            "target": op.target,
            **_lowering_summary(op.lowering if isinstance(op.lowering, dict) else {}),
        }
        for op in cuda_result.backend_ops
    ]
    frames.append(
        {
            "id": "cuda_lower",
            "stage": "cuda",
            "title": "CUDA NVRTC lowering",
            "caption": "Blocked-FC kernels emitted per graph op. Wider CUDA path supports ~15 op kinds on other models.",
            "effect": "lower_cuda",
            "graph": _layout_viz_graph(
                final_ops,
                graph_inputs=list(cuda_result.graph_ir.inputs) if cuda_result.graph_ir else [],
                graph_outputs=list(cuda_result.graph_ir.outputs) if cuda_result.graph_ir else [],
            ),
            "backends": {"cuda": cuda_cards},
        }
    )

    utpu_cards = [
        {
            "graph_op": op.graph_op,
            "op": op.op,
            "target": op.target,
            **_lowering_summary(op.lowering if isinstance(op.lowering, dict) else {}),
        }
        for op in utpu_result.backend_ops
    ]
    total_words = sum(int(c.get("program_instruction_words", 0) or 0) for c in utpu_cards)
    frames.append(
        {
            "id": "utpu_lower",
            "stage": "utpu",
            "title": "uTPU ISA lowering",
            "caption": f"Same IR → INT8 blocked-FC ISA programs ({total_words} instruction words total). Simulation only — board P0 open.",
            "effect": "lower_utpu",
            "graph": _layout_viz_graph(
                final_ops,
                graph_inputs=list(utpu_result.graph_ir.inputs) if utpu_result.graph_ir else [],
                graph_outputs=list(utpu_result.graph_ir.outputs) if utpu_result.graph_ir else [],
            ),
            "backends": {"utpu": utpu_cards},
        }
    )

    return frames


def _build_pipeline_graph(model: Any, example_inputs: Any, *, model_id: str, array_size: int = 16) -> Dict[str, Any]:
    if str(HOST_DIR) not in sys.path:
        sys.path.insert(0, str(HOST_DIR))
    from pytorch_compiler import compile_model

    cuda_result = compile_model(model, example_inputs, target="cuda", array_size=array_size)
    utpu_result = compile_model(model, example_inputs, target="utpu", array_size=array_size)

    stage_defs = [
        ("fx", "torch.fx", cuda_result),
        ("ir", "Graph IR", cuda_result),
        ("passes", "Pass pipeline", cuda_result),
        ("cuda", "CUDA lowering", cuda_result),
        ("utpu", "uTPU ISA lowering", utpu_result),
    ]

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    x_step = 280
    for idx, (sid, label, result) in enumerate(stage_defs):
        op_count = 0
        op_kinds: List[str] = []
        detail: Dict[str, Any] = {}
        if sid == "fx" and result.fx_graph is not None:
            op_count = len(list(result.fx_graph.graph.nodes))
            detail["node_count"] = op_count
            detail["fx_nodes"] = [
                {"name": n.name, "op": n.op, "target": str(n.target)}
                for n in result.fx_graph.graph.nodes
            ]
        elif sid == "ir" and result.graph_ir is not None:
            op_count = len(result.graph_ir.ops)
            op_kinds = [op.op for op in result.graph_ir.ops]
            detail["ops"] = [
                {"name": op.name, "op": op.op, "inputs": list(op.inputs), "outputs": list(op.outputs)}
                for op in result.graph_ir.ops
            ]
        elif sid == "passes":
            op_count = len(result.pass_records)
            detail["passes"] = [
                {
                    "pass_name": r.pass_name,
                    "changed": r.before != r.after,
                    "op_delta": _pass_op_delta(r.before, r.after),
                }
                for r in result.pass_records
            ]
        elif sid in ("cuda", "utpu"):
            op_count = len(result.backend_ops)
            op_kinds = [op.op for op in result.backend_ops]
            detail["lowered_ops"] = [
                {
                    "graph_op": op.graph_op,
                    "op": op.op,
                    "target": op.target,
                    "notes": list(op.notes),
                    "lowering": _lowering_summary(op.lowering if isinstance(op.lowering, dict) else {}),
                }
                for op in result.backend_ops
            ]
            detail["fallback_ops"] = [op.graph_op for op in (result.plan.fallback_ops if result.plan else [])]
            detail["unsupported_ops"] = [op.graph_op for op in (result.plan.unsupported_ops if result.plan else [])]

        nodes.append(
            {
                "id": f"{model_id}_{sid}",
                "type": "pipelineStage",
                "position": {"x": idx * x_step, "y": 0},
                "data": {
                    "label": label,
                    "stage_id": sid,
                    "op_count": op_count,
                    "op_kinds": op_kinds,
                    "detail": detail,
                    "target": result.target if sid in ("cuda", "utpu") else None,
                },
            }
        )
        if idx > 0:
            prev = stage_defs[idx - 1][0]
            edges.append(
                {
                    "id": f"{model_id}_{prev}->{sid}",
                    "source": f"{model_id}_{prev}",
                    "target": f"{model_id}_{sid}",
                }
            )

    cuda_ops = len(cuda_result.backend_ops) + len(cuda_result.plan.fallback_ops if cuda_result.plan else [])
    utpu_native = [op.op for op in (utpu_result.plan.lowered_ops if utpu_result.plan else [])]
    walkthrough = _build_walkthrough_frames(cuda_result, utpu_result, input_shape=_shape_of(example_inputs))

    return {
        "id": model_id,
        "name": cuda_result.model_name,
        "array_size": array_size,
        "example_input_shape": _shape_of(example_inputs),
        "nodes": nodes,
        "edges": edges,
        "walkthrough": {
            "replay_note": "Recorded compile_model replay — not a live compile. Press Play to step through captured stages.",
            "frame_count": len(walkthrough),
            "frames": walkthrough,
        },
        "coverage": {
            "cuda_backend_ops": cuda_ops,
            "utpu_native_ops": utpu_native,
            "asymmetry_note": (
                "CUDA backend covers ~15 graph-op kinds with NVRTC; uTPU natively lowers "
                "LINEAR/LINEAR_RELU with conv2d-via-im2col + batched_matmul extensions on wider branches."
            ),
        },
        "summary": {
            "cuda_ok": bool(cuda_result.ok),
            "utpu_ok": bool(utpu_result.ok),
            "utpu_instruction_words": sum(
                int(op.lowering.get("program_instruction_words", 0) or 0)
                for op in utpu_result.backend_ops
                if isinstance(op.lowering, dict)
            ),
        },
    }


def build_pipelines() -> List[Dict[str, Any]]:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return []

    if str(HOST_DIR) not in sys.path:
        sys.path.insert(0, str(HOST_DIR))

    class TinyVisualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 3, bias=False)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                self.fc1.weight.copy_(
                    torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                        ]
                    )
                )
                self.fc2.weight.copy_(torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.relu(self.fc1(x)))

    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    model = TinyVisualMLP().eval()
    return [_build_pipeline_graph(model, x, model_id="tiny_visual_mlp", array_size=16)]


def build_systolic_data() -> Dict[str, Any]:
    path = BENCH_RESULTS / "systolic_characterization.json"
    if not path.is_file():
        return {"tier": "sim", "status": "missing_artifact", "cases": [], "streaming_ceiling": None}

    blob = json.loads(path.read_text(encoding="utf-8"))
    array_size = int(blob.get("array_size", 16))
    cases: List[Dict[str, Any]] = []
    for row in blob.get("cases", []):
        shape = row.get("shape") or {}
        if shape.get("out_features") != 16 or shape.get("in_features") != 16:
            continue
        measured = row.get("measured") or {}
        model = row.get("model") or {}
        b = int(row.get("batch_size", 1))
        cases.append(
            {
                "batch_size": b,
                "pe_occupancy": measured.get("pe_occupancy"),
                "rtl_busy_counter": measured.get("rtl_busy_counter"),
                "rtl_cycle_counter": measured.get("rtl_cycle_counter"),
                "busy_fraction": measured.get("busy_fraction"),
                "model_per_tile_busy": model.get("per_tile_busy_cycles"),
                "streaming_ceiling": (2 * array_size + b - 2) / (array_size * array_size) if array_size else None,
            }
        )

    cases.sort(key=lambda c: c["batch_size"])
    return {
        "tier": "sim",
        "source_artifact": _rel_artifact(path.name),
        "array_size": array_size,
        "streaming_ceiling_formula": "B/(2N+B) occupancy ceiling; per-tile busy ≈ 2N+B-2",
        "cases": cases,
    }


def build_verify_ladder(artifacts_read: Sequence[str]) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []

    def add(step_id: str, label: str, artifact: str, status: Any, detail: str, tier: str = "sim") -> None:
        if artifact not in artifacts_read:
            return
        path = BENCH_RESULTS / artifact
        blob = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        steps.append(
            {
                "id": step_id,
                "label": label,
                "tier": tier,
                "source_artifact": _rel_artifact(artifact),
                "status": status,
                "detail": detail,
                "raw": blob,
            }
        )

    if "batched_gemm_correctness.json" in artifacts_read:
        p = BENCH_RESULTS / "batched_gemm_correctness.json"
        b = json.loads(p.read_text(encoding="utf-8"))
        add(
            "isa_batched_gemm",
            "ISA batched GEMM vs oracle",
            "batched_gemm_correctness.json",
            b.get("all_cases_bit_exact_vs_oracle", b.get("status")),
            "Host ISA sim bit-exact vs NumPy oracle across shape×batch sweep.",
            tier="ci",
        )

    if "scheduler_rtl_crosscheck.json" in artifacts_read:
        p = BENCH_RESULTS / "scheduler_rtl_crosscheck.json"
        b = json.loads(p.read_text(encoding="utf-8"))
        rtl = b.get("rtl") or {}
        add(
            "scheduler_isa_rtl",
            "Scheduler ISA vs RTL cross-check",
            "scheduler_rtl_crosscheck.json",
            rtl.get("status", b.get("status")),
            "iverilog tb_scheduler_cycles.sv; ±20 permille reduction tolerance.",
        )

    if "pe_packed_pair_sim.json" in artifacts_read:
        p = BENCH_RESULTS / "pe_packed_pair_sim.json"
        b = json.loads(p.read_text(encoding="utf-8"))
        add(
            "packed_pe_bitmatch",
            "Packed PE pair RTL sim",
            "pe_packed_pair_sim.json",
            b.get("result", b.get("status")),
            f"{b.get('vector_count', '?')} random vectors; column_depth={b.get('column_depth')}.",
        )

    if "systolic_characterization.json" in artifacts_read:
        add(
            "systolic_rtl_counters",
            "Systolic RTL perf counters",
            "systolic_characterization.json",
            blob.get("status") if (blob := json.loads((BENCH_RESULTS / "systolic_characterization.json").read_text())) else "ok",
            "PE occupancy from rtl/top perf counters; simulated tier.",
        )

    synth_name = (
        "baseline_8x8_current_rtl_synth.json"
        if (BENCH_RESULTS / "baseline_8x8_current_rtl_synth.json").is_file()
        else "p4_2_vivado_reports.json"
    )
    if synth_name in artifacts_read:
        p = BENCH_RESULTS / synth_name
        b = json.loads(p.read_text(encoding="utf-8"))
        if synth_name.startswith("baseline_8x8"):
            closed = b.get("closed_config") or {}
            timing = closed.get("timing") or {}
            add(
                "vivado_timing_closed",
                "Vivado timing closed (current RTL)",
                synth_name,
                timing.get("all_paths_met"),
                f"WNS {timing.get('wns_ns')} ns @ {closed.get('frequency_mhz')} MHz; bit generated, not executed on board.",
                tier="synth",
            )
        else:
            runs = [r for r in b.get("runs", []) if (r.get("timing") or {}).get("all_paths_met")]
            add(
                "vivado_timing_closed",
                "Vivado timing closed (p4_2)",
                synth_name,
                bool(runs),
                f"{len(runs)} run(s) met timing; superseded by baseline_8x8 when present.",
                tier="synth",
            )

    steps.append(
        {
            "id": "silicon_p0_open",
            "label": "On-board FPGA execution",
            "tier": "silicon",
            "source_artifact": None,
            "status": "open",
            "detail": "P0 — board arrives ~mid-July; no hardware-executed tier numbers in repo.",
            "raw": None,
        }
    )

    return {"steps": steps}


def emit_bundle(claims: List[Claim], artifacts_read: List[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts_read),
        "claim_count": len(claims),
        "artifacts": [_rel_artifact(a) for a in artifacts_read],
        "github_repo": GITHUB_REPO,
        "silicon_tier_note": "P0 on-board execution is OPEN — silicon tier renders empty until captured.",
    }
    evidence = {"meta": meta, "claims": claims, "tiers": [
        {"key": "sim", "label": "Simulated / ISA-sim", "color": "amber", "meaning": "iverilog / ISA-sim / cost-model"},
        {"key": "ci", "label": "CI-validated host", "color": "blue", "meaning": "passes in .github/workflows/ci.yml"},
        {"key": "synth", "label": "Synthesized (P&R, timing-closed)", "color": "purple", "meaning": "Vivado place-and-route, timing met"},
        {"key": "silicon", "label": "Hardware-executed", "color": "green", "meaning": "on-board capture — OPEN"},
    ]}
    (OUT_DIR / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    pipelines = {"pipelines": build_pipelines()}
    (OUT_DIR / "pipelines.json").write_text(json.dumps(pipelines, indent=2), encoding="utf-8")
    systolic = build_systolic_data()
    (OUT_DIR / "systolic.json").write_text(json.dumps(systolic, indent=2), encoding="utf-8")
    ladder = build_verify_ladder(artifacts_read)
    (OUT_DIR / "verify_ladder.json").write_text(json.dumps(ladder, indent=2), encoding="utf-8")


def main() -> None:
    claims, artifacts_read = collect_claims()
    validate_claims(claims)
    emit_bundle(claims, artifacts_read)
    print(f"Wrote {len(claims)} claims from {len(artifacts_read)} artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
