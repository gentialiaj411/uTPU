import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from generate_fused_rtl_test_vectors import _fc_block_reference, generate_vectors
from isa_simulator import simulate_program_bytes
from lowering_fused_mlp_utpu import lower_fused_mlp_program_utpu

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "multi_pe_sim.json"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _case_tensors(case: dict):
    if case["name"] == "case1_single_k":
        fc1_w = np.zeros((4, 16), dtype=np.int8)
        fc1_w[0, 0:4] = [1, -1, 2, 0]
        fc1_w[1, 0:4] = [0, 1, 1, -1]
        fc1_w[2, 0:4] = [-1, 0, 1, 1]
        fc1_w[3, 0:4] = [2, 1, 0, -2]
        x = np.zeros(16, dtype=np.int8)
        x[0:4] = [2, -1, 1, 3]
        fc2_w = np.zeros((4, 4), dtype=np.int8)
        fc2_w[0] = [1, 0, -1, 2]
        fc2_w[1] = [0, 2, 1, -1]
        fc2_w[2] = [1, 1, 1, 1]
        fc2_w[3] = [-1, 0, 2, 0]
    else:
        fc1_w = np.zeros((4, 32), dtype=np.int8)
        fc1_w[0, 0:8] = [1, 1, -1, 0, 2, -2, 1, 0]
        fc1_w[0, 16:24] = [0, 1, 1, -1, 0, 2, -1, 1]
        fc1_w[1, 0:8] = [0, -1, 2, 1, -1, 0, 1, 1]
        fc1_w[1, 16:24] = [1, 0, -2, 1, 1, -1, 0, 2]
        fc1_w[2, 0:8] = [2, 0, 1, -1, 1, 1, 0, -2]
        fc1_w[2, 16:24] = [-1, 2, 0, 1, -1, 0, 1, 1]
        fc1_w[3, 0:8] = [1, -2, 0, 1, 0, 1, -1, 2]
        fc1_w[3, 16:24] = [2, 1, -1, 0, 1, -2, 1, 0]
        x = np.zeros(32, dtype=np.int8)
        x[0:8] = [1, -1, 2, 0, -2, 1, 1, -1]
        x[16:24] = [0, 2, -1, 1, 1, -2, 0, 1]
        fc2_w = np.array(
            [
                [1, -1, 2, 0],
                [0, 2, 1, -1],
                [1, 1, 0, 1],
                [-1, 0, 2, 1],
            ],
            dtype=np.int8,
        )
    return fc1_w, fc2_w, x


def build_report() -> dict:
    vectors = generate_vectors()
    per_case = []
    for case in vectors["cases"]:
        fc1_w, fc2_w, x = _case_tensors(case)
        ref = _fc_block_reference(fc1_w, fc2_w, x, fc1_relu=True, fc2_relu=False)
        one_pe = lower_fused_mlp_program_utpu(fc1_w, fc2_w, x, num_pe=1)
        two_pe = lower_fused_mlp_program_utpu(fc1_w, fc2_w, x, num_pe=2)
        sim1 = simulate_program_bytes(one_pe["program"], num_pe=1)
        sim2 = simulate_program_bytes(two_pe["program"], num_pe=2)
        per_case.append(
            {
                "case_name": case["name"],
                "reference_fetch_bytes": ref["fc2_fetch_bytes"],
                "one_pe": {
                    "mode": one_pe.get("mode"),
                    "program_words": one_pe["program_instruction_words"],
                    "fetch_bytes": sim1.fetch_bytes,
                    "cycle_count_sequential": sim1.cycle_count_sequential,
                    "matches_reference": sim1.fetch_bytes == ref["fc2_fetch_bytes"],
                },
                "two_pe": {
                    "mode": two_pe.get("mode"),
                    "program_words": two_pe["program_instruction_words"],
                    "schedule_emitted": bool(two_pe.get("multi_pe_schedule_emitted", False)),
                    "fetch_bytes": sim2.fetch_bytes,
                    "cycle_count_sequential": sim2.cycle_count_sequential,
                    "cycle_count_parallel_estimate": sim2.cycle_count_parallel_estimate,
                    "matches_reference": sim2.fetch_bytes == ref["fc2_fetch_bytes"],
                    "matches_one_pe_fetch": sim2.fetch_bytes == sim1.fetch_bytes,
                },
            }
        )

    case2 = next(item for item in per_case if item["case_name"] == "case2_multi_k")
    seq1 = float(case2["one_pe"]["cycle_count_sequential"])
    seq2 = float(case2["two_pe"]["cycle_count_sequential"])
    par2 = float(case2["two_pe"]["cycle_count_parallel_estimate"])
    return {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "methodology": {
            "simulator": "firmware/host/isa_simulator.py",
            "lowering": "firmware/host/lowering_fused_mlp_utpu.py",
            "reference": "generate_fused_rtl_test_vectors._fc_block_reference",
            "notes": (
                "Parallel cycle estimate sums max per-PE work between barriers; "
                "2-PE schedule requires >=2 FC1 K-blocks (case2_multi_k)."
            ),
        },
        "summary": {
            "all_two_pe_match_reference": all(item["two_pe"]["matches_reference"] for item in per_case),
            "all_two_pe_match_one_pe_fetch": all(item["two_pe"]["matches_one_pe_fetch"] for item in per_case),
            "case2_multi_k": {
                "one_pe_cycle_count_sequential": seq1,
                "two_pe_cycle_count_sequential": seq2,
                "two_pe_cycle_count_parallel_estimate": par2,
                "parallel_cycle_reduction_pct": float((1.0 - (par2 / max(seq1, 1.0))) * 100.0),
                "sequential_cycle_overhead_pct": float(((seq2 / max(seq1, 1.0)) - 1.0) * 100.0),
            },
        },
        "per_case": per_case,
    }


def main() -> int:
    report = build_report()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
