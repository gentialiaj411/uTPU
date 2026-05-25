import json
from pathlib import Path

import numpy as np

from generate_fused_rtl_test_vectors import _fc_block_reference, generate_vectors
from isa_simulator import simulate_program_bytes
from lowering_fused_mlp_utpu import lower_fused_mlp_program_utpu

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "bench" / "results" / "multi_pe_sim.json"


def _case_tensors(case_name: str):
    if case_name == "case1_single_k":
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
        return fc1_w, fc2_w, x

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


def test_one_pe_and_two_pe_fused_mlp_match_reference():
    vectors = generate_vectors()
    for case in vectors["cases"]:
        fc1_w, fc2_w, x = _case_tensors(case["name"])
        ref = _fc_block_reference(fc1_w, fc2_w, x, fc1_relu=True, fc2_relu=False)
        one_pe = lower_fused_mlp_program_utpu(fc1_w, fc2_w, x, num_pe=1)
        two_pe = lower_fused_mlp_program_utpu(fc1_w, fc2_w, x, num_pe=2)
        sim1 = simulate_program_bytes(one_pe["program"], num_pe=1)
        sim2 = simulate_program_bytes(two_pe["program"], num_pe=2)
        assert sim1.fetch_bytes == ref["fc2_fetch_bytes"]
        assert sim2.fetch_bytes == ref["fc2_fetch_bytes"]
        assert sim1.fetch_bytes == sim2.fetch_bytes


def test_two_pe_schedule_emitted_for_multi_k_case():
    fc1_w, fc2_w, x = _case_tensors("case2_multi_k")
    two_pe = lower_fused_mlp_program_utpu(fc1_w, fc2_w, x, num_pe=2)
    assert two_pe["multi_pe_schedule_emitted"] is True
    assert two_pe["mode"] == "compressed_fused_2pe"
    assert two_pe["num_pe"] == 2


def test_multi_pe_sim_artifact_schema():
    assert ARTIFACT_PATH.exists(), f"Missing artifact: {ARTIFACT_PATH}"
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    required = {"version", "generated_at_utc", "methodology", "summary", "per_case"}
    assert required.issubset(report.keys())
    assert report["summary"]["all_two_pe_match_reference"] is True
    assert report["summary"]["all_two_pe_match_one_pe_fetch"] is True
    case2 = report["summary"]["case2_multi_k"]
    assert case2["one_pe_cycle_count_sequential"] > 0
    assert case2["two_pe_cycle_count_parallel_estimate"] > 0
