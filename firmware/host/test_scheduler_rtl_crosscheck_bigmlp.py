"""Schema + reality-floor tests for ``scheduler_rtl_crosscheck_bigmlp.json``.

Locks the widened board-fit RTL cross-check that covers the actual
multi-output-block shapes used by the board flash plan.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

REPO_ROOT = os.path.dirname(os.path.dirname(HOST_DIR))
ARTIFACT_PATH = os.path.join(
    REPO_ROOT, "bench", "results", "scheduler_rtl_crosscheck_bigmlp.json"
)

EXPECTED_SHAPES = [
    {"tag": "bench_32x32", "out_features": 32, "in_features": 32},
    {"tag": "bench_32x64", "out_features": 32, "in_features": 64},
    {"tag": "bench_64x32", "out_features": 64, "in_features": 32},
    {"tag": "bench_64x64", "out_features": 64, "in_features": 64},
    {"tag": "bench_128x64", "out_features": 128, "in_features": 64},
]


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT_PATH):
        pytest.skip(
            f"{ARTIFACT_PATH} not found; run "
            "`python firmware/host/run_scheduler_rtl_crosscheck_bigmlp.py` first."
        )
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_top_level_keys_present(artifact):
    for key in [
        "version",
        "generated_at_utc",
        "status",
        "suite",
        "aggregate",
        "cases",
        "methodology",
        "tolerance_permille",
        "host",
        "iverilog",
    ]:
        assert key in artifact, f"missing top-level key: {key}"
    assert artifact["version"] == 1
    assert artifact["suite"] == "board_fit_bigmlp_v1"
    assert artifact["status"] == "ok"


def test_aggregate_locks_success_and_byte_exactness(artifact):
    agg = artifact["aggregate"]
    assert agg["case_count"] == 5
    assert agg["ok_case_count"] == 5
    assert agg["all_cases_ok"] is True
    assert agg["all_cases_rtl_byte_exact"] is True
    assert agg["board_layout"] == {
        "weight_addr": 256,
        "input_addr": 0,
        "result_addr": 320,
        "prog_depth": 8192,
        "array_size": 16,
    }


def test_methodology_documents_board_fit_sweep(artifact):
    meth = artifact["methodology"]
    assert meth["tolerance_permille"] == 20
    assert "board-fit" in meth["summary"]
    assert "multi-out-block" in meth["what_it_measures"]
    assert len(meth["cases"]) == 5
    assert [c["tag"] for c in meth["cases"]] == [s["tag"] for s in EXPECTED_SHAPES]


def test_expected_shapes_are_locked(artifact):
    cases = {c["tag"]: c for c in artifact["cases"]}
    assert set(cases) == {s["tag"] for s in EXPECTED_SHAPES}
    for shape in EXPECTED_SHAPES:
        case = cases[shape["tag"]]
        assert case["shape"] == {
            "out_features": shape["out_features"],
            "in_features": shape["in_features"],
        }
        assert case["array_size"] == 16
        assert case["weight_addr"] == 256
        assert case["input_addr"] == 0
        assert case["result_addr"] == 320


def test_every_case_passes_rtl_reality_floors(artifact):
    for case in artifact["cases"]:
        exp = case["expected"]
        rtl = case["rtl_result"]
        head = case["headline"]

        assert exp["fetch_bytes_invariant_simulator"] is True
        assert exp["naive_words"] <= 8192
        assert exp["sched_words"] <= 8192
        assert exp["cycles_saved"] > 0

        assert rtl["tb_result_pass"] is True
        assert rtl["errors"] == 0
        assert rtl["iverilog_compile_ok"] is True
        assert rtl["iverilog_run_returncode"] == 0
        assert rtl["rtl_naive_bytes_matching_sim"] == rtl["rtl_naive_bytes_total"]
        assert rtl["rtl_sched_bytes_matching_sim"] == rtl["rtl_sched_bytes_total"]

        assert head["scheduler_invariant_holds"] is True
        assert head["rtl_sched_cycles"] < head["rtl_naive_cycles"]
        assert head["diff_permille"] <= head["tol_permille"]


def test_all_cases_remain_close_to_simulator_permille(artifact):
    for case in artifact["cases"]:
        head = case["headline"]
        assert head["diff_permille"] <= artifact["tolerance_permille"]

