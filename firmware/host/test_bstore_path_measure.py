"""Schema lock for BSTORE path measurement artifact."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
ART = REPO / "bench" / "results" / "bstore_path_measure.json"


@pytest.fixture(scope="module")
def report() -> dict:
    if not ART.exists():
        subprocess.check_call(
            [sys.executable, str(HOST / "run_bstore_path_measure.py")],
            cwd=REPO,
        )
    return json.loads(ART.read_text(encoding="utf-8"))


def test_bstore_identity_exact(report: dict) -> None:
    m = report["measured"]
    assert m["identity_check"]["matches_attr_bstore_cycles"] is True
    assert m["bstore_payload_words"] == 1296
    assert m["bstore_bursts"] == 13
    assert m["bstore_cycles"] == 5197
    assert abs(m["cycles_per_payload_word"] - 4.010030864197531) < 1e-9


def test_buffer_banks_documented(report: dict) -> None:
    geo = report["unified_buffer_write_width"]["sim_attr_path_n16_int4"]
    assert geo["banks"] == 64
    assert geo["compute_port_write_words_per_cycle"] == 64
    assert geo["store_port_write_words_per_cycle"] == 1
    assert geo["store_port_write_words_per_cycle_shipping"] == 8


def test_part_a_multilayer_ceiling_in_sketches(report: dict) -> None:
    a = report["amdahl_sketches_same_workload"]["part_a_if_bstore_and_compute_survive"]
    assert abs(a["ceiling_x"] - 1.1044700304774806) < 1e-9


def test_implementation_landed_width_8(report: dict) -> None:
    impl = report["implementation"]
    assert impl["BSTORE_WIDTH_shipping"] == 8
    assert impl["status"] == "landed"
    smoke = impl["post_widen_smoke"]
    assert smoke is not None
    assert smoke["status"] == "pass"
    assert smoke["cycles_per_payload_word"] is not None
    assert smoke["cycles_per_payload_word"] < 2.0
    post = impl.get("post_widen_mnist_attr")
    if post and post.get("bstore_cycles") is not None:
        assert post["bstore_cycles"] < 5197
        assert post["e2e_speedup_vs_pre_widen"] > 1.5
