"""Fail-closed mutation tests for the Phase-3 gate artifact validator."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from phase3_gate import validate_artifacts


def artifacts() -> tuple[dict, dict]:
    evidence = json.loads((ROOT / "bench/results/tiny_transformer_vec_rtl.json").read_text(encoding="utf-8"))
    synth = {"status": "ok", "part": "xc7s50csga324-1", "modes": {
        "0": {"lut": 100, "ff": 100, "bram": 1, "dsp": 1, "wns_ns": 1.0},
        "1": {"lut": 200, "ff": 200, "bram": 2, "dsp": 2, "wns_ns": 0.5}},
        "delta": {"lut": 100}, "vector_unit_ooc": {"wns_ns": 1.5}}
    return evidence, synth


def test_mismatch_one_fails() -> None:
    evidence, synth = artifacts()
    evidence["per_op"]["vadd_rq_exhaustive"]["mismatches"] = 1
    assert "VADD_RQ mismatches" in validate_artifacts(evidence, synth)


def test_deleted_decode_class_fails() -> None:
    evidence, synth = artifacts()
    del evidence["decode_errors"]["classes"]["pointer_extent"]
    assert any("decode error classes missing" in x for x in validate_artifacts(evidence, synth))


def test_missing_transition_fails() -> None:
    evidence, synth = artifacts()
    del evidence["transition_matrix"]["matrix"]["STORE:VEC->X"]
    assert any("transition matrix incomplete" in x for x in validate_artifacts(evidence, synth))


def test_negative_synthesis_wns_fails() -> None:
    evidence, synth = artifacts()
    synth["modes"]["1"]["wns_ns"] = -0.001
    assert "synthesis WNS < 0" in validate_artifacts(evidence, synth)


def test_wrong_synthesis_part_fails() -> None:
    evidence, synth = artifacts()
    synth["part"] = "xc7a100tcsg324-1"
    assert "synthesis wrong part" in validate_artifacts(evidence, synth)
