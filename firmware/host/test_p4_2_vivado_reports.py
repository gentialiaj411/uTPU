from __future__ import annotations

import json
from pathlib import Path


ARTIFACT = Path("bench/results/p4_2_vivado_reports.json")


def load_artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_schema_and_metadata() -> None:
    data = load_artifact()
    assert data["version"] == 1
    assert data["generated_at_utc"].endswith("Z")
    assert data["git_sha"] == "6d5dc2532f1f31200a2b4f687e3f2c0503cff67c"
    assert len(data["runs"]) == 2


def test_run_a_measured_numbers() -> None:
    run = load_artifact()["runs"][0]
    assert run["name"] == "pynqz2_bram_max"
    assert run["params"] == {
        "PROG_DEPTH": 8192,
        "ARRAY_SIZE": 16,
        "BUFFER_SIZE": 512,
        "BUFFER_WORD_SIZE": 16,
        "COMPUTE_DATA_WIDTH": 4,
        "EXT_ADDR_EN": 0,
    }
    assert run["part"] == "xc7a100tcsg324-1"
    assert run["timing"] == {
        "wns_ns": 0.101,
        "whs_ns": 0.019,
        "tns_ns": 0.0,
        "ths_ns": 0.0,
        "all_paths_met": True,
    }
    assert run["utilization"] == {
        "lut_used": 6721,
        "lut_available": 63400,
        "ff_used": 8960,
        "ff_available": 126800,
        "bram_36k_used": 37,
        "bram_36k_available": 135,
        "dsp_used": 240,
        "dsp_available": 240,
    }
    assert run["route_status"] == "clean"


def test_run_b_measured_numbers() -> None:
    run = load_artifact()["runs"][1]
    assert run["name"] == "widened_int8"
    assert run["params"] == {
        "PROG_DEPTH": 8192,
        "COMPUTE_DATA_WIDTH": 8,
        "ACCUMULATOR_DATA_WIDTH": 32,
        "ARRAY_SIZE": 8,
        "BUFFER_SIZE": 4096,
        "EXT_ADDR_EN": 1,
    }
    assert run["timing"] == {
        "wns_ns": 1.163,
        "whs_ns": 0.017,
        "tns_ns": 0.0,
        "ths_ns": 0.0,
        "all_paths_met": True,
    }
    assert run["utilization"] == {
        "lut_used": 2883,
        "lut_available": 63400,
        "ff_used": 4764,
        "ff_available": 126800,
        "bram_36k_used": 21,
        "bram_36k_available": 135,
        "dsp_used": 64,
        "dsp_available": 240,
    }
    assert run["route_status"] == "clean"


def test_reality_floors() -> None:
    data = load_artifact()
    for run in data["runs"]:
        assert run["timing"]["wns_ns"] >= 0.0
        assert run["timing"]["whs_ns"] >= 0.0
        assert run["timing"]["tns_ns"] == 0.0
        assert run["timing"]["ths_ns"] == 0.0
        assert run["timing"]["all_paths_met"] is True
        assert run["utilization"]["bram_36k_used"] <= run["utilization"]["bram_36k_available"]
        assert run["utilization"]["dsp_used"] <= run["utilization"]["dsp_available"]
        assert "ERROR" not in run["route_status"].upper()
        assert "UNROUTED" not in run["route_status"].upper()
