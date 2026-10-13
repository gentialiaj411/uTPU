from __future__ import annotations

import pytest

from run_onboard_execution import validate_buffer_resident_artifact


def _artifact():
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "status": "scaffolded_not_executed",
        "claim_rule": "all bytes on every image must match",
        "scope": {"images_requested": 2},
        "deployment": {},
        "float_accuracy": 0.9,
        "quantized_accuracy": 0.88,
        "isa_vs_integer_reference": {"all_images_bit_exact": True},
        "device_cycles_per_inference": {},
        "board_perf_program_cycles_per_inference": {},
        "simulation_seconds": 1.0,
        "board_host_elapsed_seconds": None,
        "board_host_seconds_per_image": None,
        "board_host_latency_scope": "board execution only",
        "bitstream": {"exists": False},
        "flash": {"attempted": False},
        "uart": {"attempted": False, "per_image": []},
    }


def test_buffer_resident_artifact_schema_accepts_simulation_only_result():
    validate_buffer_resident_artifact(_artifact())


def test_buffer_resident_artifact_cannot_claim_bit_exact_without_all_byte_rows():
    artifact = _artifact()
    artifact["status"] = "hardware_bit_exact"
    artifact["uart"] = {"attempted": True, "per_image": [
        {"output_bytes_bit_exact": True},
    ]}
    with pytest.raises(ValueError, match="every requested image"):
        validate_buffer_resident_artifact(artifact)


def test_buffer_resident_artifact_rejects_any_mismatching_output_byte_case():
    artifact = _artifact()
    artifact["status"] = "hardware_bit_exact"
    artifact["uart"] = {"attempted": True, "per_image": [
        {"output_bytes_bit_exact": True},
        {"output_bytes_bit_exact": False},
    ]}
    with pytest.raises(ValueError, match="every output byte"):
        validate_buffer_resident_artifact(artifact)


def test_buffer_resident_artifact_rejects_missing_required_fields():
    artifact = _artifact()
    del artifact["quantized_accuracy"]
    with pytest.raises(ValueError, match="missing schema fields"):
        validate_buffer_resident_artifact(artifact)
