"""Replay staged onboard UART transcript against ISA-sim expected bytes.

Passes without a board when ``docs/evidence/fpga_onboard/uart_transcript.json``
exists. Does **not** promote ``on_silicon`` — that requires
``bit_exact_vs_isa_sim=true`` in both the transcript and
``bench/results/onboard_execution.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRANSCRIPT = REPO / "docs" / "evidence" / "fpga_onboard" / "uart_transcript.json"
EXPECTED = REPO / "docs" / "evidence" / "fpga_onboard" / "onboard_expected_uart.json"
ARTIFACT = REPO / "bench" / "results" / "onboard_execution.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_onboard_evidence_scaffold_exists_after_runner():
    if not ARTIFACT.is_file():
        pytest.skip("onboard_execution.json not generated yet — run run_onboard_execution.py")
    blob = _load(ARTIFACT)
    assert blob.get("schema_version") == 1
    assert "on_silicon" in blob
    assert blob["on_silicon"]["claim_rule"]
    # Never silently claim hardware.
    if blob.get("status") != "hardware_bit_exact":
        assert blob["on_silicon"]["status"] != "hardware"


def test_onboard_transcript_replay_matches_expected_when_capture_present():
    if not TRANSCRIPT.is_file() or not EXPECTED.is_file():
        pytest.skip("onboard transcript/expected not staged")
    transcript = _load(TRANSCRIPT)
    expected_doc = _load(EXPECTED)
    expected = list(expected_doc["expected_uart_bytes"])
    captured = list(transcript.get("captured_uart_bytes") or [])

    # Always check program hash linkage.
    assert transcript.get("program_sha256") == expected_doc.get("program_sha256")

    if not captured:
        # Scaffold / blocked run: replay has nothing to compare; must not claim pass.
        assert transcript.get("bit_exact_vs_isa_sim") is False
        pytest.skip("no captured UART bytes yet (board bring-up not completed)")

    assert len(captured) == len(expected), (
        f"captured length {len(captured)} != expected {len(expected)}"
    )
    assert captured == expected, "captured UART bytes differ from ISA-sim expected"
    assert transcript.get("bit_exact_vs_isa_sim") is True


def test_onboard_artifact_bit_exact_implies_replay_ready():
    if not ARTIFACT.is_file():
        pytest.skip("onboard_execution.json missing")
    blob = _load(ARTIFACT)
    if blob.get("status") != "hardware_bit_exact":
        pytest.skip(f"status={blob.get('status')} — hardware claim not live")
    assert blob["on_silicon"]["bit_exact_vs_isa_sim"] is True
    assert TRANSCRIPT.is_file()
    transcript = _load(TRANSCRIPT)
    assert transcript.get("bit_exact_vs_isa_sim") is True
    assert list(transcript.get("captured_uart_bytes") or []) == list(
        blob["program"]["expected_uart_bytes"]
    )
