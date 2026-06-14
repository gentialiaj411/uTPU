import json
from pathlib import Path

import pytest

from run_uart_preboard_demo import OUTPUT_JSON, build_artifact
from uart_replay import build_uart_replay_demo, parse_uart_captured_bytes, serialize_program_upload


def test_uart_serialize_prefix_and_length():
    demo = build_uart_replay_demo()
    payload = serialize_program_upload(demo.program)
    assert payload[:2] == bytes([0xA3, 0xA1])
    assert int.from_bytes(payload[2:4], byteorder="little", signed=False) == demo.program_words
    assert payload[4:] == demo.program


def test_uart_parse_matches_demo_expected_outputs():
    demo = build_uart_replay_demo()
    parsed = parse_uart_captured_bytes(
        demo.expected_uart_bytes,
        out_features=8,
        batch_size=1,
        array_size=demo.array_size,
        cfg=demo.cfg,
    )
    assert parsed.tolist() == demo.expected_outputs.tolist()


def test_uart_replay_artifact_roundtrip_and_status():
    artifact = build_artifact()
    assert artifact["on_silicon"]["status"] == "simulation"
    assert artifact["demo_program"]["fits_prog_depth"] is True
    assert artifact["uart_roundtrip"]["captured_uart_matches_isa_sim"] is True
    assert artifact["rtl_simulation"]["no_x_quantizer_finalize"] is True
    assert artifact["rtl_simulation"]["no_x_uart_tx_path"] is True
    saved = json.loads(Path(OUTPUT_JSON).read_text(encoding="utf-8"))
    assert saved["on_silicon"]["status"] == "simulation"
    assert saved["uart_roundtrip"]["captured_uart_matches_isa_sim"] is True
