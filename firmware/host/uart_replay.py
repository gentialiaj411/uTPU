from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from isa_encoder import IsaConfig
from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from requantization import RequantParams


MAGIC_UPLOAD = 0xA1
MAGIC_START = 0xA2
MAGIC_REARM = 0xA3
MAGIC_READ_PERF = 0xA4

DEMO_CFG = IsaConfig(address_width=10, compute_data_width=8)
DEMO_ARRAY_SIZE = 8
DEMO_BUFFER_SIZE = 1 << DEMO_CFG.address_width
DEMO_PROG_DEPTH = 256
DEMO_WEIGHT_ADDR = 0
DEMO_INPUT_ADDR = 64
DEMO_RESULT_ADDR = 128
DEMO_UART_BAUD = 6_250_000
DEMO_ACCUMULATOR_WIDTH = 32


@dataclass(frozen=True)
class UARTReplayDemo:
    name: str
    program: bytes
    upload_bytes: bytes
    start_bytes: bytes
    expected_uart_bytes: bytes
    decoded_outputs: np.ndarray
    expected_outputs: np.ndarray
    weights: np.ndarray
    activations: np.ndarray
    requant_params: RequantParams
    program_words: int
    prog_depth: int
    cfg: IsaConfig
    array_size: int
    buffer_size: int
    accumulator_data_width: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "demo_name": self.name,
            "program_words": int(self.program_words),
            "prog_depth": int(self.prog_depth),
            "fits_prog_depth": bool(self.program_words <= self.prog_depth),
            "cfg": {
                "address_width": int(self.cfg.address_width),
                "compute_data_width": int(self.cfg.compute_data_width),
                "buffer_size": int(self.buffer_size),
                "extended_address": bool(self.cfg.extended_address),
            },
            "array_size": int(self.array_size),
            "accumulator_data_width": int(self.accumulator_data_width),
            "upload_bytes": list(self.upload_bytes),
            "start_bytes": list(self.start_bytes),
            "expected_uart_bytes": list(self.expected_uart_bytes),
            "expected_outputs": np.asarray(self.expected_outputs, dtype=np.int16).tolist(),
            "weights": np.asarray(self.weights, dtype=np.int16).tolist(),
            "activations": np.asarray(self.activations, dtype=np.int16).tolist(),
            "requant_params": self.requant_params.as_dict(),
        }


def serialize_program_upload(program: bytes) -> bytes:
    if len(program) % 2 != 0:
        raise ValueError("program must contain whole 16-bit words")
    words = len(program) // 2
    if words >= (1 << 16):
        raise ValueError(f"program too large for UART upload header: {words} words")
    return bytes([MAGIC_REARM, MAGIC_UPLOAD, words & 0xFF, (words >> 8) & 0xFF]) + program


def parse_uart_captured_bytes(
    captured: Sequence[int] | bytes,
    *,
    out_features: int,
    batch_size: int,
    array_size: int,
    cfg: IsaConfig,
) -> np.ndarray:
    raw = list(bytes(captured))
    if len(raw) % 2 != 0:
        raise ValueError(f"captured UART byte count must be even, got {len(raw)}")
    words = []
    for idx in range(0, len(raw), 2):
        words.append((int(raw[idx]) & 0xFF) | ((int(raw[idx + 1]) & 0xFF) << 8))
    values = []
    mask = (1 << cfg.compute_data_width) - 1
    sign_bit = 1 << (cfg.compute_data_width - 1)
    for word in words:
        for shift in range(0, 16, cfg.compute_data_width):
            lane = (word >> shift) & mask
            values.append(lane - (1 << cfg.compute_data_width) if lane & sign_bit else lane)
    out_blocks = (int(out_features) + int(array_size) - 1) // int(array_size)
    out_padded = out_blocks * int(array_size)
    result = np.zeros((int(batch_size), out_padded), dtype=np.int8)
    cursor = 0
    for ob in range(out_blocks):
        take = int(array_size) * int(batch_size)
        tile = np.asarray(values[cursor : cursor + take], dtype=np.int8).reshape(
            (int(array_size), int(batch_size)),
            order="F",
        )
        result[:, ob * int(array_size) : (ob + 1) * int(array_size)] = tile.T
        cursor += take
    return result[:, : int(out_features)]


def _demo_weights() -> np.ndarray:
    values = np.array(
        [
            [7, -3, 5, 1, -2, 4, -6, 2],
            [-4, 6, -1, 3, 5, -7, 2, 0],
            [2, 1, -5, 7, -3, 6, -4, 5],
            [3, -2, 4, -6, 1, 7, -5, 2],
            [-7, 5, 2, -1, 6, -3, 4, -2],
            [1, 4, -6, 2, -5, 3, 7, -1],
            [5, -7, 3, 6, -4, 2, -1, 4],
            [-2, 3, 6, -5, 4, -1, 5, 7],
        ],
        dtype=np.int8,
    )
    return values


def _demo_activations() -> np.ndarray:
    return np.array([6, -5, 4, -3, 2, -1, 7, -6], dtype=np.int8)


def build_uart_replay_demo() -> UARTReplayDemo:
    weights = _demo_weights()
    activations = _demo_activations()
    requant = RequantParams(multiplier=3, right_shift=1, enable=True)
    lowered = lower_blocked_fc_program_utpu(
        weights,
        activations,
        out_features=8,
        in_features=8,
        array_size=DEMO_ARRAY_SIZE,
        apply_relu=False,
        apply_quant=True,
        weight_addr=DEMO_WEIGHT_ADDR,
        input_addr=DEMO_INPUT_ADDR,
        result_addr=DEMO_RESULT_ADDR,
        prog_depth=DEMO_PROG_DEPTH,
        cfg=DEMO_CFG,
        requant_params=requant,
    )
    program = lowered["program"]
    program_words = int(lowered["program_instruction_words"])
    sim = simulate_program_bytes(
        program,
        array_size=DEMO_ARRAY_SIZE,
        buffer_size=DEMO_BUFFER_SIZE,
        cfg=DEMO_CFG,
        accumulator_data_width=DEMO_ACCUMULATOR_WIDTH,
    )
    expected_uart = bytes(int(v) & 0xFF for v in sim.fetch_bytes)
    parsed = parse_uart_captured_bytes(
        expected_uart,
        out_features=8,
        batch_size=1,
        array_size=DEMO_ARRAY_SIZE,
        cfg=DEMO_CFG,
    )
    if program_words > DEMO_PROG_DEPTH:
        raise ValueError(
            f"demo program does not fit PROG_DEPTH={DEMO_PROG_DEPTH}: {program_words} words"
        )
    return UARTReplayDemo(
        name="int8_fc8x8_single_tile_uart_replay",
        program=program,
        upload_bytes=serialize_program_upload(program),
        start_bytes=bytes([MAGIC_START]),
        expected_uart_bytes=expected_uart,
        decoded_outputs=parsed,
        expected_outputs=parsed.copy(),
        weights=weights,
        activations=activations,
        requant_params=requant,
        program_words=program_words,
        prog_depth=DEMO_PROG_DEPTH,
        cfg=DEMO_CFG,
        array_size=DEMO_ARRAY_SIZE,
        buffer_size=DEMO_BUFFER_SIZE,
        accumulator_data_width=DEMO_ACCUMULATOR_WIDTH,
    )


def write_uart_replay_vectors(
    demo: UARTReplayDemo,
    *,
    repo_root: Path,
    stem: str = "uart_preboard_demo",
) -> Dict[str, str]:
    tv_dir = repo_root / "build" / "test_vectors"
    tv_dir.mkdir(parents=True, exist_ok=True)
    upload_mem = tv_dir / f"{stem}_upload.mem"
    expected_mem = tv_dir / f"{stem}_expected_uart.mem"
    meta_json = tv_dir / f"{stem}.json"
    svh = tv_dir / "uart_replay_expected.svh"

    upload_mem.write_text(
        "\n".join(f"{int(b) & 0xFF:02x}" for b in demo.upload_bytes) + "\n",
        encoding="utf-8",
    )
    expected_mem.write_text(
        "\n".join(f"{int(b) & 0xFF:02x}" for b in demo.expected_uart_bytes) + "\n",
        encoding="utf-8",
    )
    meta_json.write_text(json.dumps(demo.to_dict(), indent=2), encoding="utf-8")
    svh.write_text(
        "\n".join(
            [
                "// Auto-generated by uart_replay.py",
                f'`define UART_REPLAY_UPLOAD_MEM "{upload_mem.as_posix()}"',
                f'`define UART_REPLAY_EXPECTED_MEM "{expected_mem.as_posix()}"',
                f"`define UART_REPLAY_UPLOAD_N {len(demo.upload_bytes)}",
                f"`define UART_REPLAY_EXPECTED_N {len(demo.expected_uart_bytes)}",
                f"`define UART_REPLAY_WORDS {demo.program_words}",
                f"`define UART_REPLAY_PROG_DEPTH {demo.prog_depth}",
                f"`define UART_REPLAY_ARRAY_SIZE {demo.array_size}",
                f"`define UART_REPLAY_BUFFER_SIZE {demo.buffer_size}",
                f"`define UART_REPLAY_EXT_ADDR_EN {1 if demo.cfg.extended_address else 0}",
                f"`define UART_REPLAY_COMPUTE_DATA_WIDTH {demo.cfg.compute_data_width}",
                f"`define UART_REPLAY_ACCUMULATOR_DATA_WIDTH {demo.accumulator_data_width}",
                f"`define UART_REPLAY_UART_BAUD {DEMO_UART_BAUD}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "upload_mem": str(upload_mem),
        "expected_mem": str(expected_mem),
        "meta_json": str(meta_json),
        "svh": str(svh),
    }
