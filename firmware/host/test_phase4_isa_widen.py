"""Phase 4 Stage A widening tests.

These tests exercise the new ``IsaConfig`` parameter axes in the encoder
and simulator (Phase 4 host-side scaffolding):

* ``address_width=14`` (extended-address: 2-word LOAD/RUN/FETCH/BSTORE)
* ``compute_data_width=8`` (INT8 datapath: 2 elements per 16-bit buffer
  word instead of 4)

Scope: simulator-only. The widened RTL FSM that consumes the
extended-address format on hardware is Phase 4 Stage B (deferred). These
tests prove the host encoder/simulator are parameter-driven and produce
NumPy-matching results at the widened configuration.

Legacy (default) behaviour is exercised by ``test_isa_simulator.py``,
``test_tiling_controller.py``, ``test_multi_pe_sim.py``, etc. — those
must remain byte-identical to pre-Phase-4 (verified separately).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from isa_encoder import (
    DEFAULT_CFG,
    ISAEncoder,
    IsaConfig,
    encodeBurstStore,
    encodeFetch,
    encodeHalt,
    encodeLoadInputs,
    encodeLoadWeights,
    encodeRun,
    encodeStoreValues,
    pack_values_to_word,
)
from isa_simulator import (
    UTPUISASimulator,
    _pack_compute_word,
    _unpack_compute_word,
    simulate_program_bytes,
)


# ---------------------------------------------------------------------------
# Pure config / packing math
# ---------------------------------------------------------------------------


def test_isa_config_defaults_match_legacy():
    cfg = IsaConfig()
    assert cfg.instruction_width == 16
    assert cfg.address_width == 9
    assert cfg.compute_data_width == 4
    assert cfg.items_per_word == 4
    assert cfg.address_max == 511
    assert cfg.extended_address is False


def test_isa_config_int8_extended():
    cfg = IsaConfig(address_width=14, compute_data_width=8)
    assert cfg.items_per_word == 2
    assert cfg.address_max == (1 << 14) - 1
    assert cfg.extended_address is True


def test_isa_config_rejects_invalid_compute_width():
    with pytest.raises(ValueError):
        IsaConfig(compute_data_width=3)  # not a divisor of 16
    with pytest.raises(ValueError):
        IsaConfig(compute_data_width=0)


def test_isa_config_rejects_invalid_address_width():
    with pytest.raises(ValueError):
        IsaConfig(address_width=8)  # below legacy lower bound
    with pytest.raises(ValueError):
        IsaConfig(address_width=17)


def test_pack_values_to_word_int4_matches_legacy():
    """``pack_values_to_word`` with the default config must reproduce
    ``int4To16`` for the same inputs."""
    from isa_encoder import int4To16

    samples = [
        [0, 0, 0, 0],
        [1, 2, 3, 4],
        [-1, -2, -3, -4],
        [7, -8, 7, -8],
    ]
    for s in samples:
        assert pack_values_to_word(s) == int4To16(list(s))


def test_pack_values_to_word_int8_round_trip():
    """INT8 packing: 2 signed INT8 values per 16-bit word."""
    cfg = IsaConfig(compute_data_width=8)
    samples = [
        [0, 0],
        [1, 2],
        [127, -128],
        [-50, 60],
    ]
    for s in samples:
        word = pack_values_to_word(s, cfg=cfg)
        # Round-trip through the simulator's unpacker for symmetry.
        unpacked = _unpack_compute_word(word, cfg)
        assert unpacked[: len(s)] == s


def test_simulator_int8_pack_unpack_round_trip():
    cfg = IsaConfig(compute_data_width=8)
    for v0 in (-128, -50, 0, 50, 127):
        for v1 in (-128, -1, 0, 1, 127):
            word = _pack_compute_word([v0, v1], cfg)
            assert _unpack_compute_word(word, cfg) == [v0, v1]


# ---------------------------------------------------------------------------
# Legacy byte-equivalence: the new code path with cfg=None must be
# byte-identical to the pre-Phase-4 encoding for every known opcode.
# ---------------------------------------------------------------------------


def test_legacy_encode_bytes_unchanged_loadweights():
    # 9-bit address packed into bits 7..15: addr=0x080 -> 0x4000 | 0xB = 0x400B
    assert encodeLoadWeights(0x080) == bytes([0x0B, 0x40])


def test_legacy_encode_bytes_unchanged_run():
    # default flags (compute=quantize=relu=True): 0x3A; addr 0x100 -> 0x803A
    assert encodeRun(0x100) == bytes([0x3A, 0x80])


def test_legacy_encode_bytes_unchanged_fetch_top():
    # FETCH 0x100 top -> opcode 0x1, top bit3 -> 0x9; addr<<7 = 0x8000
    assert encodeFetch(0x100, top_half=True) == bytes([0x09, 0x80])


def test_legacy_encode_bytes_unchanged_store_immediate():
    out = encodeStoreValues(0x080, [1, 2, 3, 4])
    # word1 = 0x10 (opcode 0 + bit4=imm); word2 = 0x4321 (1|2<<4|3<<8|4<<12)
    # word3 = 0x80
    assert out == bytes([0x10, 0x00, 0x21, 0x43, 0x80, 0x00])


# ---------------------------------------------------------------------------
# Extended-address single-program smoke (LOAD/RUN/FETCH 2-word)
# ---------------------------------------------------------------------------


def test_extended_address_loadweights_emits_two_words():
    cfg = IsaConfig(address_width=14)
    out = encodeLoadWeights(0x1234, cfg=cfg)
    assert len(out) == 4
    # word1: opcode=0x3, is_weights bit3 -> 0x000B (no addr in header)
    assert out[:2] == bytes([0x0B, 0x00])
    # word2: full 14-bit address in low bits
    assert out[2:] == bytes([0x34, 0x12])


def test_extended_address_run_emits_two_words():
    cfg = IsaConfig(address_width=14)
    out = encodeRun(0x2000, cfg=cfg)
    # word1: 0x3A (compute|quantize|relu flags), no address
    assert out[:2] == bytes([0x3A, 0x00])
    assert out[2:] == bytes([0x00, 0x20])


def test_extended_address_run_with_residual_emits_three_words():
    cfg = IsaConfig(address_width=14)
    out = encodeRun(0x2000, residual_en=True, residual_addr=0x1234, cfg=cfg)
    # word1: residual bit set alongside the default RUN flags.
    assert out[:2] == bytes([0xBA, 0x00])
    assert out[2:4] == bytes([0x00, 0x20])
    assert out[4:6] == bytes([0x34, 0x12])


def test_extended_address_fetch_emits_two_words():
    cfg = IsaConfig(address_width=14)
    out = encodeFetch(0x1000, top_half=False, cfg=cfg)
    assert out[:2] == bytes([0x01, 0x00])
    assert out[2:] == bytes([0x00, 0x10])


def test_extended_address_burst_store_layout():
    cfg = IsaConfig(address_width=14)
    out = encodeBurstStore(0x0F00, [0xCAFE, 0xBABE], cfg=cfg)
    # header = 0x0006 (opcode only); addr = 0x0F00; count = 2; payload follows.
    assert out[:2] == bytes([0x06, 0x00])
    assert out[2:4] == bytes([0x00, 0x0F])
    assert out[4:6] == bytes([0x02, 0x00])
    assert out[6:8] == bytes([0xFE, 0xCA])
    assert out[8:10] == bytes([0xBE, 0xBA])


# ---------------------------------------------------------------------------
# End-to-end: simulator decodes extended-address bytes and produces the
# correct INT4 buffer reads/writes.
# ---------------------------------------------------------------------------


def test_simulator_extended_address_store_and_fetch_round_trip():
    """A program that targets an INT4 buffer index >= 512 (legacy
    9-bit address can't reach it) must round-trip through the
    extended-address simulator path."""
    cfg = IsaConfig(address_width=14, compute_data_width=4)
    enc = ISAEncoder(cfg)
    enc.store(1024, [1, -2, 3, -4])  # legal in extended mode (address 1024)
    enc.fetch(1024, top_half=False)
    enc.fetch(1024, top_half=True)
    enc.halt()
    program = enc.getProgram()

    sim = UTPUISASimulator(
        array_size=16, buffer_size=4096, cfg=cfg, accumulator_data_width=16
    )
    result = sim.run_words(
        [int.from_bytes(program[i : i + 2], "little") for i in range(0, len(program), 2)]
    )
    assert result.halted
    # Same packing as the legacy ``test_isa_simulator_store_fetch_bytes``:
    # word = 0xC3E1 -> low byte 0xE1, high byte 0xC3.
    assert result.fetch_bytes == [0xE1, 0xC3]


def test_simulator_legacy_address_path_byte_identical():
    """Sanity: a default-cfg simulator is byte-identical to legacy. This
    duplicates the existing ``test_isa_simulator_store_fetch_bytes``
    against the (refactored) simulator to guard against regressions in
    the legacy decode path during further Phase 4 work."""
    enc = ISAEncoder()  # legacy
    enc.store(0, [1, -2, 3, -4])
    enc.fetch(0, top_half=False)
    enc.fetch(0, top_half=True)
    enc.halt()
    program = enc.getProgram()
    res = simulate_program_bytes(program)
    assert res.halted
    assert res.fetch_bytes == [0xE1, 0xC3]


# ---------------------------------------------------------------------------
# INT8 datapath: tiny matmul end-to-end through the simulator with
# array_size=8, buffer_size large enough to hold W + x + result.
# ---------------------------------------------------------------------------


def _int8_matmul_oracle_quantize(W: np.ndarray, x: np.ndarray, *, compute_width: int = 8) -> np.ndarray:
    """NumPy reference matching ``UTPUISASimulator._run_finalize`` semantics
    at the INT8 widened config (clip to int8 range, no leaky relu)."""
    accum = (W.astype(np.int32) @ x.astype(np.int32))
    hi = (1 << (compute_width - 1)) - 1
    lo = -(1 << (compute_width - 1))
    return np.clip(accum, lo, hi).astype(np.int32)


def _build_int8_matmul_program(
    W: np.ndarray, x: np.ndarray, *, cfg: IsaConfig, array_size: int,
    weight_addr: int, input_addr: int, result_addr: int,
) -> bytes:
    """Tiny single-block Linear program at the widened INT8 config.

    Layout matches ``lower_blocked_fc_program_utpu`` for one (out_block,
    in_block) pair, but uses the parameterised ``IsaConfig`` so addresses
    and packing widen automatically.
    """
    enc = ISAEncoder(cfg)
    items = cfg.items_per_word

    # Store weights row-major, ``items`` per buffer word.
    flat_w = W.flatten().tolist()
    addr = weight_addr
    for i in range(0, len(flat_w), items):
        chunk = flat_w[i : i + items]
        enc.store(addr, chunk)
        addr += 1

    enc.loadWeights(weight_addr)

    # Store inputs.
    flat_x = x.flatten().tolist()
    addr = input_addr
    for i in range(0, len(flat_x), items):
        chunk = flat_x[i : i + items]
        enc.store(addr, chunk)
        addr += 1

    enc.loadInputs(input_addr)
    # Fused compute+finalize in a single RUN (compute=1 quantize=1 relu=0).
    enc.run(result_addr, compute=True, quantize=True, relu=False, acc_clear=True)

    # Fetch every byte of the array_size outputs at this widened config.
    # Finalize writes ``array_size * array_size`` packed elements
    # (==``array_size``-many lanes followed by zero padding); to recover
    # ``array_size`` outputs we read ``array_size // items`` words.
    for widx in range(array_size // items):
        a = result_addr + widx
        enc.fetch(a, top_half=False)
        enc.fetch(a, top_half=True)
    enc.halt()
    return enc.getProgram()


def test_int8_array_size_8_single_block_matmul_matches_oracle():
    rng = np.random.default_rng(0xBEEF)
    array_size = 8
    cfg = IsaConfig(address_width=14, compute_data_width=8)
    # 1 output block (8 outputs) x 1 input block (8 inputs); INT8 inputs in [-50, 50)
    # so the int8 accumulator never exceeds the INT8 finalize clip range too noisily.
    W = rng.integers(-50, 50, size=(array_size, array_size), dtype=np.int8)
    x = rng.integers(-50, 50, size=(array_size,), dtype=np.int8)

    weight_addr = 0
    weight_words = (array_size * array_size) // cfg.items_per_word
    input_addr = weight_addr + weight_words
    input_words = array_size // cfg.items_per_word
    result_addr = input_addr + input_words

    program = _build_int8_matmul_program(
        W, x, cfg=cfg, array_size=array_size,
        weight_addr=weight_addr, input_addr=input_addr, result_addr=result_addr,
    )
    sim_res = simulate_program_bytes(
        program,
        array_size=array_size,
        buffer_size=4096,
        cfg=cfg,
        accumulator_data_width=32,
    )
    assert sim_res.halted

    # Decode fetch bytes into INT8 outputs.
    fetched = sim_res.fetch_bytes
    expected = _int8_matmul_oracle_quantize(W, x)
    decoded: List[int] = []
    for byte in fetched:
        decoded.append(_sign_extend_signed_byte(byte))
    assert decoded[:array_size] == expected.tolist()


def _sign_extend_signed_byte(byte: int) -> int:
    return byte - 256 if byte >= 128 else byte


def test_int8_array_size_8_with_relu_matches_oracle():
    """Same shape but ``relu=True`` (leaky-relu via right-shift) — checks
    the simulator's finalize-with-relu path at INT8."""
    rng = np.random.default_rng(0xC0DE)
    array_size = 8
    cfg = IsaConfig(address_width=14, compute_data_width=8)
    W = rng.integers(-50, 50, size=(array_size, array_size), dtype=np.int8)
    x = rng.integers(-50, 50, size=(array_size,), dtype=np.int8)

    weight_addr = 0
    input_addr = weight_addr + (array_size * array_size) // cfg.items_per_word
    result_addr = input_addr + array_size // cfg.items_per_word

    enc = ISAEncoder(cfg)
    items = cfg.items_per_word
    flat_w = W.flatten().tolist()
    addr = weight_addr
    for i in range(0, len(flat_w), items):
        enc.store(addr, flat_w[i : i + items])
        addr += 1
    enc.loadWeights(weight_addr)
    addr = input_addr
    flat_x = x.flatten().tolist()
    for i in range(0, len(flat_x), items):
        enc.store(addr, flat_x[i : i + items])
        addr += 1
    enc.loadInputs(input_addr)
    enc.run(result_addr, compute=True, quantize=True, relu=True, acc_clear=True)
    for widx in range(array_size // items):
        a = result_addr + widx
        enc.fetch(a, top_half=False)
        enc.fetch(a, top_half=True)
    enc.halt()

    sim_res = simulate_program_bytes(
        enc.getProgram(),
        array_size=array_size,
        buffer_size=4096,
        cfg=cfg,
        accumulator_data_width=32,
    )
    assert sim_res.halted

    decoded = [_sign_extend_signed_byte(b) for b in sim_res.fetch_bytes][:array_size]

    # Oracle: clip to INT8 then leaky-relu (>>2 on negatives).
    accum = (W.astype(np.int32) @ x.astype(np.int32))
    clipped = np.clip(accum, -128, 127).astype(np.int32)
    expected = np.where(clipped < 0, clipped >> 2, clipped).astype(np.int32)
    assert decoded == expected.tolist()


# ---------------------------------------------------------------------------
# Extended-address BSTORE (burst write across the 9-bit boundary).
# ---------------------------------------------------------------------------


def test_extended_address_bstore_writes_high_addresses():
    cfg = IsaConfig(address_width=14)
    enc = ISAEncoder(cfg)
    payload = list(range(1, 9))
    enc.burst_store(2048, payload)  # base address > 512 (illegal in legacy)
    enc.halt()
    program = enc.getProgram()
    sim = UTPUISASimulator(array_size=16, buffer_size=4096, cfg=cfg)
    res = sim.run_words(
        [int.from_bytes(program[i : i + 2], "little") for i in range(0, len(program), 2)]
    )
    assert res.halted
    assert res.executed_ops["bstore"] == 1
    # Every payload word must be written at base+offset.
    for offset, value in enumerate(payload):
        assert sim.pes[0].buffer[2048 + offset] == value & 0xFFFF
