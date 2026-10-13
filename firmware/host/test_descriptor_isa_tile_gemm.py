"""Part A descriptor ISA: frozen encoding + independent golden oracle checks.

Three-way plan (RTL sequencer still open):
  1) golden (spec NumPy) vs fully-unrolled microcoded emission result bytes
  2) golden vs ISA-simulator TILE_GEMM expansion
  3) golden vs RTL sequencer — TODO when sequencer lands

The golden model lives in ``tile_gemm_golden.py`` and is written from
``docs/descriptor_isa_part_a_design.md``, not from lowering helpers.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HOST = Path(__file__).resolve().parent
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from isa_encoder import (  # noqa: E402
    OPCODE_TILE_GEMM,
    TILE_GEMM_FLAG_ACC_CLEAR_FIRST,
    TILE_GEMM_FLAG_FINALIZE_LAST,
    TILE_GEMM_FLAG_HOIST_PAYLOADS,
    TILE_GEMM_WORD_COUNT,
    IsaConfig,
    ISAEncoder,
    decodeTileGemmWords,
    encodeTileGemm,
    pack_values_to_word,
)
from isa_simulator import simulate_program_bytes  # noqa: E402
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu  # noqa: E402
from tile_gemm_golden import (  # noqa: E402
    TileGemmDesc,
    golden_tile_gemm,
    hoist_layout_words,
)

DESC_CFG = IsaConfig(address_width=12, compute_data_width=4, descriptor_isa=True)
LEGACY_EXT = IsaConfig(address_width=12, compute_data_width=4, descriptor_isa=False)


def _words(program: bytes) -> list[int]:
    return [struct.unpack_from("<H", program, i)[0] for i in range(0, len(program), 2)]


def _pack_c(mat: np.ndarray, cfg: IsaConfig) -> list[int]:
    flat = list(np.asarray(mat, dtype=np.int8).reshape(-1))
    words = []
    items = cfg.items_per_word
    for i in range(0, len(flat), items):
        chunk = flat[i : i + items]
        while len(chunk) < items:
            chunk.append(0)
        words.append(pack_values_to_word(chunk, cfg))
    return words


def _pack_f(mat: np.ndarray, cfg: IsaConfig) -> list[int]:
    flat = list(np.asarray(mat, dtype=np.int8).flatten(order="F"))
    words = []
    items = cfg.items_per_word
    for i in range(0, len(flat), items):
        chunk = flat[i : i + items]
        while len(chunk) < items:
            chunk.append(0)
        words.append(pack_values_to_word(chunk, cfg))
    return words


def _build_descriptor_program(w: np.ndarray, x: np.ndarray, n: int) -> bytes:
    """Host BSTORE hoist payloads + TILE_GEMM + FETCH — emission from frozen layout."""
    out_f, in_f = w.shape
    batch = x.shape[0]
    out_blocks = (out_f + n - 1) // n
    in_blocks = (in_f + n - 1) // n
    w_words, x_words, y_words = hoist_layout_words(
        array_size=n, batch_size=batch, items_per_word=DESC_CFG.items_per_word
    )
    fetch_words = (batch * n) // DESC_CFG.items_per_word
    weight_base = 0
    input_base = out_blocks * in_blocks * w_words
    result_base = input_base + in_blocks * x_words

    w_pad = np.zeros((out_blocks * n, in_blocks * n), dtype=np.int8)
    x_pad = np.zeros((batch, in_blocks * n), dtype=np.int8)
    w_pad[:out_f, :in_f] = w
    x_pad[:, :in_f] = x

    enc = ISAEncoder(cfg=DESC_CFG)
    for ob in range(out_blocks):
        for ib in range(in_blocks):
            tile = w_pad[ob * n : (ob + 1) * n, ib * n : (ib + 1) * n]
            enc.burst_store(weight_base + (ob * in_blocks + ib) * w_words, _pack_c(tile, DESC_CFG))
    for ib in range(in_blocks):
        block = x_pad[:, ib * n : (ib + 1) * n]
        mat = np.zeros((n, batch), dtype=np.int8)
        mat[:, :batch] = block.T
        enc.burst_store(input_base + ib * x_words, _pack_f(mat, DESC_CFG))

    enc.tileGemm(
        weight_base=weight_base,
        input_base=input_base,
        result_base=result_base,
        out_blocks=out_blocks,
        in_blocks=in_blocks,
        batch_size=batch,
        array_size=n,
        hoist_payloads=True,
        finalize_last=True,
        acc_clear_first=True,
    )
    for ob in range(out_blocks):
        base = result_base + ob * y_words
        for widx in range(fetch_words):
            enc.fetch(base + widx, top_half=False)
            enc.fetch(base + widx, top_half=True)
    enc.halt()
    return enc.getProgram()


def test_tile_gemm_encoding_matches_pinned_fixture() -> None:
    fixture = json.loads(
        (HOST / "fixtures" / "tile_gemm_frozen_encoding.json").read_text(encoding="utf-8")
    )
    prog = encodeTileGemm(
        weight_base=fixture["fields"]["weight_base"],
        input_base=fixture["fields"]["input_base"],
        result_base=fixture["fields"]["result_base"],
        out_blocks=fixture["fields"]["out_blocks"],
        in_blocks=fixture["fields"]["in_blocks"],
        batch_size=fixture["fields"]["batch_size"],
        array_size=fixture["fields"]["array_size"],
        cfg=DESC_CFG,
        acc_clear_first=fixture["fields"]["acc_clear_first"],
        finalize_last=fixture["fields"]["finalize_last"],
        hoist_payloads=fixture["fields"]["hoist_payloads"],
        weight_overlap=fixture["fields"]["weight_overlap"],
    )
    assert prog.hex() == fixture["program_bytes_hex"]
    assert _words(prog) == fixture["program_words_le"]
    desc = TileGemmDesc.from_words(_words(prog), DESC_CFG)
    assert desc.batch_size == 48
    assert desc.array_size == 16


def test_tile_gemm_encoding_frozen_layout() -> None:
    prog = encodeTileGemm(
        weight_base=0x10,
        input_base=0x40,
        result_base=0x80,
        out_blocks=2,
        in_blocks=3,
        batch_size=48,
        array_size=16,
        cfg=DESC_CFG,
    )
    assert len(prog) == TILE_GEMM_WORD_COUNT * 2
    words = _words(prog)
    assert words[0] & 0xF == OPCODE_TILE_GEMM
    assert ((words[0] >> 10) & 0x3F) == 47
    flags = (words[0] >> 6) & 0xF
    assert flags & TILE_GEMM_FLAG_ACC_CLEAR_FIRST
    assert flags & TILE_GEMM_FLAG_FINALIZE_LAST
    assert flags & TILE_GEMM_FLAG_HOIST_PAYLOADS
    decoded = decodeTileGemmWords(words, cfg=DESC_CFG)
    assert decoded["batch_size"] == 48


def test_tile_gemm_requires_descriptor_flag() -> None:
    with pytest.raises(ValueError, match="descriptor_isa"):
        encodeTileGemm(
            weight_base=0,
            input_base=0,
            result_base=0,
            out_blocks=1,
            in_blocks=1,
            batch_size=1,
            array_size=16,
            cfg=LEGACY_EXT,
        )


def test_legacy_programs_unaffected_by_descriptor_constants() -> None:
    enc = ISAEncoder(cfg=LEGACY_EXT)
    enc.halt()
    words = _words(enc.getProgram())
    assert words[0] & 0x7 == 0b100
    assert words[0] & 0x8 == 0


def test_golden_vs_unrolled_microcoded_result_bytes() -> None:
    """Check 1: spec NumPy golden vs fully-unrolled lowering emission (ISA sim)."""
    n = 16
    out_f, in_f, batch = 32, 32, 4
    rng = np.random.default_rng(0x71E)
    w = rng.integers(-8, 8, size=(out_f, in_f), dtype=np.int8)
    x = rng.integers(-8, 8, size=(batch, in_f), dtype=np.int8)

    _, golden_bytes = golden_tile_gemm(w, x, array_size=n, compute_data_width=4)

    micro = lower_blocked_fc_program_utpu(
        w, x, out_f, in_f, n, False, True, 0, 128, 256,
        cfg=LEGACY_EXT, hoist_tile_payloads=True,
    )
    micro_sim = simulate_program_bytes(
        micro["program"], array_size=n, buffer_size=1 << 12, cfg=LEGACY_EXT
    )
    assert micro_sim.halted
    assert list(micro_sim.fetch_bytes) == golden_bytes


def test_golden_vs_isa_simulator_tile_gemm() -> None:
    """Check 2: spec NumPy golden vs ISA-sim TILE_GEMM expansion."""
    n = 16
    out_f, in_f, batch = 32, 32, 4
    rng = np.random.default_rng(0xA11CE)
    w = rng.integers(-8, 8, size=(out_f, in_f), dtype=np.int8)
    x = rng.integers(-8, 8, size=(batch, in_f), dtype=np.int8)

    _, golden_bytes = golden_tile_gemm(w, x, array_size=n, compute_data_width=4)
    prog = _build_descriptor_program(w, x, n)
    sim = simulate_program_bytes(
        prog, array_size=n, buffer_size=1 << 12, cfg=DESC_CFG
    )
    assert sim.halted
    assert sim.executed_ops.get("tile_gemm", 0) == 1
    assert list(sim.fetch_bytes) == golden_bytes


def test_golden_vs_unrolled_and_isa_agree_same_seed() -> None:
    """Same GEMM: unrolled emission and TILE_GEMM expansion both match golden."""
    n = 16
    out_f, in_f, batch = 32, 16, 4
    rng = np.random.default_rng(0xC0FFEE)
    w = rng.integers(-8, 8, size=(out_f, in_f), dtype=np.int8)
    x = rng.integers(-8, 8, size=(batch, in_f), dtype=np.int8)
    _, golden_bytes = golden_tile_gemm(w, x, array_size=n, compute_data_width=4)

    micro = lower_blocked_fc_program_utpu(
        w, x, out_f, in_f, n, False, True, 0, 128, 256,
        cfg=LEGACY_EXT, hoist_tile_payloads=True,
    )
    micro_bytes = list(
        simulate_program_bytes(
            micro["program"], array_size=n, buffer_size=1 << 12, cfg=LEGACY_EXT
        ).fetch_bytes
    )
    desc_bytes = list(
        simulate_program_bytes(
            _build_descriptor_program(w, x, n),
            array_size=n,
            buffer_size=1 << 12,
            cfg=DESC_CFG,
        ).fetch_bytes
    )
    assert micro_bytes == golden_bytes
    assert desc_bytes == golden_bytes


@pytest.mark.skip(reason="RTL TILE_GEMM sequencer not landed (DESCRIPTOR_ISA_EN); check 3 open")
def test_golden_vs_rtl_sequencer_check3() -> None:
    """Check 3: spec NumPy golden vs RTL DESC_* sequencer FETCH bytes."""
    raise AssertionError("unreachable until sequencer lands")
