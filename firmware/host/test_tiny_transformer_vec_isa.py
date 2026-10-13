"""Phase-2 VEC descriptor simulator contract tests."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from isa_encoder import (
    IsaConfig, decodeVecWords, encodeHalt, encode_vec_vadd_rq,
    encode_vec_vpos_add, encode_vec_vrmsnorm, encode_vec_vsoftmax,
)
from isa_simulator import UTPUISASimulator
from tiny_transformer_reference import rmsnorm_i8, softmax_u8, vadd_rq


CFG = IsaConfig(address_width=16, compute_data_width=8, descriptor_isa=True, vector_unit=True)
SRC0, SRC1, DST, PARAM, AUX, SCRATCH = 100, 2000, 4000, 7000, 9000, 12000


def _words(program: bytes) -> list[int]:
    return list(struct.unpack("<" + "H" * (len(program) // 2), program))


def _put(sim: UTPUISASimulator, base: int, values: np.ndarray) -> None:
    flat = np.asarray(values).reshape(-1)
    assert len(flat) % 2 == 0
    for i in range(0, len(flat), 2):
        sim.buffer[base + i // 2] = (int(flat[i]) & 0xFF) | ((int(flat[i + 1]) & 0xFF) << 8)


def _get(sim: UTPUISASimulator, base: int, length: int, *, unsigned: bool = False) -> np.ndarray:
    raw = np.empty(length, np.int64)
    for i in range(length // 2):
        word = sim.buffer[base + i]
        raw[2 * i], raw[2 * i + 1] = word & 0xFF, (word >> 8) & 0xFF
    if not unsigned:
        raw = np.where(raw >= 128, raw - 256, raw)
    return raw.astype(np.uint8 if unsigned else np.int8)


def _run(program: bytes, setup) -> UTPUISASimulator:
    sim = UTPUISASimulator(array_size=16, buffer_size=65536, cfg=CFG)
    setup(sim)
    sim.run_words(_words(program + encodeHalt(cfg=CFG)), preserve_buffer=True)
    return sim


def _record(sim: UTPUISASimulator, kind: int, words: list[int]) -> None:
    values = [kind, *words]
    sim.buffer[PARAM:PARAM + 8] = values + [0] * (8 - len(values))


@pytest.mark.parametrize("ea,eb,eo", [(-5, -5, -4), (-4, -1, 0), (-31, -31, -30)])
def test_vadd_rq_exhaustive_i8_pairs(ea: int, eb: int, eo: int):
    # Every int8 pair is covered; a VEC descriptor handles 1536 values at once.
    pairs = np.arange(-128, 128, dtype=np.int16)
    a, b = np.meshgrid(pairs, pairs, indexing="ij")
    flat_a, flat_b = a.reshape(-1).astype(np.int8), b.reshape(-1).astype(np.int8)
    got = []
    for start in range(0, len(flat_a), 1536):
        end = min(start + 1536, len(flat_a))
        left, right = flat_a[start:end], flat_b[start:end]
        if len(left) < 1536:
            left = np.pad(left, (0, 1536 - len(left)))
            right = np.pad(right, (0, 1536 - len(right)))
        def setup(sim):
            _put(sim, SRC0, left); _put(sim, SRC1, right)
            _record(sim, 1, [SRC1, ((eb & 0xFF) << 8) | (ea & 0xFF), eo & 0xFF])
        sim = _run(encode_vec_vadd_rq(DST, SRC0, PARAM, len(left), cfg=CFG), setup)
        got.append(_get(sim, DST, len(left))[:end - start])
    expected, _ = vadd_rq(flat_a, ea, flat_b, eb, eo)
    np.testing.assert_array_equal(np.concatenate(got), expected)


def test_vpos_add_all_positions_and_random_table():
    rng = np.random.default_rng(7)
    source = rng.integers(-128, 128, 1536, dtype=np.int8)
    position = rng.integers(-128, 128, 1536, dtype=np.int8)
    def setup(sim):
        _put(sim, SRC0, source); _put(sim, AUX, position)
        _record(sim, 3, [AUX, ((-5 & 0xFF) << 8) | (-6 & 0xFF), -4 & 0xFF])
    sim = _run(encode_vec_vpos_add(DST, SRC0, PARAM, cfg=CFG), setup)
    expected, _ = vadd_rq(source, -6, position, -5, -4)
    np.testing.assert_array_equal(_get(sim, DST, 1536), expected)


def test_rmsnorm_adversarial_vectors_and_all_lut_bins():
    rng = np.random.default_rng(9)
    rows = [np.zeros(48, np.int8), np.full(48, -128, np.int8), np.full(48, 127, np.int8)]
    rows += [rng.integers(-128, 128, 48, dtype=np.int8) for _ in range(29)]
    source = np.concatenate(rows)
    gamma = rng.integers(-128, 128, 48, dtype=np.int8)
    def setup(sim):
        _put(sim, SRC0, source); _put(sim, AUX, gamma); _record(sim, 2, [AUX, 3])
    sim = _run(encode_vec_vrmsnorm(DST, SRC0, PARAM, len(source), cfg=CFG), setup)
    expected = np.concatenate([rmsnorm_i8(row, gamma, 3) for row in rows])
    np.testing.assert_array_equal(_get(sim, DST, len(source)), expected)


def test_causal_softmax_contract_and_inplace():
    rng = np.random.default_rng(11)
    scores = rng.integers(-128, 128, 1024, dtype=np.int8)
    def setup(sim):
        _put(sim, SRC0, scores); _record(sim, 4, [SCRATCH])
    sim = _run(encode_vec_vsoftmax(SRC0, SRC0, PARAM, head_id=2, cfg=CFG), setup)
    actual = _get(sim, SRC0, 1024, unsigned=True).reshape(32, 32)
    expected = np.stack([softmax_u8(scores.reshape(32, 32)[row], row + 1) for row in range(32)])
    np.testing.assert_array_equal(actual, expected)
    for row in range(32):
        assert not actual[row, row + 1:].any()
        assert 255 - (row + 1) < int(actual[row].sum()) <= 255 + ((row + 2) // 2)


def test_vec_encode_decode_and_errors():
    encodings = [
        encode_vec_vadd_rq(DST, SRC0, PARAM, 48, cfg=CFG),
        encode_vec_vrmsnorm(DST, SRC0, PARAM, 48, cfg=CFG),
        encode_vec_vpos_add(DST, SRC0, PARAM, cfg=CFG),
        encode_vec_vsoftmax(DST, SRC0, PARAM, head_id=1, cfg=CFG),
    ]
    assert [decodeVecWords(_words(item), cfg=CFG)["subop"] for item in encodings] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="VEC requires"):
        encode_vec_vadd_rq(DST, SRC0, PARAM, 48, cfg=IsaConfig(address_width=16, compute_data_width=8, descriptor_isa=True))
    bad = _words(encodings[0]); bad[0] = (bad[0] & ~0xF0) | 0x40
    with pytest.raises(ValueError, match="unknown/reserved"):
        decodeVecWords(bad, cfg=CFG)
    bad = _words(encodings[0]); bad[4] = 0
    with pytest.raises(ValueError, match="zero length"):
        decodeVecWords(bad, cfg=CFG)
