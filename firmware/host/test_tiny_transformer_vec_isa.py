"""Phase-2 VEC descriptor simulator contract tests."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from isa_encoder import (
    IsaConfig, decodeVecWords, encodeFetch, encodeHalt, encodeLoadInputs, encodeLoadWeights, encodeRun, encodeBurstStore, encodeTileGemm, encode_vec_vadd_rq,
    encode_vec_vpos_add, encode_vec_vrmsnorm, encode_vec_vsoftmax, encode_vec_vmatmul, encode_vec_vlayout,
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
    bad = _words(encodings[0]); bad[1] = (bad[1] & ~0xF0) | 0x60
    with pytest.raises(ValueError, match="unknown/reserved"):
        decodeVecWords(bad, cfg=CFG)
    bad = _words(encodings[0]); bad[5] = 0
    with pytest.raises(ValueError, match="zero length"):
        decodeVecWords(bad, cfg=CFG)


def _matmul_record(*, m, k, n, b_base, a_stride, b_stride, transpose=False,
                   a_unsigned=False, b_unsigned=False, causal=False, multiplier=1, shift=0):
    return [5, b_base, (m - 1) | ((k - 1) << 5) | ((n - 1) << 10) | (int(transpose) << 15),
            a_stride | (b_stride << 8), int(a_unsigned) | (int(b_unsigned) << 1) | (int(causal) << 2),
            multiplier, shift, 0]


def test_vmatmul_signed_transpose_and_unsigned_causal_pv():
    # QK^T uses signed A/B and masks upper-triangular entries.
    q = np.array([[2, -1, 3, 1], [1, 2, -2, 0], [-1, 3, 1, 2], [2, 1, 0, -2]], dtype=np.int8)
    k = np.array([[1, 1, 0, 2], [2, -1, 1, 0], [0, 3, -2, 1], [-1, 2, 2, 1]], dtype=np.int8)
    def qk_setup(sim):
        _put(sim, SRC0, q); _put(sim, AUX, k)
        sim.buffer[PARAM:PARAM + 8] = _matmul_record(m=4, k=4, n=4, b_base=AUX, a_stride=4, b_stride=4, transpose=True, causal=True)
    qk = _run(encode_vec_vmatmul(DST, SRC0, PARAM, 16, causal=True, cfg=CFG), qk_setup)
    expected_qk = q.astype(np.int64) @ k.astype(np.int64).T
    expected_qk[np.triu_indices(4, 1)] = 0
    np.testing.assert_array_equal(_get(qk, DST, 16).reshape(4, 4), expected_qk.astype(np.int8))

    # P*V must zero-extend probabilities above 127 and bounds K to i+1.
    p = np.array([[255, 0, 0, 0], [128, 127, 0, 0], [1, 200, 54, 0], [10, 20, 30, 40]], dtype=np.uint8)
    v = np.array([[1, -2], [2, 3], [-1, 4], [3, -4]], dtype=np.int8)
    def pv_setup(sim):
        _put(sim, SRC0, p); _put(sim, AUX, v)
        sim.buffer[PARAM:PARAM + 8] = _matmul_record(m=4, k=4, n=2, b_base=AUX, a_stride=4, b_stride=2, a_unsigned=True, causal=True)
    pv = _run(encode_vec_vmatmul(DST, SRC0, PARAM, 8, causal=True, cfg=CFG), pv_setup)
    expected_pv = np.zeros((4, 2), dtype=np.int64)
    for row in range(4): expected_pv[row] = p[row, :row + 1].astype(np.int64) @ v[:row + 1].astype(np.int64)
    np.testing.assert_array_equal(_get(pv, DST, 8).reshape(4, 2), np.clip(expected_pv, -128, 127).astype(np.int8))


def test_vlayout_round_trip_and_descriptor_errors():
    source = np.arange(-8, 8, dtype=np.int8).reshape(4, 4)
    record0 = [6, (4 - 1) | ((4 - 1) << 5), 0, 4 | (4 << 8), 0, 0, 0, 0]
    record3 = [6, (4 - 1) | ((4 - 1) << 5), 3, 4 | (4 << 8), 0, 0, 0, 0]
    def setup(sim):
        _put(sim, SRC0, source)
        sim.buffer[PARAM:PARAM + 8] = record0
        sim.buffer[PARAM + 8:PARAM + 16] = record3
    program = (encode_vec_vlayout(AUX, SRC0, PARAM, 16, cfg=CFG) +
               encode_vec_vlayout(DST, AUX, PARAM + 8, 16, cfg=CFG))
    sim = _run(program, setup)
    np.testing.assert_array_equal(_get(sim, DST, 16).reshape(4, 4), source)
    bad = _matmul_record(m=2, k=2, n=2, b_base=AUX, a_stride=2, b_stride=2, causal=True)
    def bad_setup(sim):
        _put(sim, SRC0, np.ones(4, np.int8)); _put(sim, AUX, np.ones(4, np.int8)); sim.buffer[PARAM:PARAM + 8] = bad
    with pytest.raises(ValueError, match="illegal VMATMUL causal"):
        _run(encode_vec_vmatmul(DST, SRC0, PARAM, 4, causal=True, cfg=CFG), bad_setup)


def test_vlayout_legacy_load_stream_round_trip():
    """Modes 6/7 match the real B=4, eight-K-lane legacy LOAD ordering."""
    source = np.arange(-16, 16, dtype=np.int8).reshape(4, 8)
    record6 = [6, (4 - 1) | ((8 - 1) << 5), 6, 8 | (8 << 8), 0, 0, 0, 0]
    record7 = [6, (4 - 1) | ((8 - 1) << 5), 7, 8 | (8 << 8), 0, 0, 0, 0]
    def setup(sim):
        _put(sim, SRC0, source); sim.buffer[PARAM:PARAM + 8] = record6; sim.buffer[PARAM + 8:PARAM + 16] = record7
    sim = _run(encode_vec_vlayout(AUX, SRC0, PARAM, 32, cfg=CFG) + encode_vec_vlayout(DST, AUX, PARAM + 8, 32, cfg=CFG), setup)
    np.testing.assert_array_equal(_get(sim, DST, 32).reshape(4, 8), source)


def test_mixed_vector_and_legacy_fetch_top_half_stays_fetch():
    """The VEC escape, not low nibble 0x9, owns descriptor dispatch."""
    def setup(sim):
        _put(sim, SRC0, np.ones(48, np.int8)); _put(sim, SRC1, np.ones(48, np.int8))
        _record(sim, 1, [SRC1, ((-5 & 0xFF) << 8) | (-5 & 0xFF), -4 & 0xFF])
        sim.buffer[64] = 0x3412
    vec = encode_vec_vadd_rq(DST, SRC0, PARAM, 48, cfg=CFG)
    sim = _run(vec + encodeFetch(64, True, cfg=CFG) + encodeFetch(64, False, cfg=CFG), setup)
    assert sim.fetch_bytes == [0x34, 0x12]
    assert sim.executed_ops["vec"] == 1
