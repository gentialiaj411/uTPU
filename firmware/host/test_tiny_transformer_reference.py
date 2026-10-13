import hashlib
import json
from pathlib import Path

import numpy as np

from tools.gen_tiny_transformer_luts import build_payload, payload_bytes

from tiny_transformer_reference import TinyTransformerConfig, integer_decoder_logits, integer_decoder_trace, rmsnorm_i8, softmax_u8, vadd_rq


def _schema() -> dict:
    acts = {name: 0 for name in (
        'token', 'position', 'pos_add', 'rmsnorm1', 'qkv', 'q', 'k', 'v',
        'scores', 'softmax_probs', 'av', 'out_proj', 'residual1', 'rmsnorm2',
        'ffn1', 'leaky_relu', 'ffn2', 'residual2', 'final_norm', 'logits',
    )}
    record = {'input_exponent': 0, 'output_exponent': 0, 'multipliers': [1], 'right_shifts': [0]}
    return {
        'activations': acts,
        'linears': {name: dict(record) for name in ('qkv', 'scores', 'av', 'out', 'ff1', 'ff2', 'logits')},
        'norms': {name: {'input_exponent': 0, 'output_exponent': 0, 'gamma_exponent': 0, 'gamma_shift': 0} for name in ('n1', 'n2', 'final')},
        'residuals': {name: {'ea': 0, 'eb': 0, 'eo': 0} for name in ('pos_add', 'residual1', 'residual2')},
    }


def test_frozen_luts_are_generated_exactly_and_hash_pinned():
    fixture = Path(__file__).parent / 'fixtures' / 'tiny_transformer_integer_luts.json'
    committed = json.loads(fixture.read_text(encoding='utf-8'))
    generated = build_payload()
    assert len(committed['exp_q15']) == 129
    assert len(committed['rsqrt_q14']) == 192
    assert committed['exp_q15'] == generated['exp_q15']
    assert committed['rsqrt_q14'] == generated['rsqrt_q14']
    assert committed['payload_sha256'] == hashlib.sha256(payload_bytes(generated)).hexdigest()


def test_softmax_contract_masks_and_bounds():
    scores = np.array([12, 8, 1, -5], dtype=np.int8)
    actual = softmax_u8(scores, valid_count=3)
    assert actual.dtype == np.uint8
    assert int(actual[3]) == 0
    assert int(actual[0]) >= int(actual[1]) >= int(actual[2])
    assert 255 - 3 < int(actual.sum()) <= 255 + 2


def test_residual_uses_exported_output_exponent_and_reports_saturation():
    result, saturation_count = vadd_rq(
        np.array([127, -128], dtype=np.int8),
        0,
        np.array([127, -128], dtype=np.int8),
        0,
        1,
    )
    np.testing.assert_array_equal(result, np.array([127, -128], dtype=np.int8))
    # +127 rounds to +64 at eo=max+1; the sum is +128 and must be counted.
    assert saturation_count == 1


def test_vadd_dequantized_error_is_bounded_by_one_output_lsb():
    # Each input rounding contributes <=0.5 LSB and the final sum is exact:
    # after dequantization the absolute error is therefore <=1 output LSB.
    a, b = np.array([21, -35], np.int8), np.array([17, 11], np.int8)
    result, _ = vadd_rq(a, -7, b, -6, -5)
    expected = a.astype(np.float64) * 2.0**-7 + b.astype(np.float64) * 2.0**-6
    actual = result.astype(np.float64) * 2.0**-5
    assert np.max(np.abs(expected - actual)) <= 2.0**-5


def test_rmsnorm_dequantized_error_is_bounded_in_output_lsbs():
    code = np.array([32, -48, 64, -16], np.int8)
    # gamma=64 * 2^-6=1, and output uses the same 2^-6 exponent.
    actual = rmsnorm_i8(code, np.full(4, 64, np.int8), gamma_shift=0).astype(np.float64) * 2.0**-6
    expected = code.astype(np.float64) / np.sqrt(np.mean(code.astype(np.float64) ** 2) + 1.0)
    cosine = float(actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected)))
    # LUT seed + one Newton step is below two code LSBs in this calibrated case.
    assert cosine >= 0.99
    assert np.max(np.abs(actual - expected)) <= 2 * 2.0**-6


def test_config_is_parameterized_not_backend_pinned():
    cfg = TinyTransformerConfig(T=8, D=16, H=1, Dff=32, L=2, vocab=11)
    assert cfg.head_dim == 16
    assert cfg.parameter_count == 11 * 16 + 8 * 16 + 2 * (3 * 16 * 16 + 16 * 16 + 2 * 16 * 32 + 2 * 16) + 16


def test_independent_integer_decoder_executes_parameterized_model():
    cfg = TinyTransformerConfig(T=4, D=8, H=1, Dff=16, L=1, vocab=7)
    rng = np.random.default_rng(4)
    state = {
        'token.weight': rng.integers(-4, 5, (7, 8), dtype=np.int8), 'pos.weight': rng.integers(-4, 5, (4, 8), dtype=np.int8),
        'blocks.0.n1.weight': np.full(8, 127, np.int8), 'blocks.0.n2.weight': np.full(8, 127, np.int8),
        'blocks.0.qkv.weight': rng.integers(-4, 5, (24, 8), dtype=np.int8), 'blocks.0.out.weight': rng.integers(-4, 5, (8, 8), dtype=np.int8),
        'blocks.0.ff1.weight': rng.integers(-4, 5, (16, 8), dtype=np.int8), 'blocks.0.ff2.weight': rng.integers(-4, 5, (8, 16), dtype=np.int8), 'final.weight': np.full(8, 127, np.int8),
    }
    logits, counts = integer_decoder_logits(np.array([0,1,2,3]), state, _schema(), cfg)
    assert logits.shape == (1,4,7)
    assert set(counts) == {'position_add','residual_attention','residual_ffn'}


def test_integer_trace_exposes_every_phase1_stage_boundary():
    cfg = TinyTransformerConfig(T=2, D=4, H=1, Dff=8, L=1, vocab=5)
    state = {
        'token.weight': np.ones((5, 4), np.int8), 'pos.weight': np.ones((2, 4), np.int8),
        'blocks.0.n1.weight': np.full(4, 127, np.int8), 'blocks.0.n2.weight': np.full(4, 127, np.int8),
        'blocks.0.qkv.weight': np.ones((12, 4), np.int8), 'blocks.0.out.weight': np.ones((4, 4), np.int8),
        'blocks.0.ff1.weight': np.ones((8, 4), np.int8), 'blocks.0.ff2.weight': np.ones((4, 8), np.int8), 'final.weight': np.full(4, 127, np.int8),
    }
    trace = integer_decoder_trace(np.array([0, 1]), state, _schema(), cfg)
    expected = {'pos_add', 'rmsnorm1', 'q', 'k', 'v', 'scores', 'softmax_probs', 'av', 'out_proj', 'residual1', 'rmsnorm2', 'ffn1', 'leaky_relu', 'ffn2', 'residual2', 'final_norm', 'logits'}
    assert expected <= set(trace)
