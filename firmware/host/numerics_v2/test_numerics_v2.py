import ast
import json
from pathlib import Path

import numpy as np

from firmware.host.numerics_v2.fixed import div_const_rne, div_rne, mul_recip, reciprocal_rne, rshift_rne
from firmware.host.numerics_v2.export import accumulator_bounds, export_model, export_runq_model
from firmware.host.numerics_v2.runtime import IntegerLlama
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
from tools.llama2c.reference import Llama2CModel

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def test_constant_reciprocal_division_exact():
    values = np.arange(-100000, 100001, dtype=np.int64)
    for denominator in (7, 127):
        np.testing.assert_array_equal(div_const_rne(values, denominator), div_rne(values, denominator))
        boundary = np.asarray([-(1 << 31), -1, 0, 1, (1 << 31) - 1], dtype=np.int64)
        np.testing.assert_array_equal(div_const_rne(boundary, denominator), div_rne(boundary, denominator))


def test_integer_forward_sources_have_no_float_constants_or_float_dtypes():
    for name in ("fixed.py", "runtime.py", "runq_runtime.py"):
        source = (HERE / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float)]
        assert "float16" not in source and "float32" not in source and "float64" not in source


def test_rne_signed_ties_to_even():
    values = np.asarray([-7, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 7])
    assert rshift_rne(values, 1).tolist() == [-4, -3, -2, -2, -1, 0, 0, 1, 2, 2, 3, 4]


def test_q24_reciprocal_uses_one_divide_then_elementwise_multiply_shift():
    values = np.asarray([-1000, -7, 0, 7, 1000], dtype=np.int64)
    reciprocal = reciprocal_rne(127, 1000)
    assert mul_recip(values, reciprocal).tolist() == [-127, -1, 0, 1, 127]


def test_lut_metadata_and_mem_depths():
    document = json.loads((HERE / "luts/luts.json").read_text(encoding="utf-8"))
    for name, record in document.items():
        assert len(record["values"]) == record["shape"][0]
        assert len((HERE / f"luts/{name}.mem").read_text(encoding="ascii").splitlines()) == record["depth"]
    assert document["exp_q15"]["depth"] == 1025
    assert document["sigmoid_q15"]["depth"] == 1025


def test_all_accumulator_proofs_fit_int32():
    reference = Llama2CModel(ROOT / "models/llama2c/stories260K/stories260K.bin")
    for bits in (8, 4):
        for row in accumulator_bounds(reference, bits, 32).values():
            assert row["fits_int32"] and row["worst_case"] <= (1 << 31) - 1


def test_vector_teacher_forcing_is_bit_exact_with_cached_decode():
    reference = Llama2CModel(ROOT / "models/llama2c/stories260K/stories260K.bin")
    scales = json.loads((HERE / "calibration_260k.json").read_text(encoding="utf-8"))["activation_scales"]
    exported = export_model(reference, scales, 4, 32)
    cached = IntegerLlama(exported, HERE / "luts")
    ids = [1, 10, 20, 30, 40, 50, 60, 70]
    expected = np.stack([cached.forward(token, pos) for pos, token in enumerate(ids)])
    vector = IntegerLlama(exported, HERE / "luts").forward_sequence(ids)
    assert np.array_equal(expected, vector)


def test_runq_vector_teacher_forcing_is_bit_exact_with_cached_decode():
    reference = Llama2CModel(ROOT / "models/llama2c/stories260K/stories260K.bin")
    scales = json.loads((HERE / "calibration_260k.json").read_text(encoding="utf-8"))["activation_scales"]
    exported = export_runq_model(reference, scales, 8, 64)
    exported.pop("dynamic_base_scales")
    ids = [1, 10, 20, 30, 40, 50, 60, 70]
    cached = RunQIntegerLlama(exported, HERE / "luts")
    expected = np.stack([cached.forward(token, pos) for pos, token in enumerate(ids)])
    vector = RunQIntegerLlama(exported, HERE / "luts").forward_sequence(ids)
    assert np.array_equal(expected, vector)


def test_chunked_integer_logits_and_argmax_match_full_classifier():
    reference = Llama2CModel(ROOT / "models/llama2c/stories260K/stories260K.bin")
    scales = json.loads((HERE / "calibration_260k.json").read_text(encoding="utf-8"))["activation_scales"]
    exported = export_runq_model(reference, scales, 8, 64)
    exported.pop("dynamic_base_scales")
    ids = [1, 10, 20, 30]
    full_model = RunQIntegerLlama(exported, HERE / "luts")
    full = full_model.forward_sequence(ids)
    chunk_model = RunQIntegerLlama(exported, HERE / "luts")
    chunked = np.concatenate([values for _, values in chunk_model.forward_sequence_logits_chunks(ids, 73)], axis=-1)
    assert np.array_equal(full, chunked)
    decode = RunQIntegerLlama(exported, HERE / "luts")
    predicted = [decode.forward_argmax(token, pos, 73) for pos, token in enumerate(ids)]
    assert predicted == np.argmax(full, axis=1).tolist()


def test_rms_gamma_format_covers_checkpoint_without_saturation():
    reference = Llama2CModel(ROOT / "models/llama2c/stories260K/stories260K.bin")
    scales = json.loads((HERE / "calibration_260k.json").read_text(encoding="utf-8"))["activation_scales"]
    exported = export_runq_model(reference, scales, 8, 64)
    assert exported["gamma_fraction_bits"] == 12
    assert int(np.max(exported["gamma_final"])) < 65535


def test_exported_forward_state_contains_integer_codes_and_scale_pairs_only():
    reference = Llama2CModel(ROOT / "models/llama2c/stories260K/stories260K.bin")
    scales = json.loads((HERE / "calibration_260k.json").read_text(encoding="utf-8"))["activation_scales"]
    exported = export_model(reference, scales, 4, 64)

    def check(value):
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, np.ndarray):
            assert value.dtype.kind in "iu"
        else:
            assert isinstance(value, int) and not isinstance(value, bool)

    check(exported)
    for layer in exported["layers"]:
        for linear in layer.values():
            assert linear["multiplier"].shape == linear["shift"].shape


def test_calibration_is_train_only_and_hash_disjoint():
    metadata = json.loads((HERE / "calibration_260k.json").read_text(encoding="utf-8"))["metadata"]
    assert metadata["split"] == "train" and metadata["sequences"] == 128
    assert metadata["source_sha256"] != metadata["eval_slice_sha256"]
    assert metadata["provably_disjoint"] is True


def test_committed_int8_result_passes_quality_gate_and_reports_divergence():
    result = json.loads((ROOT / "bench/results/numerics_v2_260k.json").read_text(encoding="utf-8"))
    int8 = result["configs"][0]
    assert int8["perplexity"] <= result["quality_target"]["float_perplexity"] * 1.02
    assert int8["teacher_forced_top1_agreement_pct"] >= 97.0
    assert "logit_kl_float_to_candidate" in int8
    assert "median_first_divergence_step" in int8["greedy_agreement"]
    assert "identical_tokens_pct_vs_float" not in int8["greedy_agreement"]


# ---- numeric domain contract (audit N07/F10/F11; see docs/numeric_domain.md) ---------------------------------------
import importlib
import sys

import pytest


def _numeric_domain():
    sys.path.insert(0, str(ROOT / "tools/decode"))
    return importlib.import_module("numeric_domain")


HAVE_15M = (ROOT / "models/llama2c/stories15M.bin").exists() and (ROOT / "models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz").exists()


def test_div_const_rne_is_exact_rne_up_to_2_pow_31():
    nd = _numeric_domain()
    rng = np.random.default_rng(5)
    for d in (7, 64, 127, 288):
        lim = 1 << 31
        values = np.concatenate([rng.integers(-lim, lim + 1, 200_000), np.arange(-5000, 5001),
                                 [lim, -lim, lim - 1, 1 - lim], (lim // d) * d + np.arange(-d, d + 1)])
        values = values[np.abs(values) <= lim]
        np.testing.assert_array_equal(div_const_rne(values, d), div_rne(values, d))


def test_div_const_rne_wide_path_is_approximate_and_error_matches_first_order_bound():
    # |n| > 2^31 takes the wide path: round-half-up estimate off by about n*|R/2^32 - 1/d| and one +-1 correction.
    # Auditor example: mean of 288 squares of 65536 is exactly 2^32 but div_const_rne(288 * 2^32, 288) returns 2^32 + 31.
    ss = np.int64(288) * (1 << 32)
    assert int(div_const_rne(ss, 288)) == 4294967327 and int(div_rne(ss, 288)) == 1 << 32
    nd = _numeric_domain()
    rng = np.random.default_rng(6)
    for d, hi in ((288, nd.RMS_SS_HI), (127, 127 * (1 << 31)), (7, 7 * (1 << 31)), (64, nd.RMS_SS_HI)):
        prof = nd.error_profile(d, hi, 20_000, rng)
        assert prof["max_err"] <= int(np.ceil(prof["first_order_err_at_hi"])) + 1, prof
        if d == 64:
            assert prof["max_err"] == 0          # power-of-two reciprocal is exact


def _check_reachable(model, kv_bits, prompts, length):
    nd = _numeric_domain()
    probe, info, exp, rt = nd.probe_model(model, kv_bits, prompts, length, 8)
    assert probe.sites, "no div_const_rne call sites observed"
    for (site, d), r in probe.sites.items():
        name = site.split(":")[0]
        hi, _ = nd.site_hi(site, d, kv_bits)
        if hi is not None:
            assert r["max_abs_n"] <= hi, (model, kv_bits, site, d, r)
        if name != "_rms_high":
            # every non-RMSNorm site stays on the exact (|n| <= 2^31) path over eval text
            assert r["max_abs_n"] <= nd.NARROW_LIMIT and r["wide_values"] == 0, (model, kv_bits, site, d, r)
        else:
            assert r["max_abs_n"] <= nd.RMS_SS_HI
            bound = int(np.ceil(nd.error_profile(d, nd.RMS_SS_HI, 1000, np.random.default_rng(0))["first_order_err_at_hi"])) + 1
            assert r["max_err"] <= bound, (site, d, r)
    q = probe.quant
    # F11: no quantizer input is INT32_MIN (the runtime's _quantize rejects it), and none comes close
    assert q["int32_min_hits"] == 0 and q["min_raw"] > -(1 << 28) and q["max_raw"] < (1 << 28), q


@pytest.mark.parametrize("kv_bits", (8, 4))
def test_reachable_div_const_ranges_and_quantize_domain_260k(kv_bits):
    _check_reachable("260k", kv_bits, 6, 40)


@pytest.mark.skipif(not HAVE_15M, reason="stories15M model files not present")
@pytest.mark.parametrize("kv_bits", (8, 4))
def test_reachable_div_const_ranges_and_quantize_domain_15m(kv_bits):
    _check_reachable("15m", kv_bits, 4, 24)


def test_quantize_rejects_int32_min_and_accepts_int32_max():
    from firmware.host.numerics_v2 import runq_runtime as rr
    ok = rr._quantize(np.asarray([(1 << 31) - 1] + [0] * 63, dtype=np.int64))
    assert int(ok.scale_multiplier[0]) == (1 << 31) - 1 and int(ok.code[0]) == 127
    with pytest.raises(ValueError):
        rr._quantize(np.asarray([-(1 << 31)] + [0] * 63, dtype=np.int64))
