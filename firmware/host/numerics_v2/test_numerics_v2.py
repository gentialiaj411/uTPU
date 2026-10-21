import ast
import json
from pathlib import Path

import numpy as np

from firmware.host.numerics_v2.fixed import mul_recip, reciprocal_rne, rshift_rne
from firmware.host.numerics_v2.export import accumulator_bounds, export_model, export_runq_model
from firmware.host.numerics_v2.runtime import IntegerLlama
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
from tools.llama2c.reference import Llama2CModel

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


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
