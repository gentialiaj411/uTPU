import ast
import json
from pathlib import Path

import numpy as np

from firmware.host.numerics_v2.fixed import rshift_rne
from firmware.host.numerics_v2.export import accumulator_bounds, export_model
from firmware.host.numerics_v2.runtime import IntegerLlama
from tools.llama2c.reference import Llama2CModel

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def test_integer_forward_sources_have_no_float_constants_or_float_dtypes():
    for name in ("fixed.py", "runtime.py"):
        source = (HERE / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, float)]
        assert "float16" not in source and "float32" not in source and "float64" not in source


def test_rne_signed_ties_to_even():
    values = np.asarray([-7, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 7])
    assert rshift_rne(values, 1).tolist() == [-4, -3, -2, -2, -1, 0, 0, 1, 2, 2, 3, 4]


def test_lut_metadata_and_mem_depths():
    document = json.loads((HERE / "luts/luts.json").read_text(encoding="utf-8"))
    for name, record in document.items():
        assert len(record["values"]) == record["shape"][0]
        assert len((HERE / f"luts/{name}.mem").read_text(encoding="ascii").splitlines()) == record["depth"]


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
