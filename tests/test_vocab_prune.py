from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.llama2c.tokenizer import Tokenizer
from tools.vocabprune.core import build_idmap, decode_compact, encode_with_fallback, prune_export


def test_id_map_order_is_deterministic():
    tok = Tokenizer(ROOT / "models/llama2c/tokenizer.bin", 32000)
    counts = {token: (token % 7) + 1 for token in range(300, 2089)}
    counts.update({300: 9, 301: 9, 302: 4, 303: 1})
    a = build_idmap(tok, counts, 23, 2048, ROOT / "models/llama2c/eval_slice.txt")
    b = build_idmap(tok, dict(reversed(list(counts.items()))), 23, 2048, ROOT / "models/llama2c/eval_slice.txt")
    assert a["kept_original_ids"] == b["kept_original_ids"]
    assert a["kept_original_ids"][:259] == list(range(259))
    assert a["kept_original_ids"][259:] == sorted(a["kept_original_ids"][259:], key=lambda x: (-(counts[x]), x))


@pytest.mark.skipif(not (ROOT / "models/llama2c/tokenizer.bin").exists(), reason="model artifacts unavailable")
def test_byte_fallback_round_trip():
    tok = Tokenizer(ROOT / "models/llama2c/tokenizer.bin", 32000)
    idmap = json.loads((ROOT / "tools/vocabprune/idmaps/idmap_2048.json").read_text())
    source = "A tiny café walked through the forest."
    compact, _, _, _ = encode_with_fallback(tok, source, idmap)
    assert decode_compact(tok, compact, idmap) == source


@pytest.mark.skipif(not (ROOT / "models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz").exists(), reason="model artifacts unavailable")
def test_kept_row_logits_are_bit_identical():
    from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama, _linear_high

    payload = np.load(ROOT / "models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz", allow_pickle=True)["payload"][0]
    idmap = json.loads((ROOT / "tools/vocabprune/idmaps/idmap_8192.json").read_text())
    kept = np.asarray(idmap["kept_original_ids"], dtype=np.int64)
    full = RunQIntegerLlama(payload, ROOT / "firmware/host/numerics_v2/luts", kv_bits=4)
    reduced = RunQIntegerLlama(prune_export(payload, idmap["kept_original_ids"]),
                               ROOT / "firmware/host/numerics_v2/luts", kv_bits=4)
    original = [1, 9038, 2501, 263, 29892]
    compact = [int(idmap["original_to_compact"][x]) for x in original]
    assert all(x >= 0 for x in compact)
    expected = _linear_high(full._forward_sequence_hidden(np.asarray(original, dtype=np.int64)), full.classifier)[:, kept]
    actual = reduced.forward_sequence(compact)
    assert np.array_equal(expected, actual)


@pytest.mark.skipif(not (ROOT / "models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz").exists(), reason="model artifacts unavailable")
def test_default_pack_is_byte_identical():
    from tools.decode.pack_decode_image import load_export_15m, pack

    image, params = pack(load_export_15m())
    assert params["VOCAB"] == 32000
    assert len(image) == 12124224
    assert hashlib.sha256(image).hexdigest() == "d0ac5863710ddec7f9b3cc2d9b526c1070fb7ecf3b07f656373dfdd0aca95541"


@pytest.mark.skipif(not (ROOT / "models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz").exists(), reason="model artifacts unavailable")
def test_pruned_export_keeps_all_classifier_row_metadata_aligned():
    from tools.decode.pack_decode_image import load_export_15m

    payload = load_export_15m()
    kept = [0, 1, 2, 3, 4, 258, 29889, 29892]
    reduced = prune_export(payload, kept)
    for key in ("weight", "multiplier", "shift", "weight_scale_multiplier", "weight_scale_shift",
                "embedding_scale_multiplier", "embedding_scale_shift"):
        assert np.array_equal(reduced["classifier"][key], np.asarray(payload["classifier"][key])[kept])
    assert np.array_equal(reduced["embedding_group_multiplier"], payload["embedding_group_multiplier"][kept])
    assert np.array_equal(reduced["embedding_group_shift"], payload["embedding_group_shift"][kept])
