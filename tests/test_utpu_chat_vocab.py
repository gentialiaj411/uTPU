import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/board/host"))
import utpu_chat as chat


MAP = ROOT / "tools/vocabprune/idmaps/idmap_2048.json"


def test_idmap_load_and_validation():
    item = chat.load_idmap(MAP)
    assert item["keep_size"] == 2048
    assert item["original_to_compact"][item["kept_original_ids"][2047]] == 2047


@pytest.mark.parametrize("text", ["Hello, world!", "Résumé — café ☕🙂"])
def test_pruned_encode_decode_round_trip(text):
    tokenizer = chat.load_tokenizer()
    item = chat.load_idmap(MAP)
    compact, expanded = chat.encode_pruned(tokenizer, text, item)
    assert compact
    assert all(0 <= token < item["keep_size"] for token in compact)
    assert chat.decode_pruned(tokenizer, compact, item) == text
    assert any(original in range(3, 259) for original in expanded[1:])


def test_mismatched_image_params_are_refused(tmp_path):
    item = chat.load_idmap(MAP)
    params = {"VOCAB": item["keep_size"] + 16}
    with pytest.raises(chat.PrunedVocabularyError, match="VOCAB"):
        chat.validate_pruned_build(item, params)


def test_mismatched_image_identity_is_refused(tmp_path):
    item = chat.load_idmap(MAP)
    image = tmp_path / "image.bin"
    image.write_bytes(b"wrong image")
    params = {"VOCAB": item["keep_size"], "IMAGE_BYTES": image.stat().st_size,
              "image_sha256": hashlib.sha256(b"different image").hexdigest()}
    with pytest.raises(chat.PrunedVocabularyError, match="sha256"):
        chat.validate_pruned_build(item, params, image)


def test_image_sidecar_is_auto_discovered(tmp_path):
    item = chat.load_idmap(MAP)
    image = tmp_path / "vocab.image.bin"
    image.write_bytes(b"image")
    image.with_suffix(".params.json").write_text(json.dumps({
        "VOCAB": item["keep_size"], "IMAGE_BYTES": image.stat().st_size}), encoding="utf-8")
    chat.validate_pruned_build(item, image=image)


def test_unpruned_client_defaults_remain_unchanged():
    board = chat.LoopbackTransport()
    client = chat.Client(board)
    assert client.vocab == chat.VOCAB and client.max_seq == chat.MAX_SEQ
    with pytest.raises(ValueError, match="outside"):
        client.step(chat.VOCAB, 0)
