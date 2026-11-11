from pathlib import Path

import pytest

from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from tools.llama2c.verify import verify

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models/llama2c"


def test_tokenizer_roundtrip_ascii():
    tok = Tokenizer(MODELS / "stories260K/tok512.bin", 512)
    text = "Once upon a time, a fox ran home."
    ids = tok.encode(text)
    assert tok.decode(ids[1:]) == text


def test_checkpoint_shapes():
    model = Llama2CModel(MODELS / "stories260K/stories260K.bin")
    assert model.wq.shape == (model.config.n_layers, model.config.dim, model.config.dim)
    assert model.forward(1, 0).shape == (512,)


@pytest.mark.parametrize("checkpoint", ["stories260K/stories260K.bin", "stories15M.bin"])
def test_64_token_pytorch_parity(checkpoint):
    assert verify(str(MODELS / checkpoint))["passed"]
