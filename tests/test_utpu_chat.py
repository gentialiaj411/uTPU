import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/board/host"))
import utpu_chat as chat


def _runtime():
    return chat.load_golden_runtime()


def test_golden_260k_eight_token_generation_matches_runq():
    tokenizer = chat.load_tokenizer()
    golden = _runtime()
    prompt = "Once upon a"
    prompt_tokens = tokenizer.encode(prompt, bos=True)
    golden.reset()
    expected, token = [], prompt_tokens[0]
    for pos in range(len(prompt_tokens) + 8 - 1):
        logits = golden.forward(token, pos)
        out = int(logits.argmax())
        if pos + 1 >= len(prompt_tokens):
            expected.append(out)
        token = prompt_tokens[pos + 1] if pos + 1 < len(prompt_tokens) else out

    board_runtime = _runtime()
    client = chat.Client(chat.LoopbackTransport(board_runtime))
    golden.reset()
    actual = chat.generate(client, tokenizer, prompt, 8, verify=True, golden_runtime=golden, emit=lambda *a, **k: None)
    assert len(actual) == 8
    assert actual == expected


@pytest.mark.parametrize("code,frame,watchdog", [
    (1, bytes((0x54, 0x00, 0x80, 0, 0, 0)), False),
    (2, bytes((0x54, 1, 0, 128, 0, 0)), False),
    (3, bytes((0x54, 1, 0, 1, 0, 0)), False),
    (4, bytes((0x54, 1, 0, 0, 0, 0)), True),
    (5, bytes((0x53, 9, 0)), False),
])
def test_each_board_error_code(code, frame, watchdog):
    board = chat.LoopbackTransport(runtime=None, watchdog=watchdog)   # error paths never run a forward pass
    client = chat.Client(board)
    if code == 3:
        board.write(b"X")
        assert client._reply(ord("x"), 1, "new sequence") == b"\x01"
    board.write(frame)
    with pytest.raises(chat.BoardError) as exc:
        client._reply(ord("R") if frame[0] == 0x54 else ord("s"), 10 if frame[0] == 0x54 else 1, "fault path")
    assert exc.value.code == code


def test_quad_enable_register_sequence_matches_uart_host_sequence():
    board = chat.LoopbackTransport()
    client = chat.Client(board)
    assert client.quad_enable() == 0x02
    assert [f for f in board.frames if f[0] == ord("S")] == [
        bytes((0x53, 1, 1, 0x05)), bytes((0x53, 1, 1, 0x35)),
        bytes((0x53, 1, 0, 0x06)), bytes((0x53, 1, 1, 0x05)),
        bytes((0x53, 3, 0, 0x01, 0x00, 0x02)), bytes((0x53, 1, 1, 0x05)),
        bytes((0x53, 1, 1, 0x05)), bytes((0x53, 1, 1, 0x05)),
        bytes((0x53, 1, 1, 0x05)), bytes((0x53, 1, 1, 0x35)),
    ]


S_FRAMES_V1 = [
    bytes((0x53, 1, 1, 0x05)), bytes((0x53, 1, 1, 0x35)),
    bytes((0x53, 1, 0, 0x06)), bytes((0x53, 1, 1, 0x05)),
    bytes((0x53, 3, 0, 0x01, 0x00, 0x02)), bytes((0x53, 1, 1, 0x05)),
    bytes((0x53, 1, 1, 0x05)), bytes((0x53, 1, 1, 0x05)),
    bytes((0x53, 1, 1, 0x05)), bytes((0x53, 1, 1, 0x35)),
]


def test_ping_returns_build_version():
    for v in (1, 2):
        client = chat.Client(chat.LoopbackTransport(version=v))
        assert client.ping() == v and client.version == v


def test_version2_ddr_build_writes_lc11():
    board = chat.LoopbackTransport(version=2)
    client = chat.Client(board)
    assert client.quad_enable() == 0xC2
    s_frames = [f for f in board.frames if f[0] == ord("S")]
    # same frame sequence as version 1 except the WRR payload: CR1 = {LC=11, QUAD=1}
    assert s_frames == [bytes((0x53, 3, 0, 0x01, 0x00, 0xC2)) if f[:4] == bytes((0x53, 3, 0, 0x01)) else f for f in S_FRAMES_V1]
    assert board.cr1 == 0xC2
    assert client.quad_enable() == 0xC2                 # already set: no second WRR
    assert sum(f[:4] == bytes((0x53, 3, 0, 0x01)) for f in board.frames) == 1


def test_version2_rewrites_quad_only_flash_with_lc00():
    board = chat.LoopbackTransport(version=2, cr1=0x02)  # an SDR-configured flash under a DDR bitstream
    assert chat.Client(board).quad_enable() == 0xC2


def test_version1_with_lc11_flash_clears_lc():
    board = chat.LoopbackTransport(version=1, cr1=0xC8)
    assert chat.Client(board).quad_enable() == 0x0A       # LC cleared, bit 3 kept, QUAD set


def test_version1_frame_sequence_unchanged_with_explicit_override():
    board = chat.LoopbackTransport(version=1)
    assert chat.Client(board).quad_enable(lc=0) == 0x02
    assert [f for f in board.frames if f[0] == ord("S")] == S_FRAMES_V1


def test_override_disagreeing_with_build_is_refused_before_any_write():
    for version, lc in ((1, 3), (2, 0)):
        board = chat.LoopbackTransport(version=version)
        with pytest.raises(RuntimeError, match="refusing"):
            chat.Client(board).quad_enable(lc=lc)
        assert not any(f[0] == ord("S") for f in board.frames)


@pytest.mark.parametrize("version", [0, 3, 0xFF])
def test_unknown_build_version_is_an_error(version):
    board = chat.LoopbackTransport(version=version)
    client = chat.Client(board)
    with pytest.raises(chat.BuildVersionError, match="build version"):
        client.ping()
    with pytest.raises(chat.BuildVersionError, match="build version"):
        chat.Client(chat.LoopbackTransport(version=version)).quad_enable()
    assert not any(f[0] == ord("S") for f in board.frames)


def test_t_frame_has_single_position_byte_then_little_endian_probe():
    # Position is one byte in RTL; the two following bytes are probe index low/high.
    # Checking the actual request is done on a recording transport to protect this boundary.
    class Recorder:
        def __init__(self): self.frame = b""
        def write(self, frame): self.frame = bytes(frame)
        def read(self, n): return b""
    recorder = Recorder()
    with pytest.raises(TimeoutError):
        chat.Client(recorder).step(0x1234, 7, 0x1234)
    assert recorder.frame == bytes((0x54, 0x34, 0x12, 7, 0x34, 0x12))



def test_client_model_matches_board_build():
    """The client's tokenizer and golden must be the model boolean_core is built for (a 260K golden against the 15M
    bitstream would make --verify fail on hardware while loopback tests, which share the runtime, still pass)."""
    import re
    src = (ROOT / "rtl/board/boolean_core.sv").read_text()
    board = {k: int(re.search(rf"\b{k} = (\d+)", src).group(1)) for k in ("D", "H", "NL", "NH", "NKV", "VOCAB", "MAX_SEQ", "KV4")}
    rt = chat.load_golden_runtime()                 # what --verify actually compares the board against
    cfg = rt.cfg
    assert (cfg["dim"], cfg["hidden_dim"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"], cfg["vocab_size"]) == (
        board["D"], board["H"], board["NL"], board["NH"], board["NKV"], board["VOCAB"])
    assert rt.kv_bits == (4 if board["KV4"] else 8)
    assert (chat.VOCAB, chat.MAX_SEQ) == (board["VOCAB"], board["MAX_SEQ"])
    assert len(chat.load_tokenizer().vocab) == board["VOCAB"]
