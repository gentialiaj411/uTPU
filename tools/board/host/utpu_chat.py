"""Interactive tokenizer/generation client for the Boolean board UART protocol."""
from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "decode"))
from llama2c.tokenizer import Tokenizer

VOCAB, MAX_SEQ = 32000, 128
ERRORS = {
    1: "token is outside the model vocabulary",
    2: "position is outside MAX_SEQ",
    3: "position skips the next expected sequence position",
    4: "decode watchdog expired; the board reset the decode engine and flash reader",
    5: "SPI passthrough requested more than 8 transmit or receive bytes",
}


class BoardError(RuntimeError):
    def __init__(self, code: int, next_pos: int):
        self.code, self.next_pos = code, next_pos
        super().__init__(f"E{code:02d}: {ERRORS.get(code, 'unknown board error')} (next position {next_pos})")


# 'V' reply version byte -> CR1[7:6] latency code the build's qspi_reader needs (1 = SDR build 6Bh/LC=00, 2 = DDR build EDh/LC=11).
BUILD_LC = {1: 0b00, 2: 0b11}


class BuildVersionError(RuntimeError):
    pass


class PrunedVocabularyError(RuntimeError):
    """The compact vocabulary and the image metadata do not describe one build."""


def load_idmap(path: str | Path) -> dict:
    """Load and validate a vocabulary-pruning id map before touching the board."""
    path = Path(path)
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrunedVocabularyError(f"cannot read vocabulary id map {path}: {exc}") from exc
    required = ("schema_version", "original_vocab", "keep_size", "kept_original_ids",
                "original_to_compact", "special_ids", "byte_fallback_ids")
    if any(key not in item for key in required):
        raise PrunedVocabularyError(f"id map {path} is missing one of {required}")
    if item["schema_version"] != 1:
        raise PrunedVocabularyError(f"unsupported id-map schema {item['schema_version']}")
    original_vocab, keep = int(item["original_vocab"]), int(item["keep_size"])
    kept, inverse = item["kept_original_ids"], item["original_to_compact"]
    if len(kept) != keep or len(inverse) != original_vocab:
        raise PrunedVocabularyError(f"id map has inconsistent dimensions (K={keep}, original={original_vocab})")
    if len(set(kept)) != keep or any(not 0 <= int(x) < original_vocab for x in kept):
        raise PrunedVocabularyError("id map kept_original_ids is not a unique in-range list")
    for compact, original in enumerate(kept):
        if int(inverse[int(original)]) != compact:
            raise PrunedVocabularyError("id map forward and inverse mappings disagree")
    if [int(x) for x in item["special_ids"]] != [0, 1, 2] or [int(x) for x in item["byte_fallback_ids"]] != list(range(3, 259)):
        raise PrunedVocabularyError("id map does not contain the required special and byte-fallback prefix")
    tokenizer_sha = item.get("tokenizer_sha256", "")
    tokenizer_path = ROOT / "models/llama2c/tokenizer.bin"
    if tokenizer_sha and tokenizer_path.exists():
        actual = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        if actual != tokenizer_sha:
            raise PrunedVocabularyError("id map tokenizer_sha256 does not match models/llama2c/tokenizer.bin")
    return item


def validate_pruned_build(idmap: dict, params: dict | None = None,
                          image: str | Path | None = None) -> None:
    """Check the locally available image metadata for the selected id map.

    The UART protocol has no command that reports the flashed model image.  When an
    image or params sidecar is supplied, these checks therefore happen before UART
    traffic; an advertised image hash is checked against the image bytes as well.
    """
    if params is None and image is None:
        return
    if params is None:
        image = Path(image)
        candidates = [image.with_suffix(".params.json"), Path(str(image) + ".params.json")]
        sidecar = next((p for p in candidates if p.exists()), None)
        if sidecar is None:
            raise PrunedVocabularyError(f"no params.json sidecar found for image {image}")
        try:
            params = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PrunedVocabularyError(f"cannot read image params {sidecar}: {exc}") from exc
    expected_vocab = int(idmap["keep_size"])
    if int(params.get("VOCAB", -1)) != expected_vocab:
        raise PrunedVocabularyError(f"image params VOCAB={params.get('VOCAB')}, but id map K={expected_vocab}")
    if image is not None:
        image = Path(image)
        if not image.exists():
            raise PrunedVocabularyError(f"image does not exist: {image}")
        if "IMAGE_BYTES" in params and image.stat().st_size != int(params["IMAGE_BYTES"]):
            raise PrunedVocabularyError("image size does not match its params.json IMAGE_BYTES")
        image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
        advertised = params.get("image_sha256") or params.get("IMAGE_SHA256") or idmap.get("image_sha256")
        if advertised and image_sha != advertised:
            raise PrunedVocabularyError("image sha256 does not match the id map/params identity")


def encode_pruned(tokenizer: Tokenizer, text: str, idmap: dict) -> tuple[list[int], list[int]]:
    """Encode with llama2.c, replacing every dropped token by its UTF-8 byte tokens."""
    inverse = idmap["original_to_compact"]
    compact, expanded = [], []
    for token in tokenizer.encode(text, bos=True):
        compact_id = int(inverse[token])
        if compact_id >= 0:
            compact.append(compact_id); expanded.append(token)
        else:
            for byte in tokenizer.vocab[token]:
                original = int(byte) + 3
                mapped = int(inverse[original])
                if mapped < 0:
                    raise PrunedVocabularyError(f"id map dropped required byte fallback row {original}")
                compact.append(mapped); expanded.append(original)
    return compact, expanded


def decode_pruned_piece(tokenizer: Tokenizer, previous_original: int, compact: int, idmap: dict) -> tuple[str, int]:
    original = int(idmap["kept_original_ids"][compact])
    return tokenizer.decode_piece(previous_original, original).decode("utf-8", errors="replace"), original


def decode_pruned(tokenizer: Tokenizer, compact_ids: list[int], idmap: dict) -> str:
    """Decode compact ids, including adjacent byte-fallback tokens, as UTF-8 text."""
    original = [int(idmap["kept_original_ids"][int(compact)]) for compact in compact_ids]
    if original and original[0] == 1:       # BOS is a protocol input, not prompt text.
        original = original[1:]
        # llama2.c inserts a leading space token before non-empty input.  The
        # tokenizer's byte-fallback spelling is <0x20>, so decode_piece's
        # string-prefix shortcut does not remove it for this tokenizer.
        if original and tokenizer.decode_piece(1, original[0]) == b" ":
            original = original[1:]
    out, previous = bytearray(), 1
    for token in original:
        out.extend(tokenizer.decode_piece(previous, token))
        previous = token
    return bytes(out).decode("utf-8", errors="replace")


def build_lc(version: int) -> int:
    if version not in BUILD_LC:
        raise BuildVersionError(f"board reported build version {version:#04x}; known versions are {sorted(BUILD_LC)} "
                                f"(1 = SDR build, LC=00; 2 = DDR build, LC=11). Refusing to configure the flash")
    return BUILD_LC[version]


class SerialTransport:
    def __init__(self, port: str, baud: int, timeout: float):
        try:
            import serial
        except ImportError as e:
            raise RuntimeError("Real board transport requires optional dependency pyserial (pip install pyserial)") from e
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout, write_timeout=timeout)

    def write(self, data: bytes) -> None:
        self.ser.write(data)

    def read(self, n: int) -> bytes:
        return self.ser.read(n)

    def close(self) -> None:
        self.ser.close()


class LoopbackTransport:
    """Small board-side protocol model. Its byte parsing is deliberately independent of Client."""
    def __init__(self, runtime=None, vocab=VOCAB, max_seq=MAX_SEQ, watchdog=False, version=1, cr1=0):
        self.runtime, self.vocab, self.max_seq = runtime, vocab, max_seq
        self.watchdog, self.next_pos, self.reply = watchdog, 0, bytearray()
        self.sr1, self.cr1, self.wel, self.wip = 0, cr1, False, 0
        self.version = version
        self.frames = []

    def write(self, frame: bytes) -> None:
        if not frame:
            return
        self.frames.append(bytes(frame))
        command = frame[0]
        if command == 0x56:
            self.reply.extend((0x76, self.version))
        elif command == 0x58:
            self.next_pos = 0
            if self.runtime is not None:
                self.runtime.reset()
            self.reply.extend((0x78, 1))
        elif command == 0x54:
            if len(frame) != 6:
                raise ValueError("loopback T frame must be six bytes")
            # Board-side spec parse, written separately from Client.step serialization.
            token = frame[1] + 256 * frame[2]
            pos = frame[3]
            probe = frame[4] + 256 * frame[5]
            code = 1 if token >= self.vocab else 2 if pos >= self.max_seq else 3 if pos > self.next_pos else 0
            if self.watchdog:
                code = 4
            if code:
                self.reply.extend((0x45, code, self.next_pos))
            else:
                if self.runtime is None:
                    raise RuntimeError("LoopbackTransport needs a golden runtime for T frames")
                logits = self.runtime.forward(token, pos)
                result = int(logits.argmax())
                selected = int(logits[probe]) if probe < len(logits) else 0
                if pos == self.next_pos:
                    self.next_pos += 1
                # Board R response: uint16 token, uint32 cycle counter, int32 selected logit, LE.
                self.reply.extend((0x52, result & 255, result >> 8))
                self.reply.extend(bytes(4))
                self.reply.extend(struct.pack("<i", selected))
        elif command == 0x53:
            if len(frame) < 3:
                raise ValueError("loopback S header must contain ntx and nrx")
            ntx, nrx = frame[1], frame[2]
            if ntx > 8 or nrx > 8:
                self.reply.extend((0x45, 5, self.next_pos))
            else:
                tx = frame[3:3 + ntx]
                if len(tx) != ntx:
                    raise ValueError("loopback S frame ended before all transmit bytes")
                if tx == b"\x05":
                    rx = bytes((self.sr1 | (int(self.wel) << 1) | int(self.wip > 0),))
                    self.wip = max(0, self.wip - 1)
                elif tx == b"\x35":
                    rx = bytes((self.cr1,))
                elif tx == b"\x9f":
                    rx = bytes((1, 0x20, 0x18, 0x4d, 0, 0x80))
                elif tx == b"\x06":
                    self.wel = True
                    rx = b""
                elif tx[:1] == b"\x01" and len(tx) >= 3 and self.wel:
                    self.sr1, self.cr1, self.wel, self.wip = tx[1] & 0xFC, tx[2], False, 2
                    rx = b""
                else:
                    rx = bytes(nrx)
                self.reply.extend((0x73, nrx))
                self.reply.extend(rx[:nrx].ljust(nrx, b"\x00"))

    def read(self, n: int) -> bytes:
        data = bytes(self.reply[:n])
        del self.reply[:n]
        return data


class Client:
    """Host-side frame encoder/parser. Integers in the UART frames are little-endian."""
    def __init__(self, transport, timeout: float = 60.0, vocab: int = VOCAB, max_seq: int = MAX_SEQ):
        self.transport, self.timeout, self.vocab, self.max_seq = transport, timeout, vocab, max_seq
        self.version = None                  # build version from the last ping()

    def _read_exact(self, n: int, what: str) -> bytes:
        data = self.transport.read(n)
        if len(data) != n:
            raise TimeoutError(f"timeout waiting for {what}: received {len(data)} of {n} bytes")
        return data

    def _reply(self, tag: int, n: int, what: str) -> bytes:
        first = self._read_exact(1, what)[0]
        if first == ord("E"):
            code, nxt = self._read_exact(2, f"{what} error frame")
            raise BoardError(code, nxt)
        if first != tag:
            raise RuntimeError(f"{what}: reply tag 0x{first:02x}, expected 0x{tag:02x}")
        return self._read_exact(n, what)

    def ping(self) -> int:
        """'V': returns the build version (1 = SDR build, LC=00; 2 = DDR build, LC=11); any other value is a BuildVersionError."""
        self.transport.write(b"V")
        version = self._reply(ord("v"), 1, "link check")[0]
        build_lc(version)
        self.version = version
        return version

    def new_sequence(self) -> None:
        self.transport.write(b"X")
        if self._reply(ord("x"), 1, "new sequence") != b"\x01":
            raise RuntimeError("new sequence returned an invalid acknowledgement")

    def spi(self, tx: bytes, nrx: int) -> bytes:
        if not 1 <= len(tx) <= 8 or not 0 <= nrx <= 8:
            raise ValueError("SPI passthrough limits are 1..8 transmit and 0..8 receive bytes")
        # Independent of the board-side decoder: S, ntx, nrx, tx bytes.
        self.transport.write(bytes((ord("S"), len(tx), nrx)) + tx)
        count = self._reply(ord("s"), 1, "SPI passthrough")[0]
        if count != nrx:
            raise RuntimeError(f"SPI reply count {count}, expected {nrx}")
        return self._read_exact(nrx, "SPI data")

    def quad_enable(self, lc=None) -> int:
        # Match Board.enable_quad and run_uart_rtl.host_sequence: read SR1/CR1,
        # WREN, WRR with SR1 bits 1:0 cleared and CR1 LC=<build>/QUAD=1, poll, read back.
        # LC comes from the 'V' build version (00 for version 1, 11 for version 2); `lc` is an explicit override that must agree.
        version = self.version if self.version is not None else self.ping()
        build = build_lc(version)
        if lc is not None and lc != build:
            raise RuntimeError(f"refusing to set LC={lc:02b}: board build version {version} needs LC={build:02b}")
        sr = self.spi(b"\x05", 1)[0]
        cr = self.spi(b"\x35", 1)[0]
        want = (cr & 0x3F) | 0x02 | (build << 6)
        if cr == want and not sr & 0x61:
            return cr
        self.spi(b"\x06", 0)
        if not self.spi(b"\x05", 1)[0] & 0x02:
            raise RuntimeError("quad enable failed: WEL did not set after WREN")
        self.spi(bytes((0x01, sr & 0xFC, want)), 0)
        deadline = time.monotonic() + 5.0
        while self.spi(b"\x05", 1)[0] & 1:
            if time.monotonic() > deadline:
                raise TimeoutError("quad enable timed out while polling SR1.WIP")
        sr2, cr2 = self.spi(b"\x05", 1)[0], self.spi(b"\x35", 1)[0]
        if cr2 != want or sr2 & 0x63:
            raise RuntimeError(f"quad enable readback failed: SR1={sr2:#04x} CR1={cr2:#04x}, expected CR1={want:#04x}")
        return cr2

    def step(self, token: int, pos: int, probe: int = 0) -> tuple[int, int, int]:
        if not 0 <= token < self.vocab:
            raise ValueError(f"token {token} outside 0..{self.vocab-1}")
        if not 0 <= pos < self.max_seq:
            raise ValueError(f"position {pos} outside 0..{self.max_seq-1}")
        # T, token low/high, position byte, probe index low/high.
        self.transport.write(bytes((ord("T"), token & 255, token >> 8, pos, probe & 255, probe >> 8)))
        return struct.unpack("<HIi", self._reply(ord("R"), 10, "decode step"))


def load_tokenizer() -> Tokenizer:
    """The 32000-token llama2.c tokenizer of stories15M, the model the boolean_core bitstream is built for."""
    path = ROOT / "models/llama2c/tokenizer.bin"
    if not path.exists():
        raise FileNotFoundError(f"stories15M tokenizer not found at {path}")
    return Tokenizer(path, VOCAB)


def load_golden_runtime():
    """Golden for the board: the stories15M export with KV4, exactly as run_decode_rtl.py --model 15m --kv-bits 4."""
    from run_decode_rtl import load_model
    return load_model("15m", 4)[1]


def load_pruned_golden_runtime(idmap: dict):
    """The study's integer runtime, with the same rows pruned as the image."""
    from tools.decode.pack_decode_image import load_export_15m
    from tools.vocabprune.core import prune_export
    from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
    return RunQIntegerLlama(prune_export(load_export_15m(), idmap["kept_original_ids"]),
                            ROOT / "firmware/host/numerics_v2/luts", kv_bits=4)


def generate(client: Client, tokenizer: Tokenizer, prompt: str, gen: int, verify: bool = False,
             golden_runtime=None, emit=print, idmap=None):
    if idmap is None:
        prompt_tokens = tokenizer.encode(prompt, bos=True)
        expanded_prompt = prompt_tokens
    else:
        prompt_tokens, expanded_prompt = encode_pruned(tokenizer, prompt, idmap)
    if not prompt_tokens:
        raise ValueError("prompt encoded to no tokens")
    if gen < 1 or len(prompt_tokens) + gen - 1 > MAX_SEQ:
        raise ValueError(f"prompt tokens plus generation require {len(prompt_tokens) + gen - 1} positions; valid range is 1..{MAX_SEQ}")
    client.ping()
    client.new_sequence()
    generated, token, previous, durations, cycles_sum = [], prompt_tokens[0], expanded_prompt[-1], [], 0
    utf8_decoder = codecs.getincrementaldecoder("utf-8")("replace") if idmap is not None else None
    for pos in range(len(prompt_tokens) + gen - 1):
        expected_logits = golden_runtime.forward(token, pos) if verify else None
        expected_token = int(expected_logits.argmax()) if verify else None
        begin = time.perf_counter()                 # time the board step only, not the host-side golden
        out, cycles, chosen_logit = client.step(token, pos, expected_token if verify else 0)
        elapsed = time.perf_counter() - begin
        durations.append(elapsed); cycles_sum += cycles
        if verify:
            expected_logit = int(expected_logits[expected_token])
            if out != expected_token:
                raise AssertionError(f"token mismatch at position {pos}: board={out}, golden={expected_token}")
            if chosen_logit != expected_logit:
                raise AssertionError(f"logit mismatch at position {pos}, index {expected_token}: board={chosen_logit}, golden={expected_logit}")
        if pos + 1 < len(prompt_tokens):
            token = prompt_tokens[pos + 1]
        else:
            generated.append(out)
            if idmap is None:
                piece = tokenizer.decode_piece(previous, out).decode("utf-8", errors="replace")
                previous = out
            else:
                original = int(idmap["kept_original_ids"][out])
                piece = utf8_decoder.decode(tokenizer.decode_piece(previous, original), final=False)
                previous = original
            emit(piece, end="", flush=True)
            token = out
        emit(f"\n[token {pos}: {elapsed:.3f}s, {cycles} FPGA cycles]", end="", flush=True)
    board = sum(durations)                          # board step wall time (prompt + generated); host golden excluded
    rate = f", {50e6 * len(durations) / cycles_sum:.3f} steps/s from FPGA cycles at 50 MHz" if cycles_sum else ""
    emit(f"\n{len(durations)} board steps in {board:.3f}s: {len(durations) / board:.3f} steps/s wall{rate}; "
         f"FPGA cycles/step {cycles_sum / len(durations):.1f}")
    return generated


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", help="serial port, e.g. COM5 (omit for loopback)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--vocab-keep", type=Path, help="pruned-vocabulary id-map JSON")
    ap.add_argument("--image", type=Path, help="local image used for this board (enables params/hash identity checks)")
    ap.add_argument("--params", type=Path, help="image params.json (otherwise discovered beside --image)")
    ap.add_argument("--quad-enable", action="store_true")
    ap.add_argument("--lc", type=lambda x: int(x, 2), default=None, help="explicit LC override (2 bits); must match the board build version")
    a = ap.parse_args(argv)
    if a.port is None:
        ap.error("--port is required by the command line client; tests construct LoopbackTransport directly")
    transport = None
    try:
        idmap = load_idmap(a.vocab_keep) if a.vocab_keep else None
        if a.params and not a.params.exists():
            raise PrunedVocabularyError(f"image params do not exist: {a.params}")
        image = a.image
        if idmap is not None and image is None and a.params is None:
            candidate = ROOT / "build/vocabprune" / f"vocab_{idmap['keep_size']}.image.bin"
            if candidate.exists():
                image = candidate
            else:
                raise PrunedVocabularyError("pruned build identity is unknown; supply --image or --params")
        params = json.loads(a.params.read_text(encoding="utf-8")) if a.params else None
        if idmap is not None:
            validate_pruned_build(idmap, params, image)
        transport = SerialTransport(a.port, a.baud, a.timeout)
        client = Client(transport, a.timeout, vocab=int(idmap["keep_size"]) if idmap else VOCAB)
        if a.quad_enable:
            print(f"CR1 after quad enable: {client.quad_enable(a.lc):#04x}")
        runtime = (load_pruned_golden_runtime(idmap) if idmap is not None else load_golden_runtime()) if a.verify else None
        generate(client, load_tokenizer(), a.prompt, a.gen, a.verify, runtime, idmap=idmap)
    except (BoardError, BuildVersionError, PrunedVocabularyError, TimeoutError, ValueError, RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        if transport is not None:
            transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
