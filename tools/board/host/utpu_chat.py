"""Interactive tokenizer/generation client for the Boolean board UART protocol."""
from __future__ import annotations

import argparse
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
    def __init__(self, transport, timeout: float = 60.0):
        self.transport, self.timeout = transport, timeout
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
        if not 0 <= token < VOCAB:
            raise ValueError(f"token {token} outside 0..{VOCAB-1}")
        if not 0 <= pos < MAX_SEQ:
            raise ValueError(f"position {pos} outside 0..{MAX_SEQ-1}")
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


def generate(client: Client, tokenizer: Tokenizer, prompt: str, gen: int, verify: bool = False,
             golden_runtime=None, emit=print):
    prompt_tokens = tokenizer.encode(prompt, bos=True)
    if not prompt_tokens:
        raise ValueError("prompt encoded to no tokens")
    if gen < 1 or len(prompt_tokens) + gen - 1 > MAX_SEQ:
        raise ValueError(f"prompt tokens plus generation require {len(prompt_tokens) + gen - 1} positions; valid range is 1..{MAX_SEQ}")
    client.ping()
    client.new_sequence()
    generated, token, previous, durations, cycles_sum = [], prompt_tokens[0], prompt_tokens[-1], [], 0
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
            piece = tokenizer.decode_piece(previous, out)
            emit(piece.decode("utf-8", errors="replace"), end="", flush=True)
            previous = out
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
    ap.add_argument("--quad-enable", action="store_true")
    ap.add_argument("--lc", type=lambda x: int(x, 2), default=None, help="explicit LC override (2 bits); must match the board build version")
    a = ap.parse_args(argv)
    if a.port is None:
        ap.error("--port is required by the command line client; tests construct LoopbackTransport directly")
    transport = SerialTransport(a.port, a.baud, a.timeout)
    try:
        client = Client(transport, a.timeout)
        if a.quad_enable:
            print(f"CR1 after quad enable: {client.quad_enable(a.lc):#04x}")
        runtime = load_golden_runtime() if a.verify else None
        generate(client, load_tokenizer(), a.prompt, a.gen, a.verify, runtime)
    except (BoardError, BuildVersionError, TimeoutError, ValueError, RuntimeError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
