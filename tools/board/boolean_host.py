"""Host side of the Boolean UART protocol (rtl/board/board_ctrl.sv). Needs pyserial (pip install pyserial).

  python tools/board/boolean_host.py --port COM5 ping
  python tools/board/boolean_host.py --port COM5 info          # read-only: ID, SR1, CR1; compatible with the bitstream? (exit 1 if not)
  python tools/board/boolean_host.py --port COM5 id            # flash JEDEC ID + sector architecture
  python tools/board/boolean_host.py --port COM5 cr1           # read SR1 and CR1, report QUAD and LC
  python tools/board/boolean_host.py --port COM5 enable-quad   # set CR1 = {LC=00, QUAD=1}, other bits as read; verify SR1/CR1 by read-back
  python tools/board/boolean_host.py --port COM5 step --token 1 --pos 0 --probe 5
  python tools/board/boolean_host.py --port COM5 run --prompt 1,403,260 --gen 8
  python tools/board/boolean_host.py selftest                  # frame encoding / error handling vs a software stub (no board needed)

The frame layouts are the ones exercised by tools/board/run_uart_rtl.py against the RTL. Nothing here has talked to a real board.

Protocol additions (board_ctrl.sv): 'X' starts a new sequence; a rejected request is answered by an error frame 'E' code next
(codes in ERRORS below). Bounds are checked here before anything is sent; the board checks them again.
"""
from __future__ import annotations
import argparse, struct, sys, time

VOCAB, MAX_SEQ = 32000, 128          # stories15M build (boolean_core.sv defaults); override with --vocab / --max-seq
ERRORS = {1: 'token >= VOCAB', 2: 'pos >= MAX_SEQ', 3: 'pos > next expected position (positions must run 0,1,2,... in order)',
          4: 'step watchdog: the decode engine did not finish; engine and flash reader were reset', 5: 'S frame with ntx or nrx > 8'}

# S25FL128S register facts (datasheet rev 05, 12 Jul 2012, as recorded in docs/r3_progress.md M4):
#   CR1 (35h): [7:6] LC latency code, [5] TBPROT, [4] reserved/LOCK per table, [3] BPNV, [1] QUAD, [0] FREEZE. QUAD is non-volatile.
#   Quad Output Read 6Bh needs 8 dummy clocks at LC = 00, 01, 10 (up to 104 MHz) and 0 only at LC = 11 (tables 8.10 / 8.12 of that
#   datasheet; the table text came from a PDF extraction with scrambled columns and was cross-checked against Fig. 10.30, not read
#   graphically). qspi_reader.DUMMY = 8, so the host forces the default LC = 00 explicitly rather than accepting whatever is stored.
#   SR1 (05h): [7] SRWD [6] P_ERR [5] E_ERR [4:2] BP [1] WEL [0] WIP.
CR1_LC_MASK, CR1_QUAD, CR1_LC_WANT = 0xC0, 0x02, 0x00
SR1_WIP, SR1_WEL, SR1_ERR = 0x01, 0x02, 0x60
ID_PREFIX = bytes([0x01, 0x20, 0x18])     # Spansion/Cypress/Infineon, S25FL128S
BITSTREAM_BYTES, FLASH_BASE = 2192012, 0x280000   # xc7s50 bitstream 17,536,096 bits; weight image offset (boolean_core.FLASH_BASE)


class BoardError(RuntimeError):
    def __init__(self, code, next_pos):
        self.code, self.next_pos = code, next_pos
        super().__init__(f'board rejected the request: E{code:02d} {ERRORS.get(code, "unknown")} (next expected position {next_pos})')


class FlashConfigError(RuntimeError):
    pass


class Board:
    def __init__(self, ser, vocab=VOCAB, max_seq=MAX_SEQ):
        self.ser, self.vocab, self.max_seq = ser, vocab, max_seq

    def _rx(self, n, what):
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError(f'{what}: expected {n} bytes, got {len(data)} ({data.hex()})')
        return data

    def _reply(self, tag, n, what):
        """Read the first byte; an 'E' frame becomes BoardError, anything else must be the expected tag. Returns the n bytes after the tag."""
        t = self._rx(1, what)[0]
        if t == 0x45:
            code, nxt = self._rx(2, what + ' error frame')
            raise BoardError(code, nxt)
        if t != tag:
            raise RuntimeError(f'{what}: unexpected reply tag {t:#04x} (wanted {tag:#04x})')
        return self._rx(n, what)

    def ping(self):
        self.ser.write(b'V')
        r = self._reply(0x76, 1, 'ping')
        assert r == b'\x01', r
        return r[0]

    def new_sequence(self):
        """'X': the board expects position 0 next."""
        self.ser.write(b'X')
        assert self._reply(0x78, 1, 'new sequence') == b'\x01'

    def spi(self, tx: bytes, nrx: int) -> bytes:
        """Single-bit SPI transaction on the flash: CS# low, tx out on IO0, nrx bytes in on IO1, CS# high (<= 8 each)."""
        if not (1 <= len(tx) <= 8 and 0 <= nrx <= 8):
            raise ValueError(f'spi: ntx {len(tx)} / nrx {nrx} out of range (1..8 / 0..8)')
        self.ser.write(bytes([0x53, len(tx), nrx]) + tx)
        r = self._reply(0x73, 1, 'spi')
        if r[0] != nrx:
            raise RuntimeError(f'spi: board returned {r[0]} bytes, asked for {nrx}')
        return self._rx(nrx, 'spi data')

    def flash_id(self):
        return self.spi(b'\x9f', 6)               # 01 20 18 4D <00 uniform 256 kB | 01 hybrid 64 kB> 80

    def sr1(self):
        return self.spi(b'\x05', 1)[0]

    def cr1(self):
        return self.spi(b'\x35', 1)[0]

    def bar(self):
        """Bank Address Register (16h, BRRD). Optional (`info --bar`): not exercised against the flash model or a datasheet table in this repo."""
        return self.spi(b'\x16', 1)[0]

    def info(self, with_bar=False):
        """Read-only report. 'compatible' = the flash state the bitstream's reader (6Bh, 3-byte address, 8 dummy clocks, quad) needs."""
        self.ping()
        fid, sr, cr = self.flash_id(), self.sr1(), self.cr1()
        problems = []
        if fid[:3] != ID_PREFIX:
            problems.append(f'flash ID {fid[:3].hex()} is not S25FL128S ({ID_PREFIX.hex()})')
        if not cr & CR1_QUAD:
            problems.append('CR1.QUAD = 0 (run enable-quad)')
        if cr & CR1_LC_MASK != CR1_LC_WANT:
            problems.append(f'CR1 latency code LC = {(cr >> 6) & 3:02b}, the reader assumes LC = 00 / 8 dummy clocks (run enable-quad)')
        if sr & SR1_WIP:
            problems.append('SR1.WIP = 1 (a write/erase is still running)')
        if sr & SR1_ERR:
            problems.append(f'SR1 error bits set (P_ERR/E_ERR): SR1 = {sr:#04x}')
        rec = {'flash_id': fid.hex(), 'sector_architecture': {0: 'uniform 256 kB', 1: 'hybrid 64 kB'}.get(fid[4], f'unknown({fid[4]:#04x})'),
               'sr1': sr, 'cr1': cr, 'quad': (cr >> 1) & 1, 'lc': (cr >> 6) & 3, 'block_protect': (sr >> 2) & 7, 'srwd': sr >> 7,
               'bitstream_bytes': BITSTREAM_BYTES, 'weight_image_offset': FLASH_BASE,
               'image_offset_clear_of_bitstream': FLASH_BASE >= BITSTREAM_BYTES}
        if with_bar:
            b = self.bar(); rec['bar'] = b
            if b & 0x80 and b != 0xFF:
                problems.append(f'BAR.EXTADD set ({b:#04x}): 3-byte address commands would be interpreted as 4-byte')
        rec['problems'] = problems
        rec['compatible'] = not problems
        return rec

    def enable_quad(self, timeout=5.0):
        """Set CR1 = {LC = 00, other bits as read, QUAD = 1} with WREN + WRR and verify SR1/CR1 by read-back. Raises FlashConfigError on any mismatch."""
        sr, cr = self.sr1(), self.cr1()
        want = (cr & ~CR1_LC_MASK & 0xFF) | CR1_QUAD | CR1_LC_WANT
        if cr == want and not sr & (SR1_WIP | SR1_ERR):
            return 'already set', cr
        self.spi(b'\x06', 0)                        # WREN
        if not self.sr1() & SR1_WEL:
            raise FlashConfigError('WEL did not set after WREN: flash not responding on the register path')
        self.spi(bytes([0x01, sr & 0xFC, want]), 0)  # WRR: SR1 as read (WEL/WIP dropped), CR1 = want (OTP bits are written back as read)
        t0 = time.time()
        while self.sr1() & SR1_WIP:
            if time.time() - t0 > timeout:
                raise FlashConfigError('WRR did not finish (SR1.WIP stuck)')
        sr2, cr2 = self.sr1(), self.cr1()
        if cr2 != want:
            raise FlashConfigError(f'CR1 read back {cr2:#04x}, expected {want:#04x} (QUAD/LC not as written; check WP# / SR1.SRWD and OTP bits)')
        if sr2 & (SR1_WIP | SR1_WEL | SR1_ERR):
            raise FlashConfigError(f'SR1 = {sr2:#04x} after WRR: WIP/WEL/P_ERR/E_ERR must be clear')
        if (sr2 & 0xFC) != (sr & 0xFC & ~SR1_ERR):
            raise FlashConfigError(f'SR1 changed by the write: {sr:#04x} -> {sr2:#04x}')
        return 'set', cr2

    def step(self, token: int, pos: int, probe: int = 0):
        """One decode step -> (argmax token, core cycles, INT32 logit at index probe). Raises BoardError on an E frame."""
        if not 0 <= token < self.vocab:
            raise ValueError(f'token {token} outside 0..{self.vocab - 1}')
        if not 0 <= pos < self.max_seq:
            raise ValueError(f'pos {pos} outside 0..{self.max_seq - 1}')
        if not 0 <= probe < 65536:
            raise ValueError(f'probe {probe} outside 0..65535')
        self.ser.write(bytes([0x54, token & 255, token >> 8, pos, probe & 255, probe >> 8]))
        r = self._reply(0x52, 10, 'step')
        tok, cyc, logit = struct.unpack('<HIi', r)
        return tok, cyc, logit

    def run(self, prompt, gen):
        """Greedy decode: feed the prompt, then the board's own argmax; returns the generated token ids. Positions 0..len+gen-2 in order."""
        if len(prompt) + gen - 1 > self.max_seq:
            raise ValueError(f'prompt {len(prompt)} + gen {gen} needs {len(prompt) + gen - 1} positions, MAX_SEQ is {self.max_seq}')
        self.new_sequence()
        out, tok = [], prompt[0]
        for p in range(len(prompt) + gen - 1):
            nxt, cyc, _ = self.step(tok, p)
            if p + 1 < len(prompt):
                tok = prompt[p + 1]
            else:
                out.append(nxt); tok = nxt
        return out


class Stub:
    """Software stand-in for the board used by selftest and by run_uart_rtl.py: answers V, X, S and T frames like board_ctrl.sv
    (and the S25FL128S register model in tools/decode/tb/qspi_flash_model.sv), with the same checks on T."""
    def __init__(self, cr1=0x00, sr1=0x00, vocab=VOCAB, max_seq=MAX_SEQ):
        self.cr1, self.sr1, self.wel, self.wip, self.buf, self.log = cr1, sr1, 0, 0, b'', []
        self.vocab, self.max_seq, self.next_pos, self.n_wrr = vocab, max_seq, 0, 0
    def write(self, b):
        b = bytes(b); self.log.append(b)
        if b[0:1] == b'V':
            self.buf += b'\x76\x01'
        elif b[0:1] == b'X':
            self.next_pos = 0; self.buf += b'\x78\x01'
        elif b[0:1] == b'S':
            ntx, nrx, tx = b[1], b[2], b[3:3 + b[1]]
            if ntx > 8 or nrx > 8:
                self.buf += bytes([0x45, 5, self.next_pos]); return
            rx = b''
            if tx[:1] == b'\x9f': rx = bytes([1, 0x20, 0x18, 0x4D, 0x00, 0x80])[:nrx]
            elif tx[:1] == b'\x05':
                rx = bytes([self.sr1 | (self.wel << 1) | (1 if self.wip else 0)]); self.wip = max(0, self.wip - 1)
            elif tx[:1] == b'\x35': rx = bytes([self.cr1])
            elif tx[:1] == b'\x06': self.wel = 1
            elif tx[:1] == b'\x01' and self.wel:
                self.sr1, self.cr1, self.wel, self.wip = tx[1] & 0xFC, tx[2], 0, 2; self.n_wrr += 1
            self.buf += b'\x73' + bytes([nrx]) + rx
        elif b[0:1] == b'T':
            tok, pos = b[1] | (b[2] << 8), b[3]
            code = 1 if tok >= self.vocab else 2 if pos >= self.max_seq else 3 if pos > self.next_pos else 0
            if code:
                self.buf += bytes([0x45, code, self.next_pos])
            else:
                if pos == self.next_pos: self.next_pos += 1
                self.buf += b'R' + struct.pack('<HIi', 147, 930674, -5)
    def read(self, n):
        d, self.buf = self.buf[:n], self.buf[n:]
        return d


def selftest():
    s = Stub(); b = Board(s)
    assert b.ping() == 1
    assert b.flash_id() == bytes([1, 0x20, 0x18, 0x4D, 0x00, 0x80])
    assert b.cr1() == 0
    i = b.info(); assert not i['compatible'] and len(i['problems']) == 1 and 'QUAD' in i['problems'][0], i
    state, cr = b.enable_quad()
    assert state == 'set' and cr == 0x02, (state, cr)
    assert b.enable_quad()[0] == 'already set'
    assert b.info()['compatible']
    assert [l for l in s.log if l[:1] == b'S' and l[3:4] == b'\x01'][0] == bytes([0x53, 3, 0, 0x01, 0x00, 0x02]), 'WRR frame'
    assert b.step(1, 0, 0x0105) == (147, 930674, -5)
    assert s.log[-1] == bytes([0x54, 1, 0, 0, 5, 1]), s.log[-1]
    # latency code and other CR1 bits: LC is forced to 00, the rest is written back as read
    s = Stub(cr1=0xC8); b = Board(s)
    i = b.info(); assert not i['compatible'] and any('latency' in p for p in i['problems']) and any('QUAD' in p for p in i['problems']), i
    assert b.enable_quad() == ('set', 0x0A), 'LC=11 cleared, BPNV (bit 3) kept, QUAD set'
    assert [l for l in s.log if l[:1] == b'S' and l[3:4] == b'\x01'][0] == bytes([0x53, 3, 0, 0x01, 0x00, 0x0A])
    s = Stub(cr1=0x42); b = Board(s)          # QUAD already set but LC = 01: must still be rewritten
    assert not b.info()['compatible'] and b.enable_quad() == ('set', 0x02)
    # read-back verification: a flash that ignores WRR must be reported
    class Deaf(Stub):
        def write(self, b):
            if bytes(b)[3:4] == b'\x01': self.log.append(bytes(b)); self.buf += b'\x73\x00'; return
            super().write(b)
    b = Board(Deaf())
    try: b.enable_quad(timeout=0.2); raise SystemExit('deaf flash not detected')
    except FlashConfigError as e: assert 'CR1 read back' in str(e) or 'WEL' in str(e), e
    # bounds client-side, error frames, sequence
    s = Stub(); b = Board(s)
    for bad in ((VOCAB, 0), (-1, 0), (1, MAX_SEQ), (1, -1)):
        try: b.step(*bad); raise SystemExit(f'accepted {bad}')
        except ValueError: pass
    assert not any(l[:1] == b'T' for l in s.log), 'a rejected request must not be sent'
    try: b.spi(b'\x9f' * 9, 1); raise SystemExit('ntx 9 accepted')
    except ValueError: pass
    try: b.spi(b'\x9f', 9); raise SystemExit('nrx 9 accepted')
    except ValueError: pass
    s.write(bytes([0x54, 1, 0, 5, 0, 0]))     # the board's own check, bypassing the client: pos 5 with next 0
    try: b._reply(0x52, 10, 'step'); raise SystemExit('E frame not raised')
    except BoardError as e: assert e.code == 3 and e.next_pos == 0, e
    assert b.run([1, 2, 3], 3) == [147] * 3 and s.next_pos == 5 and s.log[-2][3] == 3 and s.log[-1][3] == 4
    try: b.run([1], MAX_SEQ + 1); raise SystemExit('overlong run accepted')
    except ValueError: pass
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port'); ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('cmd', choices=['ping', 'info', 'id', 'cr1', 'enable-quad', 'step', 'run', 'selftest'])
    ap.add_argument('--token', type=int, default=1); ap.add_argument('--pos', type=int, default=0); ap.add_argument('--probe', type=int, default=0)
    ap.add_argument('--prompt', default='1'); ap.add_argument('--gen', type=int, default=8)
    ap.add_argument('--vocab', type=int, default=VOCAB); ap.add_argument('--max-seq', type=int, default=MAX_SEQ)
    ap.add_argument('--bar', action='store_true', help='info: also read the bank address register (16h; unverified)')
    a = ap.parse_args()
    if a.cmd == 'selftest':
        return selftest()
    import serial
    b = Board(serial.Serial(a.port, a.baud, timeout=60), a.vocab, a.max_seq)
    try:
        if a.cmd == 'ping': print('board version', b.ping())
        elif a.cmd == 'info':
            r = b.info(a.bar)
            for k, v in r.items(): print(f'{k}: {v}')
            print('COMPATIBLE with the bitstream' if r['compatible'] else 'NOT COMPATIBLE: ' + '; '.join(r['problems']))
            return 0 if r['compatible'] else 1
        elif a.cmd == 'id': print(b.flash_id().hex(' '))
        elif a.cmd == 'cr1':
            cr = b.cr1(); print(f'SR1={b.sr1():#04x} CR1={cr:#04x} QUAD={(cr >> 1) & 1} LC={(cr >> 6) & 3:02b}')
        elif a.cmd == 'enable-quad': print(*b.enable_quad())
        elif a.cmd == 'step': print(b.step(a.token, a.pos, a.probe))
        else: print(b.run([int(x) for x in a.prompt.split(',')], a.gen))
    except (BoardError, FlashConfigError, ValueError, TimeoutError) as e:
        print('ERROR:', e, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
