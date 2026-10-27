"""Host side of the Boolean UART protocol (rtl/board/board_ctrl.sv). Needs pyserial (pip install pyserial).

  python tools/board/boolean_host.py --port COM5 ping
  python tools/board/boolean_host.py --port COM5 id            # flash JEDEC ID + sector architecture
  python tools/board/boolean_host.py --port COM5 cr1           # read SR1 and CR1, report the QUAD bit
  python tools/board/boolean_host.py --port COM5 enable-quad   # set CR1[1] (non-volatile) if it is 0, verify
  python tools/board/boolean_host.py --port COM5 step --token 1 --pos 0 --probe 5
  python tools/board/boolean_host.py --port COM5 run --prompt 1,403,260 --gen 8
  python tools/board/boolean_host.py selftest                  # frame encoding vs a software stub (no board needed)

The frame layouts are the ones exercised by tools/board/run_uart_rtl.py against the RTL. Nothing here has talked to a real board.
"""
from __future__ import annotations
import argparse, struct, sys, time


class Board:
    def __init__(self, ser):
        self.ser = ser

    def _rx(self, n, what):
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError(f'{what}: expected {n} bytes, got {len(data)} ({data.hex()})')
        return data

    def ping(self):
        self.ser.write(b'V')
        r = self._rx(2, 'ping')
        assert r == b'\x76\x01', r
        return r[1]

    def spi(self, tx: bytes, nrx: int) -> bytes:
        """Single-bit SPI transaction on the flash: CS# low, tx out on IO0, nrx bytes in on IO1, CS# high (<= 8 each)."""
        assert len(tx) <= 8 and nrx <= 8
        self.ser.write(bytes([0x53, len(tx), nrx]) + tx)
        r = self._rx(2 + nrx, 'spi')
        assert r[0] == 0x73 and r[1] == nrx, r
        return r[2:]

    def flash_id(self):
        return self.spi(b'\x9f', 6)               # 01 20 18 4D <00 uniform 256 kB | 01 hybrid 64 kB> 80

    def sr1(self):
        return self.spi(b'\x05', 1)[0]

    def cr1(self):
        return self.spi(b'\x35', 1)[0]

    def enable_quad(self, timeout=5.0):
        sr, cr = self.sr1(), self.cr1()
        if cr & 0x02:
            return 'already set', cr
        self.spi(b'\x06', 0)                        # WREN
        if not self.sr1() & 0x02:
            raise RuntimeError('WEL did not set after WREN')
        self.spi(bytes([0x01, sr, cr | 0x02]), 0)   # WRR: SR1 unchanged, CR1 | QUAD (OTP bits are written back as read)
        t0 = time.time()
        while self.sr1() & 0x01:                    # WIP
            if time.time() - t0 > timeout:
                raise TimeoutError('WRR did not finish')
        cr2 = self.cr1()
        if not cr2 & 0x02:
            raise RuntimeError(f'QUAD still 0 after WRR (CR1={cr2:#04x}); check WP# / SR1.SRWD and that the write was not blocked')
        return 'set', cr2

    def step(self, token: int, pos: int, probe: int = 0):
        """One decode step -> (argmax token, core cycles, INT32 logit at index probe)."""
        self.ser.write(bytes([0x54, token & 255, token >> 8, pos, probe & 255, probe >> 8]))
        r = self._rx(11, 'step')
        assert r[0] == 0x52, r
        tok, cyc, logit = struct.unpack('<HIi', r[1:])
        return tok, cyc, logit

    def run(self, prompt, gen):
        """Greedy decode: feed the prompt, then the board's own argmax; returns the generated token ids."""
        out, tok = [], prompt[0]
        for p in range(len(prompt) + gen - 1):
            nxt, cyc, _ = self.step(tok, p)
            if p + 1 < len(prompt):
                tok = prompt[p + 1]
            else:
                out.append(nxt); tok = nxt
        return out


class Stub:
    """Software stand-in for the board used by selftest: answers V and S frames like board_ctrl.sv, and T frames with a fixed reply."""
    def __init__(self):
        self.cr1, self.sr1, self.wel, self.wip, self.buf, self.log = 0x00, 0x00, 0, 0, b'', []
    def write(self, b):
        self.log.append(bytes(b))
        if b[0:1] == b'V':
            self.buf += b'\x76\x01'
        elif b[0:1] == b'S':
            ntx, nrx, tx = b[1], b[2], b[3:3 + b[1]]
            rx = b''
            if tx[:1] == b'\x9f': rx = bytes([1, 0x20, 0x18, 0x4D, 0x00, 0x80])[:nrx]
            elif tx[:1] == b'\x05':
                rx = bytes([self.sr1 | (self.wel << 1) | (1 if self.wip else 0)]); self.wip = max(0, self.wip - 1)
            elif tx[:1] == b'\x35': rx = bytes([self.cr1])
            elif tx[:1] == b'\x06': self.wel = 1
            elif tx[:1] == b'\x01' and self.wel:
                self.sr1, self.cr1, self.wel, self.wip = tx[1] & 0xFC, tx[2], 0, 2
            self.buf += b'\x73' + bytes([nrx]) + rx
        elif b[0:1] == b'T':
            self.buf += b'R' + struct.pack('<HIi', 147, 930674, -5)
    def read(self, n):
        d, self.buf = self.buf[:n], self.buf[n:]
        return d


def selftest():
    s = Stub(); b = Board(s)
    assert b.ping() == 1
    assert b.flash_id() == bytes([1, 0x20, 0x18, 0x4D, 0x00, 0x80])
    assert b.cr1() == 0
    state, cr = b.enable_quad()
    assert state == 'set' and cr & 2, (state, cr)
    assert b.enable_quad()[0] == 'already set'
    assert s.log[-1] == b'S\x01\x01\x35' or s.log[-1][:1] == b'S'
    assert b.step(1, 2, 0x0105) == (147, 930674, -5)
    assert s.log[-1] == bytes([0x54, 1, 0, 2, 5, 1]), s.log[-1]
    assert [l for l in s.log if l[:1] == b'S' and l[3:4] == b'\x01'][0] == bytes([0x53, 3, 0, 0x01, 0x00, 0x02]), 'WRR frame'
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port'); ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('cmd', choices=['ping', 'id', 'cr1', 'enable-quad', 'step', 'run', 'selftest'])
    ap.add_argument('--token', type=int, default=1); ap.add_argument('--pos', type=int, default=0); ap.add_argument('--probe', type=int, default=0)
    ap.add_argument('--prompt', default='1'); ap.add_argument('--gen', type=int, default=8)
    a = ap.parse_args()
    if a.cmd == 'selftest':
        return selftest()
    import serial
    b = Board(serial.Serial(a.port, a.baud, timeout=60))
    if a.cmd == 'ping': print('board version', b.ping())
    elif a.cmd == 'id': print(b.flash_id().hex(' '))
    elif a.cmd == 'cr1': print(f'SR1={b.sr1():#04x} CR1={b.cr1():#04x} QUAD={(b.cr1() >> 1) & 1}')
    elif a.cmd == 'enable-quad': print(*b.enable_quad())
    elif a.cmd == 'step': print(b.step(a.token, a.pos, a.probe))
    else: print(b.run([int(x) for x in a.prompt.split(',')], a.gen))


if __name__ == '__main__':
    sys.exit(main())
