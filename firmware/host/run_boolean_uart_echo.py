#!/usr/bin/env python3
"""Measure Boolean board UART echo byte error rates at supported baud candidates."""
from __future__ import annotations

import argparse
import hashlib
import secrets
import time


BAUD_RATES = (115200, 781250, 3125000)
MEASURE_BYTES = 64 * 1024
CHUNK_BYTES = 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Boolean board USB-UART COM port")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="per-chunk read/write timeout in seconds")
    args = parser.parse_args()
    try:
        import serial
    except ImportError:
        parser.error("pyserial is required (python -m pip install pyserial)")

    any_errors = False
    for baud in BAUD_RATES:
        payload = secrets.token_bytes(MEASURE_BYTES)
        received = bytearray()
        try:
            with serial.Serial(args.port, baud, timeout=args.timeout,
                               write_timeout=args.timeout) as port:
                port.reset_input_buffer()
                port.reset_output_buffer()
                time.sleep(0.1)
                for offset in range(0, MEASURE_BYTES, CHUNK_BYTES):
                    chunk = payload[offset:offset + CHUNK_BYTES]
                    written = port.write(chunk)
                    if written != len(chunk):
                        received.extend(b"")
                    port.flush()
                    received.extend(port.read(len(chunk)))
        except Exception as exc:
            any_errors = True
            print(f"baud={baud} sent={MEASURE_BYTES} received={len(received)} "
                  f"byte_errors={MEASURE_BYTES} byte_error_rate=1.0 error={exc}")
            continue

        compared = min(len(payload), len(received))
        mismatches = sum(a != b for a, b in zip(payload, received))
        errors = mismatches + (MEASURE_BYTES - compared)
        error_rate = errors / MEASURE_BYTES
        any_errors |= errors != 0
        print(
            f"baud={baud} sent={MEASURE_BYTES} received={len(received)} "
            f"byte_errors={errors} byte_error_rate={error_rate:.8f} "
            f"payload_sha256={hashlib.sha256(payload).hexdigest()}"
        )
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
