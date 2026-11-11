"""Generate decode-engine BRAM36 context budgets."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

BRAM36_BYTES = 4608
FIXED_BYTES = 36864  # activations + LUTs + 4 KiB stream FIFO, including aspect packing

def rows(model: str):
    if model == "15m":
        # Exported KV record is 3456 B/token at INT8; INT4 halves code plane.
        kv8, weight = 3456, 0
    else:
        # 260K is small enough to test residency of the 235,952-byte image.
        kv8, weight = 320, 235952
    result = []
    for length in (64, 96, 128, 192):
        for bits in (8, 4):
            total = FIXED_BYTES + weight + kv8 * length * bits / 8
            blocks = math.ceil(total / BRAM36_BYTES)
            result.append({"model": model, "context": length, "kv_bits": bits,
                           "kv_bytes": int(kv8 * length * bits / 8),
                           "fixed_bytes": FIXED_BYTES, "bram36": blocks,
                           "fits_70": blocks <= 70,
                           "weights_resident": bool(model == "260k" and blocks <= 70)})
    return result

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    payload = {"bram36_bytes": BRAM36_BYTES, "fixed_bytes": FIXED_BYTES,
               "rows": rows("15m") + rows("260k")}
    ap.parse_args().output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
if __name__ == "__main__": main()
