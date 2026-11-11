"""Generate the frozen integer transformer approximation tables.

The payload hash deliberately covers the canonical payload without its own hash
field, avoiding a self-referential file format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "firmware" / "host" / "fixtures" / "tiny_transformer_integer_luts.json"


def rne(value: Decimal) -> int:
    """Round a Decimal to nearest integer, with ties to even."""
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def build_payload() -> dict[str, list[int]]:
    """Return tables defined by the Phase-0 numerical contract.

    Decimal arithmetic (80 decimal digits) prevents binary-float rounding from
    becoming part of the ISA/RTL contract.  The specified values are not near a
    half-integer at this precision; ties, if any, use ROUND_HALF_EVEN.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        two = Decimal(2)
        exp_q15 = [rne((two ** 15) * (-(Decimal(k) / 16)).exp()) for k in range(129)]
        rsqrt_q14 = [
            rne((two ** 14) / (((Decimal(16384) + Decimal(256) * i + Decimal(128)) / (two ** 14)).sqrt()))
            for i in range(192)
        ]
    return {"exp_q15": exp_q15, "rsqrt_q14": rsqrt_q14}


def payload_bytes(payload: dict[str, list[int]]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def fixture_document() -> dict[str, object]:
    payload = build_payload()
    return {**payload, "payload_sha256": hashlib.sha256(payload_bytes(payload)).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail unless the output equals regenerated content")
    args = parser.parse_args()
    document = fixture_document()
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale LUT fixture: regenerate with {Path(__file__).name}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
