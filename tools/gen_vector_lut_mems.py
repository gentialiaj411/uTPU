"""Render RTL ROM initializers from the frozen transformer LUT fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "firmware/host/fixtures/tiny_transformer_integer_luts.json"
OUT = ROOT / "rtl/vector"


def payload(doc: dict) -> bytes:
    return json.dumps({"exp_q15": doc["exp_q15"], "rsqrt_q14": doc["rsqrt_q14"]},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def render(values: list[int]) -> str:
    return "\n".join(f"{int(v) & 0xffff:04x}" for v in values) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(payload(doc)).hexdigest()
    if digest != doc.get("payload_sha256"):
        raise SystemExit("frozen LUT fixture hash mismatch")
    outputs = {OUT / "exp_q15.mem": render(doc["exp_q15"]), OUT / "rsqrt_q14.mem": render(doc["rsqrt_q14"])}
    stale = [str(path.relative_to(ROOT)) for path, text in outputs.items() if not path.is_file() or path.read_text(encoding="ascii") != text]
    if args.check:
        if stale: raise SystemExit("stale vector LUT mem: " + ", ".join(stale))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for path, text in outputs.items(): path.write_text(text, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
