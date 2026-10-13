"""Download TinyShakespeare without committing the corpus."""
from __future__ import annotations
import argparse, hashlib, json, urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "tiny_transformer_vocab.json"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/tiny_transformer/input.txt"))
    args = parser.parse_args()
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))["corpus_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            args.output.write_bytes(response.read())
    actual = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch: expected {expected}, got {actual}")
    print(f"verified {args.output} sha256={actual}")

if __name__ == "__main__":
    main()
