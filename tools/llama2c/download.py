"""Download and verify pinned llama2.c models, tokenizers, and eval slice."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "llama2c"
MANIFEST = Path(__file__).with_name("manifest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def main() -> None:
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in spec["downloads"]:
        path = OUT / item["path"]
        fetch(item["url"], path)
        got = sha256(path)
        if item["sha256"] and got != item["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {path}: {got}")
        print(f"{got}  {path.relative_to(ROOT)}")
    source = OUT / spec["eval_slice"]["source_path"]
    target = OUT / spec["eval_slice"]["path"]
    target.write_bytes(source.read_bytes()[: spec["eval_slice"]["bytes"]])
    got = sha256(target)
    if got != spec["eval_slice"]["sha256"]:
        raise RuntimeError(f"eval slice sha256 mismatch: {got}")
    print(f"{got}  {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
