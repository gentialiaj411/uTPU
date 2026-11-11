"""Pack a numerics_v2 stories15M image after deterministic vocabulary row pruning."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decode.pack_decode_image import load_export_15m, pack
from tools.vocabprune.core import prune_export


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-keep", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    idmap = json.loads(args.vocab_keep.read_text(encoding="utf-8"))
    export = prune_export(load_export_15m(), idmap["kept_original_ids"])
    image, params = pack(export)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image)
    args.output.with_suffix(".params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"image": str(args.output), "sha256": hashlib.sha256(image).hexdigest(), **params}, indent=2))


if __name__ == "__main__":
    main()
