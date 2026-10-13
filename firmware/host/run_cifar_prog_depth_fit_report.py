#!/usr/bin/env python3
"""Report CIFAR-10 all-conv INT8 program words vs PROG_DEPTH=65536 (+ buffer-resident sketch).

Read-only fit report — does not propose RTL work.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CNN = REPO / "bench" / "results" / "utpu_small_cnn_validation.json"
OUT = REPO / "bench" / "results" / "cifar_prog_depth_fit_report.json"
PROG_DEPTH = 65536
UPLOAD_LEN_MAX = 65535


def main() -> None:
    blob = json.loads(CNN.read_text(encoding="utf-8"))
    layers = ((blob.get("rtl_bitmatch") or {}).get("layers")) or {}
    rows = []
    for name, layer in layers.items():
        words = int(layer.get("program_instruction_words") or 0)
        # Honest buffer-resident sketch: if payload were removed from the image,
        # only control words remain. Without a composition split in this artifact,
        # we can only report full program words and note that buffer-resident
        # weights are required for any layer whose words exceed PROG_DEPTH.
        rows.append(
            {
                "layer": name,
                "program_instruction_words": words,
                "fits_prog_depth_65536": words <= PROG_DEPTH,
                "fits_upload_len_max": words <= UPLOAD_LEN_MAX,
                "overflow_words": max(0, words - PROG_DEPTH),
            }
        )
    summary = {
        "schema_version": 1,
        "source_artifact": "bench/results/utpu_small_cnn_validation.json",
        "prog_depth": PROG_DEPTH,
        "upload_len_max": UPLOAD_LEN_MAX,
        "buffer_resident_weights_note": (
            "Existing CNN artifact records full program images (weights embedded). "
            "Buffer-resident weights would shrink the instruction stream by the "
            "payload fraction; without a per-layer composition split here we only "
            "report raw program_instruction_words and whether they fit as-is."
        ),
        "layers": rows,
        "any_layer_fits_as_is": any(r["fits_prog_depth_65536"] for r in rows),
        "all_layers_fit_as_is": all(r["fits_prog_depth_65536"] for r in rows) if rows else False,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
