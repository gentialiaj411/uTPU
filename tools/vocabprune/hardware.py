"""Derive packed-image, stream, and schedule-limited throughput for each id map."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decode.pack_decode_image import load_export_15m, pack
from tools.vocabprune.core import KEEP_SIZES, prune_export

IDMAP_DIR = ROOT / "tools/vocabprune/idmaps"
BUILD = ROOT / "build/vocabprune"
OUT = ROOT / "bench/results/vocab_prune_study.json"
DEFAULT_SHA_BEFORE = "d0ac5863710ddec7f9b3cc2d9b526c1070fb7ecf3b07f656373dfdd0aca95541"
MEASURED_COMPUTE = 12_338_619
MEASURED_CLASSIFIER = 6_130_002
MEASURED_NON_CLASSIFIER = MEASURED_COMPUTE - MEASURED_CLASSIFIER
BASELINE_STREAM = 10_684_413


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    exp = load_export_15m()
    default_image, default_params = pack(exp)
    default_sha_after = hashlib.sha256(default_image).hexdigest()
    if default_sha_after != DEFAULT_SHA_BEFORE:
        raise AssertionError(f"default pack changed: {DEFAULT_SHA_BEFORE} -> {default_sha_after}")
    rows = {}
    for keep in KEEP_SIZES:
        idmap = json.loads((IDMAP_DIR / f"idmap_{keep}.json").read_text(encoding="utf-8"))
        image, params = pack(prune_export(exp, idmap["kept_original_ids"]))
        image_path = BUILD / f"vocab_{keep}.image.bin"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image)
        (image_path.with_suffix(".params.json")).write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        # This is the exact four-segment prefetch stream formula used by run_decode_rtl.py.
        stream = int(params["IMAGE_BYTES"] - params["LAYERS_BASE"] + params["EMB_SCALE_BASE"] +
                     params["CLS_KB"] + 5 * params["CLS_G"])
        classifier_cycles = MEASURED_CLASSIFIER * keep / 32000.0
        compute = MEASURED_NON_CLASSIFIER + classifier_cycles
        stream_cycles = stream / 0.5
        limit = max(stream_cycles, compute)
        rows[str(keep)] = {
            "status": "DERIVED",
            "image_path": str(image_path),
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "image_bytes": len(image),
            "stream_bytes_per_token": stream,
            "stream_cycles_per_token_at_0_5B_cycle": stream_cycles,
            "stream_bound_tok_s_at_50MHz": 50_000_000 / stream_cycles,
            "classifier_cycles_estimate": classifier_cycles,
            "non_classifier_cycles_measured": MEASURED_NON_CLASSIFIER,
            "compute_cycles_estimate": compute,
            "limiting_cycles_per_token": limit,
            "predicted_tok_s": 50_000_000 / limit,
            "speedup_vs_unpruned_stream": (BASELINE_STREAM / stream),
            "params": params,
        }
    baseline_limit = max(BASELINE_STREAM / 0.5, MEASURED_COMPUTE)
    result = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    result["hardware"] = {
        "status": "DERIVED",
        "default_pack_sha256_before": DEFAULT_SHA_BEFORE,
        "default_pack_sha256_after": default_sha_after,
        "default_pack_byte_identical": default_sha_after == DEFAULT_SHA_BEFORE,
        "baseline": {
            "image_bytes": default_params["IMAGE_BYTES"],
            "stream_bytes_per_token": BASELINE_STREAM,
            "stream_cycles_per_token": BASELINE_STREAM / 0.5,
            "compute_cycles_measured": MEASURED_COMPUTE,
            "classifier_cycles_measured_from_rtl_schedule": MEASURED_CLASSIFIER,
            "non_classifier_cycles_measured_from_rtl_schedule": MEASURED_NON_CLASSIFIER,
            "limiting_cycles_per_token": baseline_limit,
            "predicted_tok_s": 50_000_000 / baseline_limit,
        },
        "candidates": rows,
        "schedule_source": "bench/results/decode_e2e_rtl_15m_kv4_p1.json; classifier profile divided by 11 steps",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["hardware"], indent=2))


if __name__ == "__main__":
    main()
