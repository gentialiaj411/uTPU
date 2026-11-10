"""Attach decision, RTL, constraint, and reproducibility metadata to the study JSON."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
OUT = ROOT / "bench/results/vocab_prune_study.json"


def main() -> None:
    result = json.loads(OUT.read_text(encoding="utf-8"))
    base = result["integer"]["32000"]["bits_per_byte"]
    base_speed = result["hardware"]["baseline"]["predicted_tok_s"]
    quality_limit = base * 1.01
    speed_limit = base_speed * 1.3
    candidates = {}
    for k in (2048, 4096, 6144, 8192):
        q = result["integer"][str(k)]["bits_per_byte"]
        speed = result["hardware"]["candidates"][str(k)]["predicted_tok_s"]
        candidates[str(k)] = {
            "speed_pass": speed >= speed_limit,
            "quality_pass": q <= quality_limit,
            "speedup": speed / base_speed,
            "integer_bpb_delta_pct": (q / base - 1.0) * 100.0,
        }
    result["decision"] = {
        "result": "GO" if any(x["speed_pass"] and x["quality_pass"] for x in candidates.values()) else "NO-GO",
        "rule": "GO iff predicted tok/s >= 1.3x unpruned stream-bound tok/s and integer bpb <= 1.01x unpruned integer bpb",
        "baseline_integer_bpb": base,
        "integer_bpb_limit": quality_limit,
        "baseline_predicted_tok_s": base_speed,
        "speed_limit_tok_s": speed_limit,
        "candidates": candidates,
        "recommended_vocab": 8192,
    }
    result["rtl_constraints"] = {
        "token_and_output_id_width": "16 bits; VOCAB must fit <= 65535 (rtl/decode/top_decode.sv:35,44,198,306-310)",
        "classifier_tiling": "g_ntiles=ceil(VOCAB/16), final tile rows are bounds-checked; all tested K are multiples of 16 (rtl/decode/top_decode.sv:306-310, tools/decode/pack_decode_image.py:9-44)",
        "int4_row_width": "CLS_KB=D/2=144 and D=288; K must be even for INT4 packing (top_decode.sv:24; pack_decode_image.py:38-40,71-76)",
        "grouping": "CLS_G=ceil(D/32)=9; K does not change CLS_G, only tile count/prefix sizes",
        "config_params": "CLS_TILE_BYTES, CLS_KB, CLS_G, VOCAB are generated in params.json and passed as top_decode parameters",
    }
    result["rtl"] = json.loads((ROOT / "bench/results/vocab_prune_rtl.json").read_text(encoding="utf-8")) if (ROOT / "bench/results/vocab_prune_rtl.json").exists() else {"status": "NOT RUN"}
    from tools.decode.provenance import provenance
    from tools.vocabprune.pruned_sources import PRUNED_SOURCES
    complete = (isinstance(result.get('integer', {}).get('8192', {}).get('bits_per_byte'), (int, float)) and
                result.get('hardware', {}).get('candidates', {}).get('8192', {}).get('predicted_tok_s', 0) > 0 and
                result['rtl'].get('status') == 'pass' and result['decision']['result'] == 'GO')
    result['status'] = 'pass' if complete else 'incomplete'
    result['provenance'] = provenance(PRUNED_SOURCES['vocab_prune_study'])
    result["not_run"] = [
        "Physical-board throughput: NOT RUN (implementation: bench/results/pruned_ddr_impl.json; DDR-flash RTL: bench/results/vocab_prune_ddr_rtl.json).",
        "Host-client code changes: NOT RUN by design; the remap is documented only.",
        "RTL smoke for K=2048, 4096, 6144: NOT RUN; only the recommended K=8192 was run.",
    ]
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
