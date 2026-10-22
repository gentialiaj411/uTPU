"""Finalize the real-GPTQ diagnostic matrix and verbatim samples."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FLOAT_PPL = 2.735893

def main():
    probe = json.loads((ROOT / "bench/results/_real_probe.json").read_text(encoding="utf-8"))
    sanity = json.loads((ROOT / "bench/results/gptq_sanity.json").read_text(encoding="utf-8"))
    samples = json.loads((ROOT / "bench/results/chunk4_samples.json").read_text(encoding="utf-8"))
    rows = probe["configs"]
    for row in rows:
        budget = row["hardware_budget"]
        row["delta_vs_float_pct"] = 100.0 * (row["perplexity"] / FLOAT_PPL - 1.0)
        row["summary"] = {"config": row["name"], "ppl": row["perplexity"],
                           "delta_vs_float_pct": row["delta_vs_float_pct"],
                           "top1_pct": row["teacher_forced_top1_agreement_pct"],
                           "flash_bytes": budget["weight_bytes_including_scales"],
                           "fits_flash": budget["fits_flash"],
                           "streamed_bytes_per_token": budget["bytes_streamed_per_token"],
                           "tok_s_25_50": [budget["estimated_tokens_per_second"]["25_MBps"], budget["estimated_tokens_per_second"]["50_MBps"]]}
    out = {"schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "model": "stories15M", "float_perplexity": FLOAT_PPL,
           "eval_tokens": probe["eval_tokens"],
           "eval_scope": "512-token diagnostic after full-GPTQ workers were resource-terminated; not a 50k claim",
           "configs": rows, "summary_table": [r["summary"] for r in rows],
           "gptq_sanity": sanity, "samples": samples,
           "sample_note": "same best-fitting tensor mix; sample export used the fast MSE-clipped offline quantizer",
           "gptq_contract": {"hessian": "2XX^T", "sequences": 256, "damping": "1% mean diagonal",
                             "act_order": True, "lazy_block": 128, "scale_search_candidates": 12}}
    (ROOT / "bench/results/numerics_v2_chunk5.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out["summary_table"], indent=2))

if __name__ == "__main__":
    main()
