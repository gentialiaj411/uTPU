"""Merge chunk-4 full rows and sensitivity probes into one summary artifact."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FLOAT_PPL = 2.735893


def read_row(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    return document["configs"][0]


def main():
    full = [read_row(ROOT / "bench/results/_chunk4_baseline.json"),
            read_row(ROOT / "bench/results/_chunk4_gptq32.json"),
            read_row(ROOT / "bench/results/_chunk4_gptq64.json"),
            read_row(ROOT / "bench/results/_chunk4_best.json")]
    sensitivity = json.loads((ROOT / "bench/results/numerics_v2_chunk4.json").read_text(encoding="utf-8"))
    sensitivity_rows = sensitivity.get("sensitivity", sensitivity["configs"])
    sensitivity_tokens = sensitivity.get("sensitivity_eval_tokens", sensitivity.get("eval_tokens", 512))
    baseline_bytes = next(row["hardware_budget"]["weight_bytes_including_scales"]
                          for row in sensitivity_rows if row["name"] == "baseline_int4_g32")
    for row in sensitivity_rows:
        row["bytes_added_vs_baseline"] = (row["hardware_budget"]["weight_bytes_including_scales"] - baseline_bytes)
    for row in full:
        row["delta_vs_float_pct"] = 100.0 * (row["perplexity"] / FLOAT_PPL - 1.0)
        budget = row["hardware_budget"]
        row["summary"] = {"config": row["name"], "ppl": row["perplexity"],
                           "delta_vs_float_pct": row["delta_vs_float_pct"],
                           "top1_pct": row["teacher_forced_top1_agreement_pct"],
                           "flash_bytes": budget["weight_bytes_including_scales"],
                           "fits_flash": budget["fits_flash"],
                           "streamed_bytes_per_token": budget["bytes_streamed_per_token"],
                           "tok_s_25_50": [budget["estimated_tokens_per_second"]["25_MBps"],
                                           budget["estimated_tokens_per_second"]["50_MBps"]]}
    # Pareto marker is restricted to full-evaluation rows that fit the payload.
    fitting = [row for row in full if row["hardware_budget"]["fits_flash"]]
    for row in full:
        row["pareto_best_fit"] = bool(row in fitting and not any(
            other["perplexity"] <= row["perplexity"] and
            other["hardware_budget"]["weight_bytes_including_scales"] <= row["hardware_budget"]["weight_bytes_including_scales"] and
            (other["perplexity"] < row["perplexity"] or other["hardware_budget"]["weight_bytes_including_scales"] < row["hardware_budget"]["weight_bytes_including_scales"])
            for other in fitting))
    output = {"schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "model": "stories15M", "float_perplexity": FLOAT_PPL,
              "full_eval_tokens": 50000, "sensitivity_eval_tokens": sensitivity_tokens,
              "flash_payload_limit_bytes": 16 * 1024 * 1024 - int(2.5 * 1024 * 1024),
              "configs": full, "sensitivity": sensitivity_rows,
              "summary_table": [row["summary"] for row in full],
              "gptq": {"method": "diagonal Hessian weighted clipping", "calibration": sensitivity.get("calibration", sensitivity.get("gptq", {}).get("calibration", {}))}}
    path = ROOT / "bench/results/numerics_v2_chunk4.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary_table"], indent=2))


if __name__ == "__main__":
    main()
