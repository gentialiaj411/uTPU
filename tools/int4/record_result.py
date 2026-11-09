"""Merge completed fixed-text eval JSON files into the study artifacts."""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = 0.5623131903694261
MAP = {"qkv_eval.json":"qkv", "wo_eval.json":"wo", "w2_eval.json":"w2",
       "all_attention_eval.json":"all_attention", "w2_qkv_eval.json":"w2_qkv",
       "everything_mse_eval.json":"everything_mse", "pruned_eval.json":"best_pruned"}

def main():
    study_path = ROOT / "bench/results/int4_study.json"
    out = json.loads(study_path.read_text()) if study_path.exists() else {"rows": []}
    rows = {r["config"]: r for r in out.get("rows", [])}
    for fn, config in MAP.items():
        p = ROOT / "build/int4" / fn
        if not p.exists(): continue
        r = json.loads(p.read_text())
        bpb = r["bits_per_byte"]
        row = rows.setdefault(config, {"config": config})
        row.update({"int_bpb": bpb, "quality_status":"MEASURED",
                    "delta_pct": (bpb / BASE - 1.0) * 100.0,
                    "wall_seconds": r["elapsed_seconds"], "eval_tokens": r["eval_tokens"],
                    "artifact": r["artifact"], "artifact_sha256": r["artifact_sha256"]})
    out["rows"] = list(rows.values())
    out["status"] = "quality_results_incremental"
    out["fixed_eval_baseline_bpb"] = BASE
    out["last_updated_from_script"] = "tools/int4/record_result.py"
    out["not_run"] = ["Remaining evals and pruned+INT4 are not run until their processes complete."]
    study_path.write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# stories15M mixed-precision INT4 study", "", "Every quality result below is MEASURED from the fixed text; byte and throughput fields are DERIVED/ESTIMATED.", "", "| config | int bpb (MEASURED) | delta | wall s |", "|---|---:|---:|---:|"]
    for r in out["rows"]:
        if r.get("quality_status") == "MEASURED": lines.append(f"| {r['config']} | {r['int_bpb']:.9f} | {r['delta_pct']:+.3f}% | {r.get('wall_seconds', 0):.1f} |")
        elif r.get("config") == "measured_baseline_integer": lines.append(f"| {r['config']} | {r['int_bpb']:.9f} | 0.000% | — |")
    lines += ["", "Baseline: 0.5623131903694261 bpb (MEASURED). Fixed text SHA-256: c8191cfe5a461bc825b3e3326a2921dfE978a63878b4c14547033b7217adea17 (MEASURED).", "", "Artifacts are built by `python tools/int4/build_artifacts.py`; eval command uses `python tools/int4/integer_eval.py` and writes only under `build/int4/`."]
    (ROOT / "docs/INT4_STUDY.md").write_text("\n".join(lines) + "\n")

if __name__ == "__main__": main()
