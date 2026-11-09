"""Reproducible INT4 design-space accounting for stories15M.

Quality is deliberately not approximated here: the fixed-text integer evaluator is
invoked separately and its JSON is copied into the study only when it completes.
"""
from __future__ import annotations
import hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz"
BASE_STREAM = 10_684_413
PRUNED_STREAM = 6_613_245
COMPUTE = 12_338_619
PRUNED_COMPUTE = 7_780_000  # supplied RTL schedule estimate for K=8192 study

def tensor_bytes(rec, rows=None):
    w = np.asarray(rec["weight"])
    rows = w.shape[0] if rows is None else rows
    k = w.shape[1]
    groups = int(np.asarray(rec["groups"]).shape[1])
    kb = k // 2 if int(rec["qmax"]) == 7 else k
    return ((rows + 15) // 16) * (groups * 16 * 3 + 16 * kb)

def run():
    p = np.load(ART, allow_pickle=True)["payload"][0]
    names = ("wq", "wk", "wv", "wo", "w1", "w3", "w2")
    rec = p["layers"][0]
    b8 = {n: tensor_bytes({**rec[n], "qmax": 127}) for n in names}
    # Shapes/groups are invariant; derive INT4 payload size without changing the artifact.
    b4 = {}
    for n in names:
        w = np.asarray(rec[n]["weight"])
        g = np.asarray(rec[n]["groups"]).shape[1]
        b4[n] = ((w.shape[0] + 15) // 16) * (g * 16 * 3 + 16 * (w.shape[1] // 2))
    # The fixed stream contains a small non-weight schedule/scale contribution.
    overhead = BASE_STREAM - sum((b4[n] if int(rec[n]["qmax"]) == 7 else b8[n]) * 6 for n in names) - tensor_bytes(p["classifier"])
    configs = {
        "baseline_pruned_only": set(),
        "wq": {"wq"}, "wk": {"wk"}, "wv": {"wv"}, "wo": {"wo"},
        "w1": {"w1"}, "w3": {"w3"}, "w2": {"w2"},
        "qkv": {"wq", "wk", "wv"}, "all_attention": {"wq", "wk", "wv", "wo"},
        "w13": {"w1", "w3"}, "all_ffn": {"w1", "w3", "w2"},
        "everything": set(names),
    }
    rows = []
    for name, extra in configs.items():
        layer = sum((b4[n] if n in extra or int(rec[n]["qmax"]) == 7 else b8[n]) * 6 for n in names)
        cls = tensor_bytes(p["classifier"])
        unpruned = overhead + layer + cls
        # K=8192 is tile-aligned; tied classifier stream scales linearly in tiles.
        pruned_cls = cls * 8192 // 32000
        pruned = overhead + layer + pruned_cls
        c = PRUNED_COMPUTE if name != "unpruned_reference" else COMPUTE
        rows.append({"config": name, "int_bpb": None, "quality_status": "NOT RUN",
                     "delta_pct": None, "stream_B_token_unpruned": unpruned,
                     "stream_B_token_pruned_K8192": pruned, "compute_cycles_est": c,
                     "tok_s_unpruned_est": 50_000_000 / max(unpruned / 0.5, COMPUTE),
                     "tok_s_pruned_est": 50_000_000 / max(pruned / 0.5, PRUNED_COMPUTE),
                     "speedup_vs_pruned_est": (50_000_000 / max(pruned / 0.5, PRUNED_COMPUTE)) /
                     (50_000_000 / max(PRUNED_STREAM / 0.5, PRUNED_COMPUTE))})
    rows.insert(0, {"config": "measured_baseline_integer", "int_bpb": 0.5623131903694261,
                    "quality_status": "MEASURED", "delta_pct": 0.0,
                    "stream_B_token_unpruned": BASE_STREAM, "stream_B_token_pruned_K8192": PRUNED_STREAM,
                    "compute_cycles_est": COMPUTE, "tok_s_unpruned_est": 50_000_000 / max(BASE_STREAM / .5, COMPUTE),
                    "tok_s_pruned_est": 50_000_000 / max(PRUNED_STREAM / .5, PRUNED_COMPUTE),
                    "speedup_vs_pruned_est": 1.0})
    out = {"schema_version": 1, "status": "complete_for_accounting_quality_pending",
           "labels": {"MEASURED": "direct artifact/protocol result", "DERIVED": "integer arithmetic from packed records",
                       "ESTIMATED": "RTL schedule / throughput model", "NOT RUN": "not executed"},
           "artifact": str(ART), "artifact_sha256": hashlib.sha256(ART.read_bytes()).hexdigest(),
           "group_size": 32, "classifier_K": 8192, "baseline_stream_B_token": BASE_STREAM,
           "pruned_stream_B_token": PRUNED_STREAM, "compute_cycles_unpruned_measured": COMPUTE,
           "compute_cycles_pruned_estimated": PRUNED_COMPUTE, "weight_bytes": {"int8": b8, "int4": b4},
           "fixed_eval_baseline_bpb": 0.5623131903694261,
           "quality_protocol_command": "python C:/Users/bhask/Documents/PROJECTS/uTPU-codesign/tools/codesign/integer_eval.py --artifact <artifact.npz> --checkpoint models/llama2c/stories15M.bin --tokenizer models/llama2c/tokenizer.bin --text C:/Users/bhask/Documents/PROJECTS/uTPU-codesign/build/codesign/eval_text.txt --output <result.json>",
           "rows": rows,
           "not_run": ["Full fixed-text quality eval for every new mixed artifact: NOT RUN; exact command recorded above.",
                       "Per-group MSE/GPTQ improvement ablation on fixed text: NOT RUN; existing exporter supports it but no result is claimed.",
                       "GO implementation and RTL simulation for a new config: NOT RUN because no candidate has measured quality."]}
    dest = ROOT / "bench/results/int4_study.json"
    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__": run()
