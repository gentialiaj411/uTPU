"""Recompute chunk-3 hardware budgets after tied-table storage was enabled."""
from __future__ import annotations
import json
from pathlib import Path
from .export import export_runq_model
from .run_15m import hardware_budget
from tools.llama2c.reference import Llama2CModel

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def main():
    path = ROOT / "bench/results/numerics_v2_15m.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    reference = Llama2CModel(ROOT / "models/llama2c/stories15M.bin")
    calibration = json.loads((HERE / "calibration_15m.json").read_text(encoding="utf-8"))["activation_scales"]
    for row in document["configs"]:
        bits, group = int(row["weight_bits"]), int(row["group_size"])
        improved = row["weight_quantizer"] == "mse_optimal_clipping"
        exported = export_runq_model(reference, calibration, bits, group, improved)
        exported.pop("dynamic_base_scales")
        row["hardware_budget"] = hardware_budget(exported, bits)
        row["hardware_budget"]["rebudgeted_tied_table"] = True
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
