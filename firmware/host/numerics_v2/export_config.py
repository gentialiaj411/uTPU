"""Export exactly one integer model artifact for a subsequent bounded eval."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from tools.llama2c.reference import Llama2CModel
from .export import export_runq_model

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mixed", default="")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    mixed = {k: int(v) for item in args.mixed.split(",") if item for k, v in [item.split("=")]}
    ref = Llama2CModel(ROOT / "models/llama2c/stories15M.bin")
    cal = json.loads((HERE / "calibration_15m.json").read_text())
    exported = export_runq_model(ref, cal["activation_scales"], 4, 32,
                                 hessian_diag=cal["hessian_diag"], mixed_precision=mixed, gptq=True)
    out = args.output or (ROOT / "models/numerics_v2" / f"{args.name}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, payload=np.asarray([exported], dtype=object))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {"name": args.name, "path": str(out), "sha256": digest, "weight_bits": 4,
            "group_size": 32, "gptq": True, "mixed_precision": mixed,
            "peak_export_ram_note": "single config; object payload written once"}
    print(json.dumps(meta, indent=2))

if __name__ == "__main__": main()
