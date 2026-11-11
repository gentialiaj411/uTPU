"""Reconstruction sanity check for one real GPTQ layer versus RTN."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from tools.llama2c.reference import Llama2CModel
from .export import export_runq_model

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def reconstruct(record, input_scale):
    codes = record["weight"].astype(np.float64)
    out, width = codes.shape
    q = np.zeros_like(codes)
    for group, begin in enumerate(range(0, width, 32)):
        end = min(width, begin + 32)
        scale = (record["weight_scale_multiplier"][:, group].astype(np.float64) /
                 np.exp2(record["weight_scale_shift"][:, group])) * input_scale
        q[:, begin:end] = codes[:, begin:end] * scale[:, None]
    return q

def main():
    ref = Llama2CModel(ROOT / "models/llama2c/stories15M.bin")
    document = json.loads((HERE / "calibration_15m.json").read_text(encoding="utf-8"))
    scales, hessian = document["activation_scales"], document["hessian_diag"]
    rtn = export_runq_model(ref, scales, 4, 32, gptq=False)["layers"][0]["wq"]
    gptq = export_runq_model(ref, scales, 4, 32, hessian_diag=hessian, gptq=True)["layers"][0]["wq"]
    # H=2XX^T; its Cholesky factor is an equivalent calibration activation matrix.
    x = np.linalg.cholesky(np.asarray(hessian["norm"], dtype=np.float64))
    weight = np.asarray(ref.wq[0], dtype=np.float64)
    values = {"rtn_weighted_reconstruction_error": float(np.linalg.norm((weight - reconstruct(rtn, scales["norm"])) @ x)),
              "gptq_weighted_reconstruction_error": float(np.linalg.norm((weight - reconstruct(gptq, scales["norm"])) @ x)),
              "gptq_lower_than_rtn": True}
    values["gptq_lower_than_rtn"] = values["gptq_weighted_reconstruction_error"] < values["rtn_weighted_reconstruction_error"]
    print(json.dumps(values, indent=2))
    (ROOT / "bench/results/gptq_sanity.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
