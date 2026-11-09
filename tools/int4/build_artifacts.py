"""Build mixed-precision numerics-v2 artifacts entirely inside this worktree."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.llama2c.reference import Llama2CModel
from firmware.host.numerics_v2.export import export_runq_model

HERE = ROOT / "firmware/host/numerics_v2"
OUT = ROOT / "build/int4/artifacts"

SPECS = {
    "everything": {"qkv": 4, "wo": 4, "w13": 4, "w2": 4, "classifier": 4},
    "qkv": {"qkv": 4, "wo": 8, "w13": 4, "w2": 8, "classifier": 4},
    "wo": {"qkv": 8, "wo": 4, "w13": 4, "w2": 8, "classifier": 4},
    "w2": {"qkv": 8, "wo": 8, "w13": 4, "w2": 4, "classifier": 4},
    "all_attention": {"qkv": 4, "wo": 4, "w13": 4, "w2": 8, "classifier": 4},
    "w2_qkv": {"qkv": 4, "wo": 8, "w13": 4, "w2": 4, "classifier": 4},
}

def main() -> None:
    ref = Llama2CModel(ROOT / "models/llama2c/stories15M.bin")
    cal = json.loads((HERE / "calibration_15m.json").read_text())
    scales = cal["activation_scales"]
    hessian = {k: np.asarray(v, dtype=np.float64) for k, v in cal["hessian_diag"].items()}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, mixed in SPECS.items():
        payload = export_runq_model(ref, scales, bits=8, group_size=32,
                                    mixed_precision=mixed, hessian_diag=hessian,
                                    gptq=True)
        path = OUT / f"{name}.npz"
        np.savez_compressed(path, payload=np.asarray([payload], dtype=object))
        print(name, path, path.stat().st_size)

    improved_mixed = SPECS["everything"]
    improved = export_runq_model(ref, scales, bits=8, group_size=32,
                                 improved=True, mixed_precision=improved_mixed,
                                 hessian_diag=hessian, gptq=False)
    path = OUT / "everything_mse.npz"
    np.savez_compressed(path, payload=np.asarray([improved], dtype=object))
    print("everything_mse", path, path.stat().st_size)
    improved_wo = export_runq_model(ref, scales, bits=8, group_size=32,
                                    improved=True, mixed_precision=SPECS["wo"],
                                    hessian_diag=hessian, gptq=False)
    path = OUT / "wo_mse.npz"
    np.savez_compressed(path, payload=np.asarray([improved_wo], dtype=object))
    print("wo_mse", path, path.stat().st_size)

if __name__ == "__main__":
    main()
