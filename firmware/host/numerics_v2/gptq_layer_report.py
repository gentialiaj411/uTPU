"""Held-out WX reconstruction for every stories15M quantized matrix."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from tools.llama2c.torch_oracle import load_torch_model
from .calibrate import calibration_sequences
from .export import quantized_linear

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def dequant(record, input_scale):
    codes = record["weight"].astype(np.float64)
    scales = record["weight_scale_multiplier"].astype(np.float64) / np.exp2(record["weight_scale_shift"])
    out, width = codes.shape
    result = np.empty_like(codes)
    for g, begin in enumerate(range(0, width, 32)):
        result[:, begin:min(width, begin + 32)] = codes[:, begin:min(width, begin + 32)] * (scales[:, g] * input_scale)[:, None]
    return result

def main():
    ref = Llama2CModel(ROOT / "models/llama2c/stories15M.bin")
    tok = Tokenizer(ROOT / "models/llama2c/tokenizer.bin", ref.config.vocab_size)
    train = ROOT / "models/llama2c/TinyStories-train.slice.txt"
    rows, _, _ = calibration_sequences(train, tok, length=64, sequences=256)
    cal = json.loads((HERE / "calibration_15m.json").read_text())
    scales, hessian = cal["activation_scales"], cal["hessian_diag"]
    model = load_torch_model(ref, "cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    matrix_specs = []
    for li, layer in enumerate(model.layers):
        matrix_specs += [(f"layer{li}.wq", ref.wq[li], layer.attention.wq, "norm", "q"),
                         (f"layer{li}.wk", ref.wk[li], layer.attention.wk, "norm", "k"),
                         (f"layer{li}.wv", ref.wv[li], layer.attention.wv, "norm", "v"),
                         (f"layer{li}.wo", ref.wo[li], layer.attention.wo, "av", "x"),
                         (f"layer{li}.w1", ref.w1[li], layer.feed_forward.w1, "norm", "h1"),
                         (f"layer{li}.w3", ref.w3[li], layer.feed_forward.w3, "norm", "h3"),
                         (f"layer{li}.w2", ref.w2[li], layer.feed_forward.w2, "hidden", "x")]
    matrix_specs.append(("classifier", ref.wcls, model.output, "norm", "logits"))
    report = []
    for name, weight, module, hs, os in matrix_specs:
        captured = []
        handle = module.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0].detach().float().cpu()))
        with torch.inference_mode():
            for start in range(0, len(rows), 16):
                batch = torch.from_numpy(rows[start:start+16]).to(device)
                model(batch, batch)
        handle.remove()
        x = torch.cat(captured, dim=0).numpy().reshape(-1, captured[0].shape[-1])
        split = x.shape[0] // 2
        x_hold = x[split:].T.astype(np.float64)
        w = np.asarray(weight, dtype=np.float32)
        h = np.asarray(hessian[hs], dtype=np.float64)
        rtn = quantized_linear(w, scales[hs], scales[os], 4, 32, improved=False)
        gptq = quantized_linear(w, scales[hs], scales[os], 4, 32, hessian=h, gptq=True)
        wx = w.astype(np.float64) @ x_hold
        er = np.linalg.norm(wx - dequant(rtn, scales[hs]) @ x_hold) / max(np.linalg.norm(wx), 1e-12)
        eg = np.linalg.norm(wx - dequant(gptq, scales[hs]) @ x_hold) / max(np.linalg.norm(wx), 1e-12)
        report.append({"matrix": name, "rtn_relative_error": float(er), "gptq_relative_error": float(eg),
                       "reduction_pct": float(100*(er-eg)/max(er, 1e-12)), "heldout_vectors": int(x_hold.shape[1])})
        print(name, report[-1]["reduction_pct"], flush=True)
    reductions = np.asarray([r["reduction_pct"] for r in report])
    out = {"matrices": report, "median_reduction_pct": float(np.median(reductions)),
           "min_reduction_pct": float(np.min(reductions)), "max_reduction_pct": float(np.max(reductions)),
           "heldout_protocol": "second half of 256 TRAIN calibration rows; WX reconstruction"}
    (ROOT / "bench/results/gptq_layer_report.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("median_reduction_pct", "min_reduction_pct", "max_reduction_pct")}, indent=2))

if __name__ == "__main__": main()
