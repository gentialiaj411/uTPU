"""Evaluate one previously exported NPZ using bounded token chunks."""
from __future__ import annotations
import argparse, hashlib, json, math, tracemalloc
from pathlib import Path
import numpy as np
import torch
from tools.llama2c.metrics import tokenize_eval
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from tools.llama2c.torch_oracle import load_torch_model
from .run_15m import LOGIT_CHUNK, hardware_budget
from .runq_runtime import RunQIntegerLlama

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--tokens", type=int, default=50000)
    args = ap.parse_args(); tracemalloc.start()
    payload = np.load(args.artifact, allow_pickle=True)["payload"][0]; base_scales = payload.pop("dynamic_base_scales")
    ref = Llama2CModel(ROOT / "models/llama2c/stories15M.bin")
    tok = Tokenizer(ROOT / "models/llama2c/tokenizer.bin", ref.config.vocab_size)
    ids = tokenize_eval(tok, ROOT / "models/llama2c/eval_slice.txt", minimum=args.tokens)[:args.tokens + 1]
    float_model = load_torch_model(ref, "cuda" if torch.cuda.is_available() else "cpu")
    runtime = RunQIntegerLlama(payload, HERE / "luts")
    # Logit codes are in the export's dynamic base scale (as in run_15m.teacher_forced_metrics),
    # not the static calibration scale from earlier chunks.
    base_scale = float(base_scales["logits"])
    total_nll = total_kl = 0.0; matches = count = 0; seq = int(ref.config.seq_len)
    device = next(float_model.parameters()).device
    with torch.inference_mode():
      for start in range(0, len(ids)-1, seq):
        chunk = ids[start:min(start+seq+1, len(ids))]; inp = torch.tensor(chunk[:-1], dtype=torch.long, device=device)[None]
        fl = float_model(inp, inp).cpu().numpy()[0].astype(np.float64); target=np.asarray(chunk[1:],dtype=np.int64)
        cand = np.empty_like(fl); cand_arg=np.zeros(target.size,dtype=np.int64)
        for begin, codes in runtime.forward_sequence_logits_chunks(chunk[:-1], LOGIT_CHUNK):
          cand[:,begin:begin+codes.shape[1]] = codes.astype(np.float64)*base_scale
        fm=np.max(fl,1); fs=fm+np.log(np.sum(np.exp(fl-fm[:,None]),1)); cm=np.max(cand,1); cs=cm+np.log(np.sum(np.exp(cand-cm[:,None]),1))
        total_nll += float(np.sum(cs-cand[np.arange(target.size),target])); total_kl += float(np.sum(np.exp(fl-fs[:,None]) * ((fl-fs[:,None])-(cand-cs[:,None])))); matches += int(np.sum(np.argmax(fl,1)==np.argmax(cand,1))); count += target.size
    current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out={"config":args.artifact.stem,"artifact_sha256":hashlib.sha256(args.artifact.read_bytes()).hexdigest(),"eval_tokens":count,"perplexity":math.exp(total_nll/count),"teacher_forced_top1_agreement_pct":100*matches/count,"logit_kl_float_to_candidate":total_kl/count,"hardware_budget":hardware_budget(payload,4),"peak_tracemalloc_bytes":peak}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))

if __name__ == "__main__": main()
