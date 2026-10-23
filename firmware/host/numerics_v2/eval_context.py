"""Bounded context/KV precision evaluation for an exported stories15M model."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np, torch
from tools.llama2c.metrics import tokenize_eval
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from tools.llama2c.torch_oracle import load_torch_model
from .run_15m import LOGIT_CHUNK
from .runq_runtime import RunQIntegerLlama
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--context',type=int,required=True); ap.add_argument('--kv-bits',type=int,choices=(4,8),required=True); ap.add_argument('--tokens',type=int,default=50000); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    payload=np.load(a.artifact,allow_pickle=True)['payload'][0]; payload.pop('dynamic_base_scales',None)
    ref=Llama2CModel(ROOT/'models/llama2c/stories15M.bin'); tok=Tokenizer(ROOT/'models/llama2c/tokenizer.bin',ref.config.vocab_size)
    ids=tokenize_eval(tok,ROOT/'models/llama2c/eval_slice.txt',minimum=a.tokens+1)[:a.tokens+1]
    float_model=load_torch_model(ref,'cuda' if torch.cuda.is_available() else 'cpu'); device=next(float_model.parameters()).device; runtime=RunQIntegerLlama(payload,HERE/'luts',kv_bits=a.kv_bits)
    logit_scale=float(json.loads((HERE/'calibration_15m.json').read_text())['activation_scales']['logits']); nll=0.; float_nll=0.; matches=0; count=0; t0=time.time()
    with torch.inference_mode():
      for start in range(0,len(ids)-1,a.context):
        chunk=ids[start:min(start+a.context+1,len(ids))]; target=np.asarray(chunk[1:],dtype=np.int64)
        inp=torch.tensor(chunk[:-1],dtype=torch.long,device=device)[None]; fl=float_model(inp,inp).cpu().numpy()[0].astype(np.float64)
        cand=np.empty_like(fl)
        for begin,codes in runtime.forward_sequence_logits_chunks(chunk[:-1],LOGIT_CHUNK): cand[:,begin:begin+codes.shape[1]]=codes.astype(np.float64)*logit_scale
        fm=np.max(fl,1); fs=fm+np.log(np.sum(np.exp(fl-fm[:,None]),1)); cm=np.max(cand,1); cs=cm+np.log(np.sum(np.exp(cand-cm[:,None]),1))
        nll += float(np.sum(cs-cand[np.arange(target.size),target])); float_nll += float(np.sum(fs-fl[np.arange(target.size),target])); matches += int(np.sum(np.argmax(fl,1)==np.argmax(cand,1))); count += target.size
    out={'artifact_sha256':hashlib.sha256(a.artifact.read_bytes()).hexdigest(),'context_length':a.context,'kv_bits':a.kv_bits,'eval_tokens':count,'float_perplexity':math.exp(float_nll/count),'perplexity':math.exp(nll/count),'teacher_forced_top1_agreement_pct':100*matches/count,'elapsed_seconds':time.time()-t0}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
