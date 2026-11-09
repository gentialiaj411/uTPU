"""Measured RunQIntegerLlama evaluation on the fixed document-preserving text."""
from __future__ import annotations
import argparse, json, math, sys, time, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--tokenizer',type=Path,required=True); ap.add_argument('--text',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--precision',default='from-artifact'); a=ap.parse_args()
 payload=np.load(a.artifact,allow_pickle=True)['payload'][0]; scale=float(payload['dynamic_base_scales']['logits']); payload.pop('dynamic_base_scales',None)
 ref=Llama2CModel(a.checkpoint); raw=a.text.read_bytes(); parts=[x for x in raw.decode('utf-8').split('<|endoftext|>') if x]
 if a.tokenizer.name=='tokenizer.bin': tok=Tokenizer(a.tokenizer,ref.config.vocab_size); docs=[tok.encode(x,bos=False,eos=False) for x in parts]
 else:
  import sentencepiece as spm; p=spm.SentencePieceProcessor(model_file=str(a.tokenizer)); docs=[p.encode(x,out_type=int) for x in parts]
 rt=RunQIntegerLlama(payload,ROOT/'firmware/host/numerics_v2/luts',kv_bits=4); nll=0.; n=0; t0=time.time()
 for di,doc in enumerate(docs):
  for s in range(0,len(doc)-1,128):
   chunk=doc[s:min(s+129,len(doc))]; target=np.asarray(chunk[1:],dtype=np.int64)
   if not target.size: continue
   codes=np.empty((len(target),ref.config.vocab_size),dtype=np.float64)
   for begin,x in rt.forward_sequence_logits_chunks(chunk[:-1],1024): codes[:,begin:begin+x.shape[1]]=x*scale
   m=np.max(codes,1); lse=m+np.log(np.sum(np.exp(codes-m[:,None]),1)); nll+=float(np.sum(lse-codes[np.arange(len(target)),target])); n+=len(target)
   if n and n%10000<128: print('tokens',n,'seconds',time.time()-t0,flush=True)
 out={'status':'measured','checkpoint':str(a.checkpoint),'artifact':str(a.artifact),'artifact_sha256':hashlib.sha256(a.artifact.read_bytes()).hexdigest(),'tokenizer':str(a.tokenizer),'eval_tokens':n,'validation_bytes_scored':len(raw),'nll_nats':nll,'bits_per_byte':nll/math.log(2)/len(raw),'kv_bits':4,'precision':a.precision,'group_size':32,'protocol':'fixed eval_text.txt; split on <|endoftext|>; non-overlapping 128-token windows reset at each document; denominator is all fixed-text UTF-8 bytes','elapsed_seconds':time.time()-t0}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
