"""Capture runtime-owned _quantize groups for act-quant RTL verification."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from tools.llama2c.reference import Llama2CModel
from firmware.host.numerics_v2 import runq_runtime as rr
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
from firmware.host.numerics_v2.export import export_runq_model
def hx(p,b,width=2):
    b=bytes(b); p.write_text("\n".join(f"{x:0{width}x}" for x in b)+"\n")
    return hashlib.sha256(b).hexdigest()
def save(root,idx,raw,q):
 p=f'g{idx:04d}'; rb=np.asarray(raw,dtype='<i4').tobytes(); cb=np.asarray(q.code,dtype=np.int8).tobytes(); sb=np.asarray(q.scale_multiplier,dtype='<i4').reshape(-1).tobytes()
 files={'raw':hx(root/f'{p}_r.hex',rb),'code':hx(root/f'{p}_c.hex',cb),'scale':hx(root/f'{p}_s.hex',sb)}; return {'id':p,'length':len(raw),'qmax':int(q.qmax),'group_size':int(q.group_size),'files':files}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'build/decode/golden_act_quant'); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True); records=[]; original=rr._quantize
 def wrap(raw,group_size=64,qmax=127):
  q=original(raw,group_size,qmax); v=np.asarray(raw,dtype=np.int64); n=v.shape[-1];
  lead=np.ndindex(v.shape[:-1]) if v.ndim>1 else [()]
  for ix in lead:
   row=v[ix] if ix else v
   for b in range(0,n,group_size):
    block=row[b:b+group_size]; qc=q.code[ix][b:b+group_size] if ix else q.code[b:b+group_size]; qs=q.scale_multiplier[ix+(b//group_size,)] if ix else q.scale_multiplier[b//group_size]
    records.append(save(a.output,len(records),block,type(q)(qc,np.asarray([qs]),min(group_size,n-b),q.qmax)))
  return q
 rr._quantize=wrap
 ref=Llama2CModel(ROOT/'models/llama2c/stories260K/stories260K.bin'); cal=json.loads((ROOT/'firmware/host/numerics_v2/calibration_260k.json').read_text()); exp=export_runq_model(ref,cal['activation_scales'],4,32,improved=True,mixed_precision={'qkv':8,'wo':8,'w2':8}); exp.pop('dynamic_base_scales'); runtime=RunQIntegerLlama(exp,ROOT/'firmware/host/numerics_v2/luts')
 for token in (1,2,3): runtime.forward_argmax(token,len(runtime.tokens),4096)
 runtime4=RunQIntegerLlama(exp,ROOT/'firmware/host/numerics_v2/luts',kv_bits=4)
 for token in (1,2,3): runtime4.forward_argmax(token,len(runtime4.tokens),4096)
 rng=np.random.default_rng(77); directed=[np.zeros(64,dtype=np.int64),np.r_[1,np.zeros(63,dtype=np.int64)],np.r_[-(2**31-1),np.zeros(63,dtype=np.int64)],np.r_[2**31-1,np.zeros(63,dtype=np.int64)],np.arange(172,dtype=np.int64)]
 # Exact RNE ties in rshift_rne(x*recip, 24): peak=254*2^j gives recip=2^(23-j), so x = odd*2^j ties.
 for j in range(4):
  pk=254<<j
  for t in range(6):
   vv=((2*rng.integers(-pk//(2<<j),pk//(2<<j),size=64)+1)<<j).astype(np.int64); vv[t]=pk if t%2==0 else -pk; directed.append(vv)
 # Codes exactly at the +-127 clamp and peaks near the int32 limit with small neighbours.
 for t in range(6):
  pk=int(rng.integers(1<<20,(1<<31)-1)); vv=rng.integers(-pk,pk+1,size=64).astype(np.int64); vv[t]=pk; vv[63-t]=-pk; directed.append(vv)
 n_real=len(records)
 for i in range(40):
  n=8 if i%2 else 48; pk=int(rng.integers(1,1<<int(rng.integers(2,31)))); v=rng.integers(-pk,pk+1,size=n).astype(np.int64); v[i%n]=pk
  if i%4==0: v=((2*rng.integers(-3,4,size=n)+1)*(1<<20)).astype(np.int64); v[0]=7<<21   # recip = 2^23/... exact ties at qmax 7
  q=original(v,n,7); records.append(save(a.output,len(records),v,type(q)(q.code,np.asarray(q.scale_multiplier).reshape(-1),n,7)))
 for i,v in enumerate(directed):
  q=original(v,64)
  for b in range(0,len(v),64):
   block=v[b:b+64]; qc=q.code[b:b+64]; qs=q.scale_multiplier[b//64]; records.append(save(a.output,len(records),block,type(q)(qc,np.asarray([qs]),len(block),q.qmax)))
 for i in range(300):
  n=64; v=rng.integers(-(2**31-1),2**31,size=n,dtype=np.int64); q=original(v,64); records.append(save(a.output,10000+i,v,q))
 n_rd=len(records)-n_real; manifest={'real_groups':n_real,'random_directed_groups':n_rd,'real_elements':sum(r['length'] for r in records[:n_real]),'random_directed_elements':sum(r['length'] for r in records[n_real:]),'records':records,'source':'wrapped runq_runtime._quantize'}; (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({k:manifest[k] for k in ('real_groups','random_directed_groups','real_elements','random_directed_elements')}))
if __name__=='__main__': main()
