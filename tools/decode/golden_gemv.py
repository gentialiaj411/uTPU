"""Capture GEMV vectors by wrapping numerics-v2's own _linear_high."""
from __future__ import annotations
import argparse,hashlib,json,struct
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from tools.llama2c.reference import Llama2CModel
from firmware.host.numerics_v2 import runq_runtime as rr
from firmware.host.numerics_v2.runq_runtime import QValue,RunQIntegerLlama
from firmware.host.numerics_v2.export import export_runq_model
def hexfile(path, raw): path.write_text(raw.hex()+"\n",encoding='ascii'); return hashlib.sha256(raw).hexdigest()
def pack_codes(w,qmax):
 if qmax==127:return np.asarray(w,dtype=np.int8).tobytes()
 x=(np.asarray(w,dtype=np.int8).reshape(-1)&15).astype(np.uint8); out=np.zeros((len(x)+1)//2,dtype=np.uint8); out[:len(x)//2]=x[0:2*(len(x)//2):2]|(x[1:2*(len(x)//2):2]<<4)
 if len(x)&1:out[-1]=x[-1]
 return out.tobytes()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'build/decode/golden_gemv'); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 ref=Llama2CModel(ROOT/'models/llama2c/stories260K/stories260K.bin'); cal=json.loads((ROOT/'firmware/host/numerics_v2/calibration_260k.json').read_text()); exp=export_runq_model(ref,cal['activation_scales'],4,32,improved=True,mixed_precision={'qkv':8,'wo':8,'w2':8}); exp.pop('dynamic_base_scales'); runtime=RunQIntegerLlama(exp,ROOT/'firmware/host/numerics_v2/luts'); records=[]
 original=rr._linear_high
 def wrapped(value,record,output_begin=0,output_end=None):
  out=original(value,record,output_begin,output_end); idx=len(records); prefix=f'v{idx:04d}'; w=record['weight']; qmax=int(record['qmax']); rows=w.shape[0] if output_end is None else output_end-output_begin; ws=record['multiplier'][output_begin:output_end if output_end is not None else record['multiplier'].shape[0]]; act=np.asarray(value.code,dtype=np.int8); scales=np.asarray(value.scale_multiplier,dtype=np.int32); files={}
  files['activation_codes']=hexfile(a.output/f'{prefix}_a.hex',act.tobytes()); files['activation_scales']=hexfile(a.output/f'{prefix}_as.hex',scales.tobytes()); files['weight_codes']=hexfile(a.output/f'{prefix}_w.hex',pack_codes(w[:rows],qmax)); files['weight_scales']=hexfile(a.output/f'{prefix}_ws.hex',np.asarray(ws,dtype=np.int64).tobytes()); files['output']=hexfile(a.output/f'{prefix}_o.hex',np.asarray(out,dtype='<i4').tobytes()); records.append({'id':prefix,'rows':rows,'k':w.shape[1],'qmax':qmax,'files':files,'activation_shape':list(act.shape),'output_shape':list(out.shape)})
  return out
 rr._linear_high=wrapped
 for token in (1,2,3): runtime.forward_argmax(token,len(runtime.tokens),4096)
 rng=np.random.default_rng(123); base=exp['layers'][0]['wq']; random_records=[]
 for i in range(300):
  k=base['weight'].shape[1]; rows=16; qmax=int(base['qmax']); codes=rng.choice(np.array([-128,-127,-8,-7,0,7,127],dtype=np.int8),size=(1,k)); scales=np.ones((1,(k+63)//64),dtype=np.int32); q=QValue(codes,scales,64,127); out=original(q,base,0,rows); random_records.append({'rows':rows,'k':k,'qmax':qmax,'output_sha256':hashlib.sha256(np.asarray(out,dtype='<i4').tobytes()).hexdigest()})
 manifest={'model':'stories260K','matmuls':len(records),'random_edge_matmuls':len(random_records),'records':records,'random_records':random_records,'source':'RunQIntegerLlama._linear_high wrapper'}; (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'matmuls':len(records),'random':len(random_records)}))
if __name__=='__main__': main()
