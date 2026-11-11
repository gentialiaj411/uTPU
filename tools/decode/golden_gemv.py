"""Deterministic RunQIntegerLlama._linear_high vector capture."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from tools.llama2c.reference import Llama2CModel
from firmware.host.numerics_v2 import runq_runtime as rr
from firmware.host.numerics_v2.runq_runtime import QValue,RunQIntegerLlama
from firmware.host.numerics_v2.export import export_runq_model
def write_hex(path,raw): path.write_text(raw.hex()+"\n",encoding='ascii'); return hashlib.sha256(raw).hexdigest()
def pack_codes(w,qmax):
 w=np.asarray(w,dtype=np.int8)
 if qmax==127:return w.tobytes()
 x=(w.reshape(-1)&15).astype(np.uint8); out=np.zeros((len(x)+1)//2,dtype=np.uint8); out[:len(x)//2]=x[0::2][:len(out)-len(x)%2]|(x[1::2][:len(out)-len(x)%2]<<4)
 if len(x)&1:out[-1]=x[-1]
 return out.tobytes()
def save_record(root,idx,a,record,out,begin,end):
 p=f'v{idx:05d}'; w=np.asarray(record['weight'][begin:end]); qmax=int(record['qmax']); groups=np.asarray(record['groups'],dtype='<i4'); wm=np.asarray(record['multiplier'][begin:end],dtype='<i8'); sh=np.asarray(record['shift'][begin:end],dtype='<i8'); files={}
 files['a']=write_hex(root/f'{p}_a.hex',np.asarray(a.code,dtype=np.int8).tobytes()); files['as']=write_hex(root/f'{p}_as.hex',np.asarray(a.scale_multiplier,dtype='<i4').tobytes()); files['w']=write_hex(root/f'{p}_w.hex',pack_codes(w,qmax)); files['wm']=write_hex(root/f'{p}_wm.hex',wm.tobytes()); files['ws']=write_hex(root/f'{p}_ws.hex',sh.tobytes()); files['gr']=write_hex(root/f'{p}_gr.hex',groups.tobytes()); files['o']=write_hex(root/f'{p}_o.hex',np.asarray(out[begin:end],dtype='<i4').tobytes())
 return {'id':p,'rows':end-begin,'k':w.shape[1],'qmax':qmax,'groups':groups.tolist(),'files':files,'output_elements':end-begin}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'build/decode/golden_gemv'); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 ref=Llama2CModel(ROOT/'models/llama2c/stories260K/stories260K.bin'); cal=json.loads((ROOT/'firmware/host/numerics_v2/calibration_260k.json').read_text()); exp=export_runq_model(ref,cal['activation_scales'],4,32,improved=True,mixed_precision={'qkv':8,'wo':8,'w2':8}); exp.pop('dynamic_base_scales'); runtime=RunQIntegerLlama(exp,ROOT/'firmware/host/numerics_v2/luts'); records=[]; original=rr._linear_high; maxima={'act_scale_multiplier':0,'small':0,'shift':0}
 def wrapped(value,record,output_begin=0,output_end=None):
  out=original(value,record,output_begin,output_end); begin=0 if output_begin is None else output_begin; end=record['weight'].shape[0] if output_end is None else output_end; maxima['act_scale_multiplier']=max(maxima['act_scale_multiplier'],int(np.max(np.abs(value.scale_multiplier)))); maxima['small']=max(maxima['small'],int(np.max(np.abs(record['multiplier'][begin:end]>>16)))); maxima['shift']=max(maxima['shift'],int(np.max(record['shift'][begin:end])))
  for b in range(begin,end,16): records.append(save_record(a.output,len(records),value,record,out,b,min(b+16,end)))
  return out
 rr._linear_high=wrapped
 for token in (1,2,3): runtime.forward_argmax(token,len(runtime.tokens),4096)
 rng=np.random.default_rng(123); random=[]; edge=np.array([-128,-127,-8,-7,-1,0,1,7,127],dtype=np.int8); i=0
 while len(random)<300:
  i+=1; k=64 if i%3 else 172; qmax=127 if i%2 else 7; rows=16
  begins=list(range(0,k,32)); groups=np.asarray([begins,[min(b+32,k) for b in begins]],dtype=np.int32)
  weight=rng.choice(edge,size=(rows,k)) if i%4 else rng.integers(-128,128,size=(rows,k)).astype(np.int8)
  if qmax==7: weight=np.clip(weight,-8,7).astype(np.int8)
  if i%7==0: weight[:,:32]=0
  multip=rng.integers(1<<16,(1<<31)-1,size=(rows,groups.shape[1]),dtype=np.int64); shifts=rng.integers(16,43,size=(rows,groups.shape[1]),dtype=np.int64)
  ac=(rng.choice(edge,size=k) if i%5 else rng.integers(-128,128,size=k)).astype(np.int8)
  if i%11==0: ac[:]=0
  scales=rng.integers(1,1<<17,size=((k+63)//64,),dtype=np.int64).astype(np.int32)
  rec={'weight':weight,'groups':groups,'multiplier':multip,'shift':shifts,'qmax':qmax}; q=QValue(ac,scales,64,127)
  try: out=original(q,rec)
  except OverflowError: continue
  random.append(save_record(a.output,100000+len(random),q,rec,out,0,rows))
 # Directed RNE ties. shift=17 makes s=rshift_rne(acc,1); acc=127*(4j+3) puts the tie exactly on a
 # div127 rounding boundary, so half-up vs ties-to-even changes the output. A multiplier with low
 # half-word 0x8000 makes small=rshift_rne(multiplier,16) itself a tie.
 directed=0
 for j in range(40):
  k=64; rows=16; groups=np.asarray([[0,32],[32,64]],dtype=np.int32); weight=np.zeros((rows,k),dtype=np.int8); ac=np.zeros(k,dtype=np.int8)
  ac[0]=127; ac[33]=1; sign=-1 if j%2 else 1
  for r in range(rows): weight[r,0]=sign*(4*((j*rows+r)%31)+3); weight[r,33]=sign*rng.integers(1,127)
  half=np.int64(1<<15); multip=np.empty((rows,2),dtype=np.int64); multip[:,0]=1<<16
  multip[:,1]=(2*rng.integers(1,1<<14,size=rows)+(j%2))*(1<<16)+half
  shifts=np.empty((rows,2),dtype=np.int64); shifts[:,0]=17; shifts[:,1]=rng.integers(16,24,size=rows)
  rec={'weight':weight,'groups':groups,'multiplier':multip,'shift':shifts,'qmax':127}; q=QValue(ac,np.ones(1,dtype=np.int32),64,127)
  random.append(save_record(a.output,100000+len(random),q,rec,original(q,rec),0,rows)); directed+=1
 manifest={'model':'stories260K','real_matmuls':len(records),'random_matmuls':len(random),'real_output_elements':sum(r['output_elements'] for r in records),'random_output_elements':sum(r['output_elements'] for r in random),'observed_max_widths':maxima,'records':records,'random_records':random,'source':'wrapped firmware.host.numerics_v2.runq_runtime._linear_high'}; (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({k:manifest[k] for k in ('real_matmuls','random_matmuls','real_output_elements','random_output_elements')}))
if __name__=='__main__': main()
