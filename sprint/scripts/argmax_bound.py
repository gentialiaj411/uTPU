import os,sys,time,json,csv,math
from pathlib import Path
R=Path(__file__).resolve().parents[2];os.chdir(R);sys.path.insert(0,str(R))
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from bound_math import coefficient_scaled_bound
from tools.decode.pack_decode_image import load_export_15m
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama,_linear_high,_small_multiplier
from firmware.host.numerics_v2.fixed import rshift_rne,INV127_Q32
D=R/'sprint/data';exp=load_export_15m();rt=RunQIntegerLlama(exp,R/'firmware/host/numerics_v2/luts',kv_bits=4);rec=exp['classifier'];W=rec['weight'].astype(np.int64).reshape(32000,9,32);l1=np.abs(W).sum(2);l2=(W*W).sum(2);M=_small_multiplier(rec['multiplier']);SH=rec['shift'].astype(np.int64)-16
assert np.all((SH>=0)&(SH<=63)) and np.all(M>=0)
# Immutable precomputed norm/scale coefficients; ceil coefficient rounding is outward.
coeff=np.array([[(int(l1[r,g])*int(M[r,g])*(1<<32)+(1<<int(SH[r,g]))-1)//(1<<int(SH[r,g])) for g in range(9)] for r in range(32000)],dtype=np.int64)
assert coeff.max() < 2**32
old_prompts=json.loads((R.parent/'uTPU-research/research/data/prompts.json').read_text())
prompts=[];first_indices=[]
for i,prompt in enumerate(old_prompts):
 if prompt not in prompts:prompts.append(prompt);first_indices.append(i)
reused=len(prompts)
from tools.llama2c.tokenizer import Tokenizer
tokenizer=Tokenizer(R/'models/llama2c/tokenizer.bin',32000)
for story in (R/'models/llama2c/TinyStories-valid.txt').read_text(encoding='utf-8').split('<|endoftext|>'):
 if story.strip():
  candidate=tokenizer.encode(story.strip()[:160])[:8]
  if len(candidate)==8 and candidate not in prompts:prompts.append(candidate)
 if len(prompts)==50:break
assert len(prompts)==50 and len({tuple(x) for x in prompts})==50
(D/'argmax_prompts.json').write_text(json.dumps(prompts))
records=[]
if (D/'argmax_repeated_prefix_dataset.csv').exists():
 mapping={old:new for new,old in enumerate(first_indices)}
 with (D/'argmax_repeated_prefix_dataset.csv').open() as source:
  for row in csv.DictReader(source):
   if int(row['prompt']) in mapping:
    row['prompt']=mapping[int(row['prompt'])];records.append(row)
deadline=time.monotonic()+90*60
for pi,prompt in enumerate(prompts[:50]):
 if pi<reused:continue
 rt.reset()
 for pos,t in enumerate(prompt[:-1]):rt.forward_argmax(t,pos)
 token=prompt[-1]
 for step in range(120):
  if time.monotonic()>deadline:break
  pos=len(prompt)-1+step;hidden=rt._forward_hidden(token,pos);codes=hidden.code.astype(np.int64).reshape(9,32);factor=hidden.scale_multiplier[np.arange(9)//2].astype(np.int64)
  logits=_linear_high(hidden,rec);best=int(np.argmax(logits));maxlog=int(logits[best])
  xmax=np.abs(codes).max(1);x2=(codes*codes).sum(1)
  bounds=[]
  for variant in ['l1','l2','coeff']:
   if variant=='coeff':
    n=coefficient_scaled_bound(coeff,factor[None,:],xmax[None,:])
   else:
    dot=l1*xmax[None,:] if variant=='l1' else np.ceil(np.sqrt(l2*x2[None,:])).astype(np.int64)
    # Integer-correct sqrt rounding: float proposal checked and corrected with exact products.
    if variant=='l2':
     square=l2*x2[None,:];dot+=dot*dot<square;dot-=((dot-1)*(dot-1)>=square)&(dot>0)
    assert int(dot.max())*int(factor.max())*int(M.max())<2**63
    n=rshift_rne(dot*factor[None,:]*M,SH)
   # Overflow-free reciprocal envelope. q estimate uses limbs, at most one correction.
   hi=n>>32;lo=n&((1<<32)-1);bound=((hi*INV127_Q32)+((lo*INV127_Q32+(1<<31))>>32)+1).sum(1)
   assert np.all(logits.astype(np.int64)<=bound), 'UNSOUND bound'
   for block in [16,64,256]:
    blocks=bound.reshape(-1,block).max(1)
    if variant=='coeff':
     bc=coeff.reshape(-1,block,9).max(1)
     nn=coefficient_scaled_bound(bc,factor[None,:],xmax[None,:])
     blocks=((nn>>32)*INV127_Q32+(((nn&((1<<32)-1))*INV127_Q32+(1<<31))>>32)+1).sum(1)
     assert np.all(logits.astype(np.int64)<=np.repeat(blocks,block))
    first=np.arange(len(blocks))*block
    for ordering in ['natural','upper_first']:
     order=np.arange(len(blocks)) if ordering=='natural' else np.argsort(-blocks,kind='stable');current=-(1<<63);idx=32000;read=0;seeks=0;previous=-2
     for bi in order:
      start=int(first[bi]);ub=int(blocks[bi])
      if ub<current or (ub==current and start>idx):continue
      chunk=logits[start:start+block];j=start+int(np.argmax(chunk));v=int(logits[j]);read+=block;seeks+=bi!=previous+1;previous=int(bi)
      if v>current or (v==current and j<idx):current=v;idx=j
     assert idx==best and current==maxlog
     # coeff: one u32/group/block; L1/L2 require norms plus old row scales (streamed).
     metadata=len(blocks)*9*4 if variant=='coeff' else 32000*9*7
     bytecount=read*189+metadata # existing 144 codes +45 row-scale bytes, plus bound metadata
     for rate in [.25,.5]:
      cycles=(11633661-6048000+bytecount)/rate+seeks*80 # 6Bh:8 opcode+24 addr+8dummy =40 SCLK=80 core clocks
      records.append(dict(prompt=pi,step=step,variant=variant,block=block,ordering=ordering,rate=rate,skipped_fraction=1-read/32000,rows_read=read,seeks=seeks,bound_metadata=metadata,bytes_per_token=11633661-6048000+bytecount,estimated_tps=50e6/cycles,identical=True))
  token=best
 with (D/'argmax_sweep.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
 (D/'argmax_progress.json').write_text(json.dumps(dict(prompts=pi+1,steps=(pi+1)*120,records=len(records))))
 if time.monotonic()>deadline:break

(D/"argmax_unique_complete.json").write_text(json.dumps(dict(unique_prompts=len(prompts),records=len(records))))
