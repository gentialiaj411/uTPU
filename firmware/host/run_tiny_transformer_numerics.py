"""Train/export the Phase-1 tiny transformer numerical artifact."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tiny_transformer_model import TinyDecoder
from tiny_transformer_reference import TinyTransformerConfig, integer_decoder_logits

HERE=Path(__file__).resolve().parent
def _fixture(name:str): return json.loads((HERE/'fixtures'/name).read_text(encoding='utf-8'))
def _q(a:np.ndarray):
    scale=max(float(np.max(np.abs(a)))/127.0,1e-8); return np.clip(np.rint(a/scale),-128,127).astype(np.int8),scale
def main():
 p=argparse.ArgumentParser(); p.add_argument('--corpus',type=Path,default=Path('build/tiny_transformer/input.txt')); p.add_argument('--steps',type=int,default=400);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--seed',type=int,default=20261013);p.add_argument('--output',type=Path,default=Path('bench/results/tiny_transformer_numerics.json'));a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.use_deterministic_algorithms(True)
 vocab=_fixture('tiny_transformer_vocab.json'); split=_fixture('tiny_transformer_split.json'); text=a.corpus.read_text(encoding='utf-8')
 if hashlib.sha256(a.corpus.read_bytes()).hexdigest()!=vocab['corpus_sha256']: raise SystemExit('corpus SHA-256 mismatch')
 stoi={c:i for i,c in enumerate(vocab['characters'])}; data=torch.tensor([stoi[c] for c in text],dtype=torch.long);cfg=TinyTransformerConfig(vocab=len(stoi))
 model=TinyDecoder(cfg.T,cfg.D,cfg.H,cfg.Dff,cfg.L,cfg.vocab);opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=0.01)
 for step in range(a.steps):
  starts=torch.randint(split['train_start'],split['train_end_exclusive']-cfg.T-1,(a.batch_size,));x=torch.stack([data[s:s+cfg.T] for s in starts]);y=torch.stack([data[s+1:s+cfg.T+1] for s in starts])
  opt.zero_grad();loss=F.cross_entropy(model(x).reshape(-1,cfg.vocab),y.reshape(-1));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
 # Full contiguous validation: every next-char target once, in fixed T windows.
 model.eval(); total_loss=correct=count=0
 with torch.no_grad():
  for start in range(split['validation_start'],split['validation_end_exclusive']-cfg.T-1,cfg.T):
   x=data[start:start+cfg.T][None];y=data[start+1:start+cfg.T+1][None];logits=model(x);total_loss+=float(F.cross_entropy(logits.reshape(-1,cfg.vocab),y.reshape(-1),reduction='sum'));correct+=int((logits.argmax(-1)==y).sum());count+=cfg.T
 state={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}; qstate={k:_q(v)[0] for k,v in state.items()}; scales={k:_q(v)[1] for k,v in state.items()}
 iloss=icorrect=icount=0; saturation_counts={'position_add':0,'residual_attention':0,'residual_ffn':0}
 starts=list(range(split['validation_start'],split['validation_end_exclusive']-cfg.T-1,cfg.T))
 for offset in range(0,len(starts),8):
  group=starts[offset:offset+8];x=np.stack([data[s:s+cfg.T].numpy() for s in group]);y=np.stack([data[s+1:s+cfg.T+1].numpy() for s in group]);logits,counts=integer_decoder_logits(x,qstate,cfg)
  logits_t=torch.from_numpy(logits.astype(np.float32));y_t=torch.from_numpy(y);iloss+=float(F.cross_entropy(logits_t.reshape(-1,cfg.vocab),y_t.reshape(-1),reduction='sum'));icorrect+=int((logits_t.argmax(-1)==y_t).sum());icount+=int(y.size)
  for key,value in counts.items(): saturation_counts[key]+=int(value)
 a.output.parent.mkdir(parents=True,exist_ok=True);payload={'status':'phase1_independent_integer_reference','seed':a.seed,'config':cfg.__dict__,'parameter_count':cfg.parameter_count,'corpus_sha256':vocab['corpus_sha256'],'split':split,'training_steps':a.steps,'float_validation':{'loss':total_loss/count,'bits_per_char':total_loss/count/np.log(2),'top1_accuracy':correct/count,'tokens':count},'integer_validation':{'loss':iloss/icount,'bits_per_char':iloss/icount/np.log(2),'top1_accuracy':icorrect/icount,'tokens':icount},'degradation':{'loss':iloss/icount-total_loss/count,'bits_per_char':(iloss/icount-total_loss/count)/np.log(2),'top1_accuracy':icorrect/icount-correct/count},'saturation_counts':saturation_counts,'limits':['independent NumPy integer reference; ISA-simulator and RTL equality are Phase 2/3 work']}
 a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(payload['degradation'],indent=2));torch.save({'state_dict':state,'qstate':qstate,'scales':scales,'config':cfg.__dict__},a.output.with_suffix('.pt'))
if __name__=='__main__': main()
