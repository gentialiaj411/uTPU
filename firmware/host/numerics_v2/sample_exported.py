"""Generate the fixed three prompts from an exported integer artifact."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from tools.llama2c.torch_oracle import load_torch_model
from .runq_runtime import RunQIntegerLlama

ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    payload=np.load(a.artifact,allow_pickle=True)['payload'][0]; payload.pop('dynamic_base_scales',None)
    ref=Llama2CModel(ROOT/'models/llama2c/stories15M.bin'); tok=Tokenizer(ROOT/'models/llama2c/tokenizer.bin',ref.config.vocab_size)
    prompt_texts=['\n<s>\nSpot. Spot saw the','climb. She climbed trees,','?"\nRoxy told Billy about']
    prompts=[np.asarray(tok.encode(p,bos=False,eos=False),dtype=np.int64) for p in prompt_texts]
    runtime=RunQIntegerLlama(payload,HERE/'luts'); float_model=load_torch_model(ref,'cuda' if torch.cuda.is_available() else 'cpu'); device=next(float_model.parameters()).device
    rows=[]
    for prompt in prompts:
        runtime.reset(); cand=[]; next_token=None
        for pos,t in enumerate(prompt): next_token=runtime.forward_argmax(int(t),pos,4096)
        for step in range(64-len(prompt)):
            cand.append(int(next_token)); next_token=runtime.forward_argmax(int(next_token),len(prompt)+step,4096)
        with torch.inference_mode():
            seq=torch.tensor(prompt,dtype=torch.long,device=device)[None]; out=float_model(seq,seq); token=int(torch.argmax(out[0,-1]).item()); ft=list(prompt)
            for _ in range(64-len(prompt)):
                ft.append(token); z=float_model(torch.tensor(ft,dtype=torch.long,device=device)[None],torch.tensor(ft,dtype=torch.long,device=device)[None]); token=int(torch.argmax(z[0,-1]).item())
        rows.append({'prompt':tok.decode(prompt),'float':tok.decode(ft),'integer':tok.decode(list(prompt)+cand)})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__': main()
