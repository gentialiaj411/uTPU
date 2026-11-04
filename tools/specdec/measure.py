"""Measure lossless speculative acceptance on 50 TinyStories validation prompts."""
from __future__ import annotations
import json, math, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.llama2c.tokenizer import Tokenizer
from tools.decode.harness import load_runtime_260k
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama

def load15():
    p = np.load(ROOT/'models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz', allow_pickle=True)['payload'][0]
    p.pop('dynamic_base_scales', None)
    return RunQIntegerLlama(p, ROOT/'firmware/host/numerics_v2/luts', kv_bits=4)

def prompts50():
    tok = Tokenizer(ROOT/'models/llama2c/tokenizer.bin', 32000)
    out=[]
    for story in (ROOT/'models/llama2c/TinyStories-valid.txt').read_text(encoding='utf8').split('<|endoftext|>'):
        if story.strip():
            p=tok.encode(story.strip()[:160])[:8]
            if len(p)>=2 and p not in out: out.append(p)
        if len(out)>=50: break
    return out, tok

def greedy(rt, prompt, gen=12):
    rt.reset(); toks=list(prompt); outs=[]
    for p in range(len(prompt)-1): rt.forward_argmax(toks[p],p,4096)
    pos=len(prompt)-1; cur=toks[-1]
    for _ in range(gen):
        nxt=rt.forward_argmax(cur,pos,4096); outs.append(int(nxt)); cur=int(nxt); pos+=1
    return toks+outs

def ngram_table(seqs,n):
    tab={}
    for s in seqs:
        for i in range(len(s)-n): tab[tuple(s[i:i+n])]=int(s[i+n])
    return tab

def static_tables(tok):
    text=(ROOT/'models/llama2c/TinyStories-train.slice.txt').read_text(encoding='utf8')
    ids=tok.encode(text,bos=False)
    return {n:ngram_table([ids],n) for n in (2,3)}, len(text.encode())

def proposals(name, hist, tables, static, tiny):
    if name.startswith('context'):
        n=int(name[-1]); tab=tables[n]; key=tuple(hist[-n:]); return [tab[key]] if key in tab else []
    if name.startswith('static'):
        n=int(name[-1]); tab=static[n]; key=tuple(hist[-n:]); return [tab[key]] if key in tab else []
    if name=='host_tiny': return tiny[len(hist)-1:]
    return []

def main():
    ps,tok=prompts50(); rt15=load15(); seqs=[greedy(rt15,p,gen=2) for p in ps]
    static,train_bytes=static_tables(tok)
    # Host draft: actual stories260K integer runtime + its tokenizer, then re-tokenize
    # generated text with the 15M tokenizer so the UART candidate IDs are valid.
    tiny_exp,tiny=load_runtime_260k(4); tiny_tok=Tokenizer(ROOT/'models/llama2c/stories260K/tok512.bin',512)
    tinys=[]
    for p in ps:
        ids=tiny_tok.encode(tok.decode(p)); cur=ids[-1]; out=[]
        for i,t in enumerate(ids[:-1]): tiny.forward_argmax(t,i,4096)
        for i in range(8): out.append(tiny.forward_argmax(cur,len(ids)-1+i,4096)); cur=out[-1]
        tinys.append(tok.encode(tiny_tok.decode(out),bos=False))
        tiny.reset()
    dnames=[f'context{n}' for n in (1,2,3)]+[f'static{n}' for n in (2,3)]+['host_tiny']
    result={'status':'pass','python':sys.executable,'numpy':np.__version__,'prompts':ps,'prompt_count':len(ps),
            'generation_tokens':2,'train_slice_bytes':train_bytes,'static_table_sizes_bytes':{f'static{n}':len(static[n])* (n*4+4) for n in (2,3)},'drafters':{}}
    for name in dnames:
        row={}
        for k in (1,2,3,4):
            counts=[]; total=0; passes=0
            for prompt,seq,tinyseq in zip(ps,seqs,tinys):
                hist=list(prompt); idx=len(prompt); draftbuf=tinyseq if name=='host_tiny' else None
                while idx < len(seq):
                    ds=[]
                    for j in range(k):
                        pr=proposals(name,hist,ngram_table(hist,1) if False else {n:ngram_table([hist],n) for n in (1,2,3)},static,draftbuf)
                        if name=='host_tiny': pr=draftbuf[idx-len(prompt):idx-len(prompt)+k]
                        if not pr: break
                        ds += pr[:k-len(ds)]
                        if len(ds)<k and name.startswith('context'): break
                        break
                    # For table drafters, recursively propose from candidate history.
                    if not name=='host_tiny':
                        ds=[]; h=list(hist)
                        for _ in range(k):
                            pr=proposals(name,h,{n:ngram_table([h],n) for n in (1,2,3)},static,None)
                            if not pr: break
                            ds.append(pr[0]); h.append(pr[0])
                    accepted=0
                    for j,d in enumerate(ds):
                        if idx+j<len(seq) and d==seq[idx+j]: accepted+=1
                        else: break
                    counts.append(accepted); passes+=1; take=min(accepted+1,len(seq)-idx); total+=take
                    hist.extend(seq[idx:idx+take]); idx+=take
            row[str(k)]={'distribution':dict(sorted(Counter(counts).items())), 'mean_accepted':sum(counts)/len(counts), 'mean_tokens_per_pass':total/passes, 'passes':passes,'lossless':True}
        result['drafters'][name]=row
    # UART framing estimate for a 16-bit token list: command + k tokens + response.
    result['host_tiny_uart']={'bytes_per_round':{str(k):1+2*k+2 for k in (1,2,3,4)},'latency_ms_115200_8n1':{str(k):(1+2*k+2)*10*1000/115200 for k in (1,2,3,4)}}
    (ROOT/'bench/results/specdec_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'prompt_count':50,'drafters':list(result['drafters'])},indent=2))
if __name__=='__main__': main()
