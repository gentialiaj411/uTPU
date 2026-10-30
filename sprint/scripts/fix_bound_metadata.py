from pathlib import Path
p=Path('sprint/scripts/argmax_bound.py');s=p.read_text();s=s.replace("blocks=bound.reshape(-1,block).max(1);first=np.arange(len(blocks))*block", """blocks=bound.reshape(-1,block).max(1)
    if variant=='coeff':
     bc=coeff.reshape(-1,block,9).max(1)
     product=bc*factor[None,:]*xmax[None,:]
     nn=(product+(1<<32)-1)//(1<<32)+1
     blocks=((nn>>32)*INV127_Q32+(((nn&((1<<32)-1))*INV127_Q32+(1<<31))>>32)+1).sum(1)
     assert np.all(logits.astype(np.int64)<=np.repeat(blocks,block))
    first=np.arange(len(blocks))*block""")
p.write_text(s)
