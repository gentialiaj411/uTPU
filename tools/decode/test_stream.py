"""Pack/unpack differential test for both model images."""
from __future__ import annotations
import argparse, numpy as np
from pathlib import Path
from .pack_stream import pack_payload, unpack_payload
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
def check(npz, image, lut):
    p=np.load(npz,allow_pickle=True)['payload'][0]; p.pop('dynamic_base_scales',None)
    pack_payload(p,image,image.stem)
    q,_=unpack_payload(image)
    x=RunQIntegerLlama(p,lut); y=RunQIntegerLlama(q,lut); tokens=np.arange(8,dtype=np.int64)
    a=np.concatenate([v for _,v in x.forward_sequence_logits_chunks(tokens,4096)],1)
    b=np.concatenate([v for _,v in y.forward_sequence_logits_chunks(tokens,4096)],1)
    assert np.array_equal(a,b), np.max(np.abs(a-b)); return image.stat().st_size
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model260k',type=Path,required=True); ap.add_argument('--model15m',type=Path,required=True); ap.add_argument('--work',type=Path,required=True); a=ap.parse_args(); a.work.mkdir(parents=True,exist_ok=True); lut=Path('firmware/host/numerics_v2/luts')
    s260=check(a.model260k,a.work/'stories260k.flash',lut); s15=check(a.model15m,a.work/'stories15m.flash',lut)
    print({'stories260K_bytes':s260,'stories15M_bytes':s15,'stories15M_fits':s15<=14155776,'tokens':8,'sequential':True})
if __name__=='__main__': main()
