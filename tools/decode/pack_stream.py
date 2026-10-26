"""Pack/unpack an exported RunQ payload into a sequential flash image."""
from __future__ import annotations
import argparse, hashlib, json, struct, zlib
from pathlib import Path
import numpy as np

MAGIC=b"UTPUDEC1"; VERSION=1; HEADER=262144; ALIGN=16
def _align(x): return (x + ALIGN - 1) // ALIGN * ALIGN
def _keep(path):
    return not any(x in path for x in ('weight_scale_', 'embedding_scale_','dynamic_base_scales'))
def _walk(value, path=""):
    if isinstance(value, np.ndarray): return {"__array__": path}
    if isinstance(value, dict): return {k:_walk(v, f"{path}.{k}" if path else k) for k,v in value.items() if _keep(f"{path}.{k}" if path else k)}
    if isinstance(value, list): return [_walk(v, f"{path}.{i}" if path else str(i)) for i,v in enumerate(value)]
    if isinstance(value, (int,float,bool,str)) or value is None: return value
    raise TypeError((path,type(value)))
def _arrays(value, path=""):
    out=[]
    if isinstance(value,np.ndarray): out.append((path,value))
    elif isinstance(value,dict):
        for k,v in value.items():
            p=f"{path}.{k}" if path else k
            if _keep(p): out.extend(_arrays(v,p))
    elif isinstance(value,list):
        for i,v in enumerate(value): out.extend(_arrays(v,f"{path}.{i}" if path else str(i)))
    return out
def _encode(path, arr):
    name=path.rsplit('.',1)[-1]
    if name=='weight' and int(np.max(np.abs(arr))) <= 7:
        x=(arr.astype(np.int8).reshape(-1) & 0x0f).astype(np.uint8); out=np.zeros((x.size+1)//2,dtype=np.uint8); out[:x.size//2] = x[0:2*(x.size//2):2] | (x[1:2*(x.size//2):2] << 4)
        if x.size&1: out[-1]=x[-1]
        return out.tobytes(), 'int4'
    if 'shift' in name: return arr.astype(np.uint8).tobytes(), 'u8'
    if 'multiplier' in name: return arr.astype('<i4').tobytes(), 'i32'
    return np.ascontiguousarray(arr).tobytes(order='C'), 'raw'
def _decode(raw, rec):
    enc=rec.get('encoding','raw'); shape=tuple(rec['shape']); n=int(np.prod(shape))
    if enc=='int4':
        b=np.frombuffer(raw,dtype=np.uint8); x=np.empty(n,dtype=np.int8); x[0::2]=(b&15).astype(np.int8); x[1::2]=((b>>4)&15).astype(np.int8); x[x>=8]-=16; return x.reshape(shape)
    dtype=np.dtype(rec['dtype']); return np.frombuffer(raw,dtype=np.dtype({'u8':'u1','i32':'<i4'}.get(enc,dtype))).reshape(shape).copy().astype(dtype,copy=False)
def _set(tree,path,value):
    bits=path.split('.'); cur=tree
    for b in bits[:-1]: cur=cur[int(b)] if b.isdigit() else cur[b]
    cur[int(bits[-1])] = value if bits[-1].isdigit() else value
    if not bits[-1].isdigit(): cur[bits[-1]]=value
def _replace(tree,path,value):
    bits=path.split('.'); cur=tree
    for b in bits[:-1]: cur=cur[int(b)] if b.isdigit() else cur[b]
    b=bits[-1]; cur[int(b)] = value if b.isdigit() else value
def pack_payload(payload, output, model):
    arrays=_arrays(payload)
    rank={'embedding':0,'layers':1,'classifier':2,'gamma_att':3,'gamma_ffn':4,'gamma_final':5}
    def key(item):
        p=item[0]; top=p.split('.')[0]; tail=p.rsplit('.',1)[-1]
        field={'weight':0,'multiplier':1,'shift':2,'groups':3}.get(tail,9)
        tensor=p.split('.')[2] if top=='layers' and len(p.split('.'))>2 else ''
        tr={'wq':0,'wk':1,'wv':2,'wo':3,'w1':4,'w3':5,'w2':6}.get(tensor,0)
        return (rank.get(top,8), p.split('.')[1] if top=='layers' else '', tr, field, p)
    arrays.sort(key=key); manifest=[]; cursor=HEADER; blobs=[]
    for path,arr in arrays:
        packed,enc=_encode(path,arr); raw=zlib.compress(packed, 1); cursor=_align(cursor); manifest.append({'path':path,'dtype':arr.dtype.str,'shape':arr.shape,'offset':cursor,'nbytes':len(raw),'codec':'zlib1','encoding':enc}); blobs.append((cursor,raw)); cursor += len(raw)
    tree=_walk(payload)
    header={'magic':MAGIC.decode(),'version':VERSION,'model':model,'header_bytes':HEADER,'alignment':ALIGN,'layout':'embedding; layers in index order wq,wk,wv,wo,w1,w3,w2; classifier; gamma; code planes then scale planes as exported','tree':tree,'arrays':manifest,'image_bytes':_align(cursor)}
    h=json.dumps(header,separators=(',',':')).encode();
    if len(h)>HEADER-8: raise ValueError('header too large')
    image=bytearray(_align(cursor)); image[:8]=MAGIC; image[8:12]=struct.pack('<I',len(h)); image[12:12+len(h)]=h
    for off,raw in blobs: image[off:off+len(raw)]=raw
    digest=hashlib.sha256(image).hexdigest(); header['sha256']=digest
    h=json.dumps(header,separators=(',',':')).encode(); image[8:12]=struct.pack('<I',len(h)); image[12:12+len(h)]=h
    output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(image); return header
def unpack_payload(image):
    data=image.read_bytes(); assert data[:8]==MAGIC
    n=struct.unpack('<I',data[8:12])[0]; header=json.loads(data[12:12+n]); tree=header['tree']
    def restore(node):
        if isinstance(node,dict) and '__array__' in node:
            rec=next(x for x in header['arrays'] if x['path']==node['__array__']); raw=data[rec['offset']:rec['offset']+rec['nbytes']]; raw=zlib.decompress(raw) if rec.get('codec')=='zlib1' else raw
            return _decode(raw,rec)
        if isinstance(node,dict): return {k:restore(v) for k,v in node.items()}
        if isinstance(node,list): return [restore(v) for v in node]
        return node
    return restore(tree),header
def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('pack'); p.add_argument('npz',type=Path); p.add_argument('image',type=Path); p.add_argument('--model',required=True)
    u=sub.add_parser('unpack'); u.add_argument('image',type=Path); u.add_argument('npz',type=Path)
    a=ap.parse_args()
    if a.cmd=='pack':
        payload=np.load(a.npz,allow_pickle=True)['payload'][0]; print(json.dumps(pack_payload(payload,a.image,a.model),indent=2))
    else:
        payload,header=unpack_payload(a.image); np.savez_compressed(a.npz,payload=np.asarray([payload],dtype=object)); print(json.dumps(header,indent=2))
if __name__=='__main__': main()
