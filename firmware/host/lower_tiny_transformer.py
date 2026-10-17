"""Deterministic Boolean program/image lowering for the fixed tiny decoder."""
from __future__ import annotations
import hashlib, struct
from dataclasses import dataclass
from typing import Any
import numpy as np
from isa_encoder import (IsaConfig, encodeHalt, encodeLoadInputs, encodeLoadWeights,
                         encodeRequantParamsVector, encodeRun, encode_vec_vadd_rq,
                         encode_vec_vlayout, encode_vec_vmatmul, encode_vec_vpos_add,
                         encode_vec_vrmsnorm, encode_vec_vsoftmax)

CFG=IsaConfig(address_width=16,compute_data_width=8,descriptor_isa=True,vector_unit=True)
N,T,D,H,DH,DFF=8,32,48,3,16,192

@dataclass(frozen=True)
class Map:
    token:int=0; pos:int=1728; qkv:int=2496; out:int=5952; ff1:int=7104; ff2:int=11712
    gamma:int=16320; records:int=16392; x:int=16968; r:int=17736; qkv_a:int=18504; score:int=20808; head:int=21832; nextx:int=22088; hidden:int=22856; vec:int=25928; scratch:int=26256
M=Map()

def _words(blob:bytes)->list[int]: return list(struct.unpack('<'+'H'*(len(blob)//2),blob))
def _pack(values:np.ndarray)->list[int]:
    a=np.asarray(values,dtype=np.int8).reshape(-1); return [(int(a[i])&255)|((int(a[i+1])&255)<<8) for i in range(0,len(a),2)]
def _tile_words(weight:np.ndarray)->list[int]:
    out=[]; w=np.asarray(weight,np.int8)
    for ob in range((w.shape[0]+7)//8):
        for ib in range((w.shape[1]+7)//8):
            tile=np.zeros((8,8),np.int8); tile[:min(8,w.shape[0]-8*ob),:min(8,w.shape[1]-8*ib)]=w[8*ob:8*ob+8,8*ib:8*ib+8]
            out.extend(_pack(tile))
    return out
def _vec(subop,dst,src,param,length,*,causal=False,head=0):
    funcs=(encode_vec_vadd_rq,encode_vec_vrmsnorm,encode_vec_vpos_add,encode_vec_vsoftmax,encode_vec_vmatmul,encode_vec_vlayout)
    if subop==3:return funcs[subop](dst,src,param,length,head_id=head,cfg=CFG)
    if subop==4:return funcs[subop](dst,src,param,length,causal=causal,cfg=CFG)
    return funcs[subop](dst,src,param,length,cfg=CFG)
def _record_mat(b,m,k,n,tr,au,ca,mult,shift): return [5,b,(m-1)|((k-1)<<5)|((n-1)<<10)|(int(tr)<<15),k|((k if tr else n)<<8),int(au)|(int(ca)<<2),mult,shift,0]

def build_weight_image(state:dict[str,np.ndarray], schema:dict[str,Any])->list[int]:
    """Static A5 image: tiled linear weights plus VEC tables/records."""
    image=[0]*16968
    def write(base,data): image[base:base+len(data)]=data
    write(M.token,_pack(state['token.weight']));write(M.pos,_pack(state['pos.weight']));write(M.qkv,_tile_words(state['blocks.0.qkv.weight']));write(M.out,_tile_words(state['blocks.0.out.weight']));write(M.ff1,_tile_words(state['blocks.0.ff1.weight']));write(M.ff2,_tile_words(state['blocks.0.ff2.weight']))
    write(M.gamma,_pack(np.concatenate((state['blocks.0.n1.weight'],state['blocks.0.n2.weight'],state['final.weight']))))
    # Records are fixed-address and fully zero-padded; dynamic tensor regions never alias them.
    rec=[]
    for key,kind,aux in (('pos_add',3,M.pos),('residual1',1,M.nextx),('residual2',1,M.nextx)):
        q=schema['residuals'][key];rec += [kind,aux,((q['eb']&255)<<8)|(q['ea']&255),q['eo']&255,0,0,0,0]
    for off,key in ((0,'n1'),(1,'n2'),(2,'final')): rec += [2,M.gamma+off*24,schema['norms'][key]['gamma_shift'],0,0,0,0,0]
    rec += [4,M.scratch,0,0,0,0,0,0]
    write(M.records,rec)
    return image

def build_program(state:dict[str,np.ndarray], schema:dict[str,Any])->list[int]:
    """Emit control only; caller supplies gathered token rows at ``M.x``."""
    p=[]
    def emit(blob):p.extend(_words(blob))
    # records 0..2 residual/pos, 3..5 norms, 6 softmax; extra records are
    # populated below for QK/AV and each boundary layout at M.vec.
    emit(_vec(2,M.x,M.x,M.records,1536)); emit(_vec(1,M.r,M.x,M.records+24,1536)); emit(_vec(5,M.qkv_a,M.r,M.vec,1536))
    # The legacy GEMM body is emitted per B=4 group/output tile.  QPARAM is
    # intentionally outside the group loop, avoiding 8x duplicated records.
    def linear(weight_base, weight, in_base, out_base, in_cols, out_cols, line, relu=False):
        in_blocks=(in_cols+7)//8;out_blocks=(out_cols+7)//8
        mult,shift=line['multipliers'],line['right_shifts']
        for ob in range(out_blocks):
            emit(encodeRequantParamsVector(mult[ob*8:ob*8+8]+[1]*max(0,ob*8+8-len(mult)),shift[ob*8:ob*8+8]+[0]*max(0,ob*8+8-len(shift)),cfg=CFG))
            for g in range(8):
                for ib in range(in_blocks):
                    emit(encodeLoadWeights(weight_base+(ob*in_blocks+ib)*32,cfg=CFG));emit(encodeLoadInputs(in_base+g*in_cols*2+ib*16,cfg=CFG,batch_count=4));emit(encodeRun(out_base+ob*128+g*16,compute_en=True,quantize_en=False,relu_en=False,acc_clear_en=(ib==0),batch_count=4,cfg=CFG))
                emit(encodeRun(out_base+ob*128+g*16,compute_en=False,quantize_en=True,relu_en=relu,batch_count=4,cfg=CFG))
    linear(M.qkv,state['blocks.0.qkv.weight'],M.qkv_a,M.qkv_a,48,144,schema['linears']['qkv'])
    emit(encodeHalt(cfg=CFG));return p

def image_hash(words:list[int])->str:return hashlib.sha256(struct.pack('<'+'H'*len(words),*words)).hexdigest()
