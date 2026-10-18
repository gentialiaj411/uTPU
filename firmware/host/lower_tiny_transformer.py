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
    gamma:int=16320; records:int=16392; x:int=16968; r:int=17736; qkv_l:int=18504
    score:int=20808; out_l:int=21320; head:int=22088; hidden_l:int=22856
    vec:int=25928; scratch:int=26256; qkv_r:int=26768; out_r:int=29072
    ffn2_l:int=29840; logits_l:int=30608
M=Map()
R_POS,R_RES1,R_RES2,R_N1,R_N2,R_FINAL,R_SOFT=range(M.records,M.records+56,8)
R_QK0,R_QK1,R_QK2,R_AV0,R_AV1,R_AV2=range(M.records+56,M.records+104,8)
R_RM_LOAD_48_QKV,R_LFIN_RM_144,R_RM_LOAD_48_OUT,R_LFIN_RM_48_OUT,R_RM_LOAD_48_FF,R_LFIN_RM_48_FF,R_RM_LOAD_48_LM=range(M.records+104,M.records+160,8)
R_LFIN_LOAD_192=M.records+160

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
def _record_mat(b,m,k,n,tr,au,ca,mult,shift,astride=None,bstride=None):
    return [5,b,(m-1)|((k-1)<<5)|((n-1)<<10)|(int(tr)<<15),(astride or k)|((bstride or (k if tr else n))<<8),int(au)|(int(ca)<<2),mult,shift,0]

def build_weight_image(state:dict[str,np.ndarray], schema:dict[str,Any])->list[int]:
    """Static A5 image: tiled linear weights plus VEC tables/records."""
    image=[0]*16968
    def write(base,data): image[base:base+len(data)]=data
    # The tied embedding is host-gathered in row-major order, but its stored
    # image is also the legacy-tiled 72x48 LM-head matrix (seven padded rows).
    write(M.token,_tile_words(state['token.weight']))
    write(M.pos,_pack(state['pos.weight']));write(M.qkv,_tile_words(state['blocks.0.qkv.weight']));write(M.out,_tile_words(state['blocks.0.out.weight']));write(M.ff1,_tile_words(state['blocks.0.ff1.weight']));write(M.ff2,_tile_words(state['blocks.0.ff2.weight']))
    write(M.gamma,_pack(np.concatenate((state['blocks.0.n1.weight'],state['blocks.0.n2.weight'],state['final.weight']))))
    # Records are fixed-address and fully zero-padded; dynamic tensor regions never alias them.
    rec=[]
    for key,kind,aux in (('pos_add',3,M.pos),('residual1',1,M.out_r),('residual2',1,M.r)):
        q=schema['residuals'][key];rec += [kind,aux,((q['eb']&255)<<8)|(q['ea']&255),q['eo']&255,0,0,0,0]
    for off,key in ((0,'n1'),(1,'n2'),(2,'final')): rec += [2,M.gamma+off*24,schema['norms'][key]['gamma_shift'],0,0,0,0,0]
    rec += [4,M.scratch,0,0,0,0,0,0]
    score, av=schema['linears']['scores'],schema['linears']['av']
    for h in range(3):
        q,k=M.qkv_r+h*8,M.qkv_r+24+h*8
        rec += _record_mat(k,32,16,32,True,False,True,score['multipliers'][0],score['right_shifts'][0],144,144)
    for h in range(3):
        v=M.qkv_r+48+h*8
        rec += _record_mat(v,32,32,16,False,True,True,av['multipliers'][0],av['right_shifts'][0],32,144)
    # RM->physical legacy LOAD (mode6) and LFIN->RM (mode1) descriptors.
    # All conversions are non-overlapping: VLAYOUT clears its destination.
    for rows,cols,mode in ((32,48,6),(32,144,1),(32,48,4),(32,48,1),
                           (32,48,6),(32,48,1),(32,48,6),(32,192,5)):
        rec += [6,(rows-1)|((cols-1)<<5),mode,cols|(cols<<8),0,0,0,0]
    write(M.records,rec)
    return image

def build_program_stages(state:dict[str,np.ndarray], schema:dict[str,Any])->list[tuple[str,list[int]]]:
    """Emit deterministic executable stages; callers append the one HALT."""
    p=[];stages=[]
    def emit(blob):p.extend(_words(blob))
    def begin(): return len(p)
    def end(name,start): stages.append((name,p[start:]))
    # records 0..2 residual/pos, 3..5 norms, 6 softmax; extra records are
    # populated below for QK/AV and each boundary layout at M.vec.
    at=begin();emit(_vec(2,M.x,M.x,R_POS,1536));end('pos_add',at)
    at=begin();emit(_vec(1,M.r,M.x,R_N1,1536));end('rmsnorm1',at)
    at=begin();emit(_vec(5,M.out_l,M.r,R_RM_LOAD_48_QKV,1536));end('norm1_to_qkv_load',at)
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
    at=begin();linear(M.qkv,state['blocks.0.qkv.weight'],M.out_l,M.qkv_l,48,144,schema['linears']['qkv']);end('qkv',at)
    at=begin();emit(_vec(5,M.qkv_r,M.qkv_l,R_LFIN_RM_144,4608));end('qkv_to_rm',at)
    for h in range(H):
        at=begin();emit(_vec(4,M.score,M.qkv_r+h*8,R_QK0+h*8,1024,causal=True));end(f'scores_head{h}',at)
        at=begin();emit(_vec(3,M.score,M.score,R_SOFT,1024,causal=True,head=h));end(f'softmax_head{h}',at)
        at=begin();emit(_vec(4,M.head+h*256,M.score,R_AV0+h*8,512,causal=True));end(f'av_head{h}',at)
    at=begin();emit(_vec(5,M.r,M.head,R_RM_LOAD_48_OUT,1536));end('heads_to_out_load',at)
    at=begin();linear(M.out,state['blocks.0.out.weight'],M.r,M.out_l,48,48,schema['linears']['out']);end('out_proj',at)
    at=begin();emit(_vec(5,M.out_r,M.out_l,R_LFIN_RM_48_OUT,1536));end('out_to_rm',at)
    at=begin();emit(_vec(0,M.out_r,M.x,R_RES1,1536));end('residual1',at)
    at=begin();emit(_vec(1,M.r,M.out_r,R_N2,1536));end('rmsnorm2',at)
    at=begin();emit(_vec(5,M.x,M.r,R_RM_LOAD_48_FF,1536));end('norm2_to_ff_load',at)
    at=begin();linear(M.ff1,state['blocks.0.ff1.weight'],M.x,M.hidden_l,48,192,schema['linears']['ff1'],relu=True);end('ffn1_leaky_relu',at)
    at=begin();emit(_vec(5,M.x,M.hidden_l,R_LFIN_LOAD_192,6144));end('ffn1_to_ff2_load',at)
    at=begin();linear(M.ff2,state['blocks.0.ff2.weight'],M.x,M.ffn2_l,192,48,schema['linears']['ff2']);end('ffn2',at)
    at=begin();emit(_vec(5,M.r,M.ffn2_l,R_LFIN_RM_48_FF,1536));end('ffn2_to_rm',at)
    at=begin();emit(_vec(0,M.r,M.out_r,R_RES2,1536));end('residual2',at)
    at=begin();emit(_vec(1,M.out_l,M.r,R_FINAL,1536));end('final_norm',at)
    at=begin();emit(_vec(5,M.x,M.out_l,R_RM_LOAD_48_LM,1536));end('final_to_lm_load',at)
    at=begin();linear(M.token,state['token.weight'],M.x,M.logits_l,48,72,schema['linears']['logits']);end('logits',at)
    return stages

def build_program(state:dict[str,np.ndarray], schema:dict[str,Any])->list[int]:
    """Emit the complete program; caller supplies gathered token rows at ``M.x``."""
    return [word for _,stage in build_program_stages(state,schema) for word in stage]+_words(encodeHalt(cfg=CFG))

def image_hash(words:list[int])->str:return hashlib.sha256(struct.pack('<'+'H'*len(words),*words)).hexdigest()
