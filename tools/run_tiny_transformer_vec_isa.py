"""Generate Phase-2 VEC ISA/reference differential evidence."""
from __future__ import annotations

import hashlib, json, struct, subprocess, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "firmware" / "host"
sys.path.insert(0, str(HOST))
from isa_encoder import IsaConfig, encodeHalt, encode_vec_vadd_rq, encode_vec_vpos_add, encode_vec_vrmsnorm, encode_vec_vsoftmax
from isa_simulator import UTPUISASimulator
from tiny_transformer_reference import TinyTransformerConfig, integer_decoder_trace, rmsnorm_i8, softmax_u8, vadd_rq

CFG = IsaConfig(address_width=16, compute_data_width=8, descriptor_isa=True, vector_unit=True)
X, R, SCORE, NEXT, FF, POS, G1, G2, GF = 16968, 17736, 20808, 22088, 22856, 0, 768, 792, 816
P0, P1, PS, PR1, P2, PR2, PF, SCRATCH = 25928, 25936, 25944, 25952, 25960, 25968, 25976, 26256

def words(b): return list(struct.unpack("<" + "H" * (len(b)//2), b))
def put(sim, base, values):
    a = np.asarray(values).reshape(-1)
    for i in range(0, len(a), 2): sim.buffer[base+i//2] = (int(a[i]) & 255) | ((int(a[i+1]) & 255) << 8)
def get(sim, base, n, unsigned=False):
    a = np.empty(n, np.int64)
    for i in range(n//2): a[2*i], a[2*i+1] = sim.buffer[base+i]&255, (sim.buffer[base+i]>>8)&255
    return (a.astype(np.uint8) if unsigned else np.where(a >= 128, a-256, a).astype(np.int8))
def record(sim, addr, kind, data): sim.buffer[addr:addr+8] = [kind, *data] + [0]*(7-len(data))
def run(sim, program): sim.run_words(words(program + encodeHalt(cfg=CFG)), preserve_buffer=True)
def source_tree(): return {p: subprocess.check_output(["git","rev-parse",f"HEAD:{p}"], cwd=ROOT, text=True).strip() for p in ("firmware/host", "tools")}
def expword(ea, eb, eo): return [((eb&255)<<8)|(ea&255), eo&255]
def make_sim(): return UTPUISASimulator(array_size=16,buffer_size=30000,cfg=CFG)

def main():
    checkpoint = torch.load(ROOT / "bench/results/tiny_transformer_numerics.pt", map_location="cpu", weights_only=False)
    schema, state, cfg = checkpoint["export_schema"], checkpoint["qstate"], TinyTransformerConfig(**checkpoint["config"])
    rng = np.random.default_rng(20261013); compared = {k: 0 for k in ("vadd_rq","vpos_add","vrmsnorm","vsoftmax")}; mismatch = dict.fromkeys(compared, 0)
    # 10,000 adversarial/random RMS and softmax heads, grouped only by the fixed v1 shape.
    rms_rows = rng.integers(-128, 128, (10000,48), dtype=np.int8); gamma = rng.integers(-128,128,48,dtype=np.int8)
    bins=set()
    sim=make_sim()
    for start in range(0, 10000, 32):
        rows0=rms_rows[start:start+32]; put(sim,X,rows0.reshape(-1)); put(sim,G1,gamma); record(sim,P1,2,[G1,3]); run(sim,encode_vec_vrmsnorm(NEXT,X,P1,len(rows0)*48,cfg=CFG)); actual=get(sim,NEXT,len(rows0)*48).reshape(rows0.shape); expected=np.stack([rmsnorm_i8(row,gamma,3) for row in rows0]); mismatch["vrmsnorm"] += int(np.count_nonzero(actual != expected)); compared["vrmsnorm"] += actual.size
        for row in rows0:
            v=int(np.rint(np.mean(row.astype(np.int64)**2)))+1; q=(v.bit_length()-1)&~1; mq=min(65535,max(16384,int(round((v<<14)/(1<<q))))); bins.add(min(191,(mq-16384)>>8))
    soft = rng.integers(-128,128,(10000,32,32),dtype=np.int8)
    sim=make_sim()
    for head in soft:
        put(sim,SCORE,head.reshape(-1)); record(sim,PS,4,[SCRATCH]); run(sim,encode_vec_vsoftmax(SCORE,SCORE,PS,head_id=0,cfg=CFG)); actual=get(sim,SCORE,1024,True).reshape(32,32); expected=np.stack([softmax_u8(head[i],i+1) for i in range(32)]); mismatch["vsoftmax"]+=int(np.count_nonzero(actual!=expected)); compared["vsoftmax"]+=1024
    # Exhaustive all int8 pairs at each exported residual exponent triple.
    pairs=np.arange(-128,128,dtype=np.int16); aa,bb=np.meshgrid(pairs,pairs,indexing="ij"); aa,bb=aa.reshape(-1).astype(np.int8),bb.reshape(-1).astype(np.int8)
    triples={tuple(int(r[k]) for k in ("ea","eb","eo")) for r in schema["residuals"].values()} | {(-31,-31,-30)}
    for ea,eb,eo in triples:
        want,_=vadd_rq(aa,ea,bb,eb,eo); got=[]
        for off in range(0,len(aa),1536):
            end=min(off+1536,len(aa)); a=np.pad(aa[off:end],(0,1536-(end-off))); b=np.pad(bb[off:end],(0,1536-(end-off))); sim=make_sim(); put(sim,X,a); put(sim,NEXT,b); record(sim,PR1,1,[NEXT,*expword(ea,eb,eo)]); run(sim,encode_vec_vadd_rq(R,X,PR1,1536,cfg=CFG)); got.append(get(sim,R,1536)[:end-off])
        mismatch["vadd_rq"]+=int(np.count_nonzero(np.concatenate(got)!=want)); compared["vadd_rq"]+=len(want)
    # Real-table position add and every vector boundary for 512 validation sequences.
    data=np.asarray([json.loads((HOST/"fixtures/tiny_transformer_vocab.json").read_text())["characters"].index(c) for c in (ROOT/"build/tiny_transformer/input.txt").read_text(encoding="utf-8")],np.int64)
    split=json.loads((HOST/"fixtures/tiny_transformer_split.json").read_text()); starts=range(split["validation_start"],split["validation_start"]+512*32,32); stage={k:{"compared_elements":0,"mismatches":0} for k in ("pos_add","rmsnorm1","softmax_head0","softmax_head1","softmax_head2","residual1","rmsnorm2","residual2","final_norm")}
    pos=state["pos.weight"][:32]; g1,g2,gf=state["blocks.0.n1.weight"],state["blocks.0.n2.weight"],state["final.weight"]
    sim=make_sim()
    for st in starts:
        trace=integer_decoder_trace(data[st:st+32],state,schema,cfg); put(sim,X,state["token.weight"][data[st:st+32]].reshape(-1)); put(sim,POS,pos.reshape(-1)); put(sim,G1,g1); put(sim,G2,g2); put(sim,GF,gf)
        rp=schema["residuals"]["pos_add"]; record(sim,P0,3,[POS,*expword(rp["ea"],rp["eb"],rp["eo"])]); run(sim,encode_vec_vpos_add(X,X,P0,cfg=CFG)); actual=get(sim,X,1536).reshape(32,48); expected=trace["pos_add"][0]; stage["pos_add"]["mismatches"]+=int(np.count_nonzero(actual!=expected)); stage["pos_add"]["compared_elements"]+=actual.size
        record(sim,P1,2,[G1,int(schema["norms"]["n1"]["gamma_shift"])]); run(sim,encode_vec_vrmsnorm(NEXT,X,P1,1536,cfg=CFG)); actual=get(sim,NEXT,1536).reshape(32,48); expected=trace["rmsnorm1"][0]; stage["rmsnorm1"]["mismatches"]+=int(np.count_nonzero(actual!=expected)); stage["rmsnorm1"]["compared_elements"]+=actual.size
        for h in range(3):
            put(sim,SCORE,trace["scores"][0,h].reshape(-1)); record(sim,PS,4,[SCRATCH]); run(sim,encode_vec_vsoftmax(SCORE,SCORE,PS,head_id=h,cfg=CFG)); actual=get(sim,SCORE,1024,True).reshape(32,32); expected=trace["softmax_probs"][0,h]; key=f"softmax_head{h}"; stage[key]["mismatches"]+=int(np.count_nonzero(actual!=expected)); stage[key]["compared_elements"]+=actual.size
        put(sim,R,trace["pos_add"][0].reshape(-1)); put(sim,NEXT,trace["out_proj"][0].reshape(-1)); rr=schema["residuals"]["residual1"]; record(sim,PR1,1,[NEXT,*expword(rr["ea"],rr["eb"],rr["eo"])]); run(sim,encode_vec_vadd_rq(X,R,PR1,1536,cfg=CFG)); actual=get(sim,X,1536).reshape(32,48); expected=trace["residual1"][0]; stage["residual1"]["mismatches"]+=int(np.count_nonzero(actual!=expected)); stage["residual1"]["compared_elements"]+=actual.size
        record(sim,P2,2,[G2,int(schema["norms"]["n2"]["gamma_shift"])]); run(sim,encode_vec_vrmsnorm(NEXT,X,P2,1536,cfg=CFG)); actual=get(sim,NEXT,1536).reshape(32,48); expected=trace["rmsnorm2"][0]; stage["rmsnorm2"]["mismatches"]+=int(np.count_nonzero(actual!=expected)); stage["rmsnorm2"]["compared_elements"]+=actual.size
        put(sim,R,trace["residual1"][0].reshape(-1)); put(sim,FF,trace["ffn2"][0].reshape(-1)); rr=schema["residuals"]["residual2"]; record(sim,PR2,1,[FF,*expword(rr["ea"],rr["eb"],rr["eo"])]); run(sim,encode_vec_vadd_rq(X,R,PR2,1536,cfg=CFG)); actual=get(sim,X,1536).reshape(32,48); expected=trace["residual2"][0]; stage["residual2"]["mismatches"]+=int(np.count_nonzero(actual!=expected)); stage["residual2"]["compared_elements"]+=actual.size
        record(sim,PF,2,[GF,int(schema["norms"]["final"]["gamma_shift"])]); run(sim,encode_vec_vrmsnorm(NEXT,X,PF,1536,cfg=CFG)); actual=get(sim,NEXT,1536).reshape(32,48); expected=trace["final_norm"][0]; stage["final_norm"]["mismatches"]+=int(np.count_nonzero(actual!=expected)); stage["final_norm"]["compared_elements"]+=actual.size
    payload={"status":"phase2_vec_isa","per_op":{k:{"compared_elements":compared[k],"mismatches":mismatch[k]} for k in compared},"rsqrt_lut_bin_coverage":{"covered":len(bins),"total":192},"sequences_used":512,"stages":stage,"source_tree":source_tree()}
    out=ROOT/"bench/results/tiny_transformer_vec_isa.json"; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(payload,indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
