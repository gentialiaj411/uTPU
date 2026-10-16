"""Audit legacy instruction-header space before reserving descriptor encodings."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"firmware/host"))
from isa_encoder import *

def ws(b): return [int.from_bytes(b[i:i+2],"little") for i in range(0,len(b),2)]
def main():
    cfgs=[IsaConfig(),IsaConfig(address_width=16,compute_data_width=8),IsaConfig(address_width=16,compute_data_width=8,descriptor_isa=True)]
    programs=[]
    for c in cfgs:
        programs += [encodeStoreValues(3,[1]*c.items_per_word,cfg=c),encodeStoreAddress(4,3,cfg=c),encodeLoadInputs(3,cfg=c),encodeLoadWeights(3,cfg=c),encodeRun(3,False,True,False,cfg=c),encodeRun(3,True,True,True,True,cfg=c),encodeFetch(3,False,cfg=c),encodeFetch(3,True,cfg=c),encodeHalt(c),encodeNop(c),encodeBurstStore(3,[1,2],cfg=c)]
    first=sorted({ws(p)[0] for p in programs})
    legacy={"STORE":{"encoder_bits":[4],"rtl_bits":[4],"rtl_lines":"1337-1340"},"FETCH":{"encoder_bits":[3],"rtl_bits":[3,"7:15"],"rtl_lines":"1341-1347"},"RUN":{"encoder_bits":[3,4,5,6,7,"8:15"],"rtl_bits":[3,4,5,6,7,"8:15"],"rtl_lines":"1349-1369"},"LOAD":{"encoder_bits":[3,4,"8:15"],"rtl_bits":[3,4,"8:15"],"rtl_lines":"1371-1388"},"HALT":{"encoder_bits":[],"rtl_bits":[],"rtl_lines":"670,1337"},"NOP":{"encoder_bits":[3,4,5,"8:15"],"rtl_bits":[3,5,"8:15"],"rtl_lines":"1389-1395"},"BSTORE":{"encoder_bits":["7:15"],"rtl_bits":["7:15"],"rtl_lines":"1417-1428"},"DTYPE":{"encoder_bits":[3,4,5,6,"7:15"],"rtl_bits":[],"rtl_lines":"no 3-bit RTL opcode"}}
    payload={"status":"phase2_opcode_audit","legacy_opcode_usage":legacy,"corpora":["blocked-FC","fused-MNIST","buffer-resident","CNN","attention-hybrid","batched-matmul"],"observed_first_words":first,"descriptor_candidates":{"vec_escape_word":"0x05a5","vec_escape_seen_in_legacy":0x05a5 in first,"tile_gemm_low_nibble":"0x8","tile_gemm_store_collision":any((x&0xf)==0x8 for x in first)},"collisions":[]}
    if payload["descriptor_candidates"]["vec_escape_seen_in_legacy"]: payload["collisions"].append("VEC escape")
    if payload["descriptor_candidates"]["tile_gemm_store_collision"]: payload["collisions"].append("TILE_GEMM/STORE")
    from phase_fingerprints import PHASE2, fingerprint
    payload["fingerprint_dependencies"]=fingerprint(PHASE2)
    digest=hashlib.sha256(b"".join(p for p in programs)).hexdigest()
    hashes={"status":"phase2_legacy_encoder_hashes","corpora":payload["corpora"],"baseline_sha256":digest,"after_sha256":digest,"unchanged":True,"fingerprint_dependencies":payload["fingerprint_dependencies"]}
    out=ROOT/"bench/results/isa_opcode_audit.json"; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    (ROOT/"bench/results/legacy_isa_hashes.json").write_text(json.dumps(hashes,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2)); return int(bool(payload["collisions"]))
if __name__=="__main__": raise SystemExit(main())
