"""Assemble GPTQ layer evidence and sequential export/eval manifests."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    layer=json.loads((ROOT/'bench/results/gptq_layer_report.json').read_text())
    out={'schema_version':1,'model':'stories15M','eval_tokens':50000,
         'eval_protocol':'sequential exported artifacts; each evaluator is chunked; peak RAM recorded per row',
         'gptq_layer_reconstruction':layer,
         'configs':[],'samples':None}
    for name in ('gptq_g32','gptq_qkv_wo_int8_g32','gptq_qkv_wo_w2_int8_g32'):
        p=ROOT/f'models/numerics_v2/{name}.npz'; row={'config':name,'artifact_sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}
        e=ROOT/f'build/{name}_eval.json'
        if e.exists(): row.update(json.loads(e.read_text()))
        else: row['status']='pending_resource_safe_eval'
        out['configs'].append(row)
    s=ROOT/'bench/results/chunk6_samples.json'
    if s.exists(): out['samples']=json.loads(s.read_text())
    (ROOT/'bench/results/numerics_v2_chunk6.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
