"""Finalize incremental INT4 measurements into the committed study artifacts."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; BASE=0.5623131903694261; PRUNED_BASE=0.564346
BASE_STREAM=10684413; PRUNED_STREAM=6613245; COMPUTE=12338619; PRUNED_COMPUTE=7734699
def load(p): return json.loads((ROOT/p).read_text())
def main():
    old=load(Path('bench/results/int4_study.json')); rows={r['config']:r for r in old['rows']}
    # Existing accounting rows contain the exact derived byte model for all original configs.
    # Add the missing w2+qkv derived row from the same tensor deltas.
    q=rows['qkv']; w2=rows['w2']; base=rows['measured_baseline_integer']
    w2q=dict(rows.get('w2_qkv', {'config':'w2_qkv'})); w2q.update({
        'stream_B_token_unpruned': int(q['stream_B_token_unpruned']-(BASE_STREAM-w2['stream_B_token_unpruned'])),
        'stream_B_token_pruned_K8192': int(q['stream_B_token_pruned_K8192']-(PRUNED_STREAM-w2['stream_B_token_pruned_K8192'])),
        'compute_cycles_est':PRUNED_COMPUTE})
    rows['w2_qkv']=w2q
    # Measured quality files, including the already-completed all-INT4 file.
    files={'everything':'build/int4/everything_eval.json','qkv':'build/int4/qkv_eval.json','wo':'build/int4/wo_eval.json',
           'w2':'build/int4/w2_eval.json','all_attention':'build/int4/all_attention_eval.json','w2_qkv':'build/int4/w2_qkv_eval.json'}
    for name,fn in files.items():
        r=load(fn); row=rows.setdefault(name,{'config':name}); row.update({'int_bpb':r['bits_per_byte'],'quality_status':'MEASURED',
          'delta_pct':(r['bits_per_byte']/BASE-1)*100,'wall_seconds':r['elapsed_seconds'],'eval_tokens':r['eval_tokens'],
          'artifact':r['artifact'],'artifact_sha256':r['artifact_sha256']})
    for name in ('everything','qkv','wo','w2','all_attention','w2_qkv'):
        row=rows[name]; b=row.get('stream_B_token_unpruned'); p=row.get('stream_B_token_pruned_K8192')
        row.update({'compute_cycles_est':PRUNED_COMPUTE,'tok_s_unpruned_derived':50_000_000/max(b/0.5,COMPUTE),
                    'tok_s_pruned_derived':50_000_000/max(p/0.5,PRUNED_COMPUTE),
                    'speedup_vs_pruned_derived':(50_000_000/max(p/0.5,PRUNED_COMPUTE))/(50_000_000/max(PRUNED_STREAM/0.5,PRUNED_COMPUTE))})
    improvements=[]
    for method,fn,config in [('non_gptq_improved_scale_clipping','build/int4/everything_mse_eval.json','everything'),('non_gptq_improved_scale_clipping','build/int4/wo_mse_eval.json','wo')]:
        r=load(fn); improvements.append({'config':config,'method':method,'int_bpb':r['bits_per_byte'],'delta_pct_vs_baseline':(r['bits_per_byte']/BASE-1)*100,
          'wall_seconds':r['elapsed_seconds'],'artifact':r['artifact'],'artifact_sha256':r['artifact_sha256'],'quality_status':'MEASURED'})
    pr=load('build/int4/pruned_eval.json'); best=rows['everything']
    out={'schema_version':2,'status':'complete_no_go','labels':{'MEASURED':'direct artifact/protocol result','DERIVED':'integer arithmetic from packed records','ESTIMATED':'RTL schedule/throughput model','NOT RUN':'not executed'},
      'python':'C:\\Users\\bhask\\AppData\\Local\\Python\\bin\\python.exe (python command)','group_size':32,'classifier_K':8192,
      'fixed_eval_text':{'path':'build/int4/eval_text.txt','sha256':'c8191cfe5a461bc825b3e3326a2921dfe978a63878b4c14547033b7217adea17'},
      'fixed_eval_baseline_bpb':BASE,'pruned_integer_baseline_bpb':PRUNED_BASE,'baseline_stream_B_token':BASE_STREAM,'pruned_stream_B_token':PRUNED_STREAM,
      'compute_cycles_unpruned_measured':COMPUTE,'compute_cycles_pruned_measured':PRUNED_COMPUTE,
      'pruned_throughput_formula':'50e6 / max(stream_bytes/0.5, compute_cycles); INT4 does not change GEMV cycles in RTL',
      'rows':[rows[k] for k in ['measured_baseline_integer','everything','qkv','wo','w2','all_attention','w2_qkv']],
      'improvements':improvements,'pruned_int4':pr,'decision':{'unpruned_threshold':BASE*1.01,'pruned_threshold':PRUNED_BASE*1.01,
        'speed_threshold_x':1.2,'candidate':'everything','candidate_unpruned_bpb':best['int_bpb'],'candidate_pruned_bpb':pr['bits_per_byte'],
        'candidate_pruned_speedup_x':best['speedup_vs_pruned_derived'],'result':'NO-GO','reason':'quality exceeds both 1% thresholds; speed leg passes for everything'},
      'not_run':['packer option/default 15m byte-identity proof: NOT RUN (NO-GO)','matching golden and tools/decode/run_decode_rtl.py ideal-stream KV4 1-prompt 2-token simulation: NOT RUN (NO-GO); exact command would be: python tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --prompts 1 --gen 2 --output bench/results/int4_decode_rtl.json','RTL edits, constraint edits, decode_gate.py changes, default-numerics changes: NOT RUN']}
    (ROOT/'bench/results/int4_study.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# stories15M mixed-precision INT4 study','',f'Fixed text SHA-256: `c8191cfe5a461bc825b3e3326a2921dfe978a63878b4c14547033b7217adea17` (MEASURED). Group size 32.','',
      '| config | int bpb (MEASURED) | delta | unpruned B/token (DERIVED) | pruned B/token (DERIVED) | pruned tok/s (DERIVED) | wall s |','|---|---:|---:|---:|---:|---:|---:|']
    for r in out['rows']:
        ts = r.get('tok_s_pruned_derived'); ts = ts if isinstance(ts, str) else (f'{ts:.3f}' if ts is not None else '—')
        ws = r.get('wall_seconds'); ws = ws if isinstance(ws, str) else (f'{ws:.1f}' if ws else '—')
        lines.append(f"| {r['config']} | {r['int_bpb']:.9f} | {r['delta_pct']:+.3f}% | {r.get('stream_B_token_unpruned','—')} | {r.get('stream_B_token_pruned_K8192','—')} | {ts} | {ws} |")
    lines += ['', 'Improvement method (MEASURED): non-GPTQ improved scale/clipping search (`improved=True`, `gptq=False`) was worse: everything 0.607202815 bpb; wo 0.570795902 bpb.',
      f"Pruned+INT4 everything (MEASURED): {pr['bits_per_byte']:.9f} bpb, restricted K=8192 softmax, {pr['elapsed_seconds']:.1f} s. The pruned-only integer reference is 0.564346 bpb (MEASURED in vocab-prune study).",
      '', 'Decision: **NO-GO**. Everything passes the speed leg at 1.335x pruned-only (DERIVED), but fails unpruned and pruned 1% quality thresholds.',
      '', 'INT4 does not reduce GEMV cycles in the existing RTL datapath; its throughput benefit here is stream bandwidth only (DERIVED from RTL parameters).',
      '', 'Reproduction: `python tools/int4/build_artifacts.py`; `python tools/int4/integer_eval.py ...`; `python tools/int4/pruned_eval.py ...`; `python tools/int4/finalize_study.py`.',
      '', 'NOT RUN: GO-only packer option, byte-identical proof, golden, and new-image RTL simulation; exact skipped simulation command is recorded in `bench/results/int4_study.json`.']
    (ROOT/'docs/INT4_STUDY.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
