import os,sys,subprocess,json,time,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];os.chdir(R)
def run(cmd,label,timeout=2700):
 t=time.perf_counter()
 with open(f'sprint/data/{label}.log','w') as f:
  try:r=subprocess.run([sys.executable,*cmd],stdout=f,stderr=subprocess.STDOUT,timeout=timeout);rc=r.returncode
  except subprocess.TimeoutExpired:rc=124
 return dict(label=label,returncode=rc,seconds=time.perf_counter()-t)
records=json.loads(Path('sprint/data/crosscheck_progress.json').read_text()) if Path('sprint/data/crosscheck_progress.json').exists() else []
for gold in ['gemv15','rms','vec','attn','attn15']:
 if any(r['label']=='golden_'+gold and r['returncode']==0 for r in records):continue
 records.append(run([f'tools/decode/golden_{gold}.py'],f'golden_{gold}'))
commands=[('gemv',['tools/decode/run_gemv_rtl.py']),('gemv15',['tools/decode/run_gemv_rtl.py','--golden','build/decode/golden_gemv15']),('actq',['tools/decode/run_act_quant_rtl.py']),('rms',['tools/decode/run_rms_rtl.py']),('vec',['tools/decode/run_vec_rtl.py']),('attn',['tools/decode/run_attn_rtl.py']),('attn15',['tools/decode/run_attn15_rtl.py']),('qspi',['tools/decode/run_qspi_rtl.py']),('uart',['tools/board/run_uart_rtl.py']),('e2e_ideal',['tools/decode/run_decode_rtl.py','--model','260k','--kv-bits','4','--prompts','1','--gen','8']),('e2e_qspi',['tools/decode/run_decode_rtl.py','--model','260k','--kv-bits','4','--prompts','1','--gen','8','--flash','qspi'])]
for label,cmd in commands:
 for backend in ['verilator','icarus']:
  tag=f'{label}_{backend}'
  if any(r['label']==tag and r['returncode']==0 for r in records) and Path(f'sprint/data/{tag}.json').exists():continue
  records.append(run([*cmd,'--sim',backend,'--output',f'sprint/data/{tag}.json'],tag))
  Path('sprint/data/crosscheck_progress.json').write_text(json.dumps(records,indent=2))
with open('sprint/data/crosscheck.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['label','returncode','seconds']);w.writeheader();w.writerows(records)
