import subprocess,sys,time,json,os
from pathlib import Path
os.environ['UTPU_SIM']='verilator'
rows=[]
for label,extra in [('15m_small',[]),('15m_big',['--prompts','8','--gen','32'])]:
 cmd=[sys.executable,'tools/decode/run_decode_rtl.py','--model','15m','--kv-bits','4','--gen','8','--sim','verilator','--output',f'sprint/data/{label}.json',*extra]
 t=time.perf_counter()
 with open(f'sprint/data/{label}.log','w') as f:
  try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=3600);rc=r.returncode
  except subprocess.TimeoutExpired:rc=124
 rows.append(dict(label=label,returncode=rc,wall_seconds=time.perf_counter()-t,command=cmd))
 Path('sprint/data/15m_timing.json').write_text(json.dumps(rows,indent=2))
 if rc:break
