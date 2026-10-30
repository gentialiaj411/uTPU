import subprocess,sys,json,time
from pathlib import Path
rows=[]
for name,runner in [('uart','tools/board/run_uart_rtl.py'),('qspi','tools/decode/run_qspi_rtl.py')]:
 for sim in ['verilator','icarus']:
  label=name+'_'+sim;t=time.perf_counter()
  with open('sprint/data/'+label+'.log','w') as f:
   try:r=subprocess.run([sys.executable,runner,'--sim',sim,'--output','sprint/data/'+label+'.json'],stdout=f,stderr=subprocess.STDOUT,timeout=2700);rc=r.returncode
   except subprocess.TimeoutExpired:rc=124
  rows.append(dict(label=label,returncode=rc,seconds=time.perf_counter()-t))
  Path('sprint/data/deterministic_recovery_progress.json').write_text(json.dumps(rows,indent=2))
Path('sprint/data/deterministic_recovery_complete.json').write_text(json.dumps(rows,indent=2))
