import subprocess,sys,time,json
from pathlib import Path
rows=[]
for name,runner in [('qspi','tools/decode/run_qspi_rtl.py'),('uart','tools/board/run_uart_rtl.py')]:
 t=time.perf_counter()
 with open('sprint/data/'+name+'_verilator.log','w') as f:
  try:r=subprocess.run([sys.executable,runner,'--sim','verilator','--output','sprint/data/'+name+'_verilator.json'],stdout=f,stderr=subprocess.STDOUT,timeout=2700);rc=r.returncode
  except subprocess.TimeoutExpired:rc=124
 rows.append(dict(label=name+'_verilator_recovery',returncode=rc,seconds=time.perf_counter()-t))
Path('sprint/data/backend_recovery_complete.json').write_text(json.dumps(rows,indent=2))
