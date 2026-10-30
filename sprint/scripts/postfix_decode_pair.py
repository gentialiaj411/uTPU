import sys,subprocess,json,time
from pathlib import Path
backend=sys.argv[1];rows=[]
for flash in ['ideal','qspi']:
 cmd=[sys.executable,'sprint/scripts/isolated_postfix_decode.py',backend,flash,f'sprint/data/e2e_{flash}_{backend}.json'];t=time.perf_counter()
 with open(f'sprint/data/postfix_{flash}_{backend}.log','w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 rows.append(dict(flash=flash,backend=backend,rc=r.returncode,seconds=time.perf_counter()-t))
 if r.returncode:break
Path(f'sprint/data/postfix_pair_{backend}.json').write_text(json.dumps(rows,indent=2))
