import os,json,subprocess,sys,time,datetime
from pathlib import Path
D=Path('sprint/data');env=os.environ.copy();env['UTPU_SIM']='icarus';r={}
for u in ['actq','rms','vec','attn','attn15']:
 cmd=[sys.executable,'tools/decode/fault_inject.py',u];t=time.perf_counter()
 with (D/('fault_'+u+'_icarus_final.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=300)
 r['fault_'+u]=dict(command=cmd,rc=p.returncode,seconds=time.perf_counter()-t,backend='icarus')
 (D/'remaining_fault_recovery.json').write_text(json.dumps(r,indent=2))
 with Path('SPRINT_LOG.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' UTPU_SIM=icarus '+subprocess.list2cmdline(cmd)+' EXIT '+str(p.returncode)+'\n')
