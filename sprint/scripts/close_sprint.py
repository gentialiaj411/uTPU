import json,time,datetime,subprocess,sys
from pathlib import Path
D=Path('sprint/data');deadline=json.loads((D/'wall_deadline.json').read_text())['unix']+120
while time.time()<deadline:
 try:r=json.loads((D/'fast_fault_recovery.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):r={}
 if (D/'final_flash_repair_complete.json').exists() and len(r)==6:break
 time.sleep(10)
s=json.loads((D/'engineering_status.json').read_text())
for k,v in r.items():
 if k in s:s[k+'_prior_attempt']=s[k]
 s[k]=v
(D/'engineering_status.json').write_text(json.dumps(s,indent=2))
cmd=[sys.executable,'tools/decode_gate.py'];t=time.perf_counter()
with (D/'strict_gate_closed.log').open('w') as f:q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
s['strict_gate_closed']=dict(command=cmd,rc=q.returncode,seconds=time.perf_counter()-t)
(D/'engineering_status.json').write_text(json.dumps(s,indent=2))
with Path('SPRINT_LOG.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' Final strict gate after all recovery producers: EXIT '+str(q.returncode)+'; sprint/data/strict_gate_closed.log.\n')
with (D/'report_closed.log').open('w') as f:subprocess.run([sys.executable,'sprint/scripts/report.py'],stdout=f,stderr=subprocess.STDOUT)
(D/'close_sprint_complete.json').write_text(json.dumps(dict(timestamp=datetime.datetime.now().isoformat(),faults=r,gate_rc=q.returncode)))
