import json,time,sys,subprocess,datetime,collections,re
from pathlib import Path
D=Path('sprint/data');stop_at=json.loads((D/'wall_deadline.json').read_text())['unix']-45
while time.time()<stop_at:time.sleep(min(10,stop_at-time.time()))
# Stop only validated task-specific roots; recurse into their own children.
for pid,needle in [(35420,'sprint/scripts/close_sprint.py'),(9228,'sprint/scripts/fast_fault_recovery.py'),(25632,'sprint/scripts/remaining_fault_recovery.py'),(13368,'tools/decode/fault_inject.py qspi')]:
 command=f"$r=Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; if ($r.CommandLine -like '*{needle}*') {{ function Stop-SprintTree($id) {{ Get-CimInstance Win32_Process -Filter \"ParentProcessId = $id\" | ForEach-Object {{ Stop-SprintTree $_.ProcessId }}; Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }}; Stop-SprintTree {pid} }}"
 subprocess.run(['powershell','-NoProfile','-Command',command],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
s=json.loads((D/'engineering_status.json').read_text());recovery={}
for name in ['fast_fault_recovery.json','remaining_fault_recovery.json']:
 try:recovery.update(json.loads((D/name).read_text()))
 except (FileNotFoundError,json.JSONDecodeError):pass
for unit in ['qspi','actq','rms','vec','attn','attn15']:
 key='fault_'+unit
 if key in s:s[key+'_prior_attempt']=s[key]
 if key in recovery:s[key]=recovery[key]
 else:
  started=(D/('fault_'+unit+'_icarus_final.log')).exists()
  s[key]=dict(command=[sys.executable,'tools/decode/fault_inject.py',unit],backend='icarus',status='TIMEBOXED' if started else 'NOT RUN',rc=124 if started else None,reason='Final sprint timebox; exact complete producer command is queued in OWNER_RUN2.ps1')
(D/'engineering_status.json').write_text(json.dumps(s,indent=2))
cmd=[sys.executable,'tools/decode_gate.py'];t=time.perf_counter()
with (D/'strict_gate_closed.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
s['strict_gate_closed']=dict(command=cmd,rc=p.returncode,seconds=time.perf_counter()-t)
s['P6']=dict(status='TIMEBOXED: complete stale fault producers and physical jobs via OWNER_RUN2; strict gate unchanged')
(D/'engineering_status.json').write_text(json.dumps(s,indent=2))
txt=(D/'strict_gate_closed.log').read_text();gate=json.loads(txt[txt.find('{'):]);counts=collections.Counter()
for u in gate['unmet']:
 m=re.search(r'(decode_fault_injection\[[^]]+\]|decode_synth_ooc\[[^]]+\]|boolean_[a-z_]+)',u)
 counts[m.group(1) if m else u.split(':')[0]]+=1
(D/'gate_unmet_categories.json').write_text(json.dumps(dict(counts),indent=2))
with Path('SPRINT_LOG.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' Final validation TIMEBOXED at fixed sprint deadline; remaining full fault sweeps queued before Vivado in OWNER_RUN2. Strict gate EXIT '+str(p.returncode)+'; unmet categories sprint/data/gate_unmet_categories.json. No checks relaxed or artifacts patched.\n')
with (D/'report_closed.log').open('w') as f:subprocess.run([sys.executable,'sprint/scripts/report.py'],stdout=f,stderr=subprocess.STDOUT)
(D/'close_sprint_complete.json').write_text(json.dumps(dict(timestamp=datetime.datetime.now().isoformat(),gate_rc=p.returncode,unmet_count=len(gate['unmet']),categories=dict(counts)),indent=2))
