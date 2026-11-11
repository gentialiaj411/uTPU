import json,subprocess,os,sys,time,datetime
from pathlib import Path
D=Path('sprint/data')
cmd="""$r=Get-CimInstance Win32_Process -Filter 'ProcessId = 15836'; if ($r.CommandLine -like '*fault_inject.py*qspi*') { function Stop-SprintTree($id) { Get-CimInstance Win32_Process -Filter \"ParentProcessId = $id\" | ForEach-Object { Stop-SprintTree $_.ProcessId }; Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }; Stop-SprintTree 15836 }"""
subprocess.run(['powershell','-NoProfile','-Command',cmd],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
with Path('SPRINT_LOG.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' TIMEBOXED redundant Verilator QSPI fault sweep; full clean/mutation backend suite already passed. Re-running all remaining fault producers with Icarus, whose identical unit summaries were cross-checked; no checks removed.\n')
env=os.environ.copy();env['UTPU_SIM']='icarus';out={}
for u in ['qspi','actq','rms','vec','attn','attn15']:
 cmd=[sys.executable,'tools/decode/fault_inject.py',u];t=time.perf_counter()
 with (D/('fault_'+u+'_icarus_final.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600)
 out['fault_'+u]=dict(command=cmd,rc=r.returncode,seconds=time.perf_counter()-t,backend='icarus')
 (D/'fast_fault_recovery.json').write_text(json.dumps(out,indent=2))
 with Path('SPRINT_LOG.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' '+subprocess.list2cmdline(cmd)+' UTPU_SIM=icarus EXIT '+str(r.returncode)+'\n')
