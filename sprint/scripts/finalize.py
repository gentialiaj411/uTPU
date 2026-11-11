import ctypes,time,json,subprocess,datetime
from pathlib import Path
K=ctypes.windll.kernel32
K.OpenProcess.restype=ctypes.c_void_p
new=json.loads(Path('sprint/data/restarted_jobs.json').read_text())
jobs={33636:'argmax_bound',new['engineering']:'final_flash_repair'}
extra=json.loads(Path('sprint/data/compact_big_job.json').read_text());jobs[extra['pid']]='compact_big'
snapshot=subprocess.run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json"],capture_output=True,text=True)
current=json.loads(snapshot.stdout or '[]')
if isinstance(current,dict):current=[current]
jobs={pid:name for pid,name in jobs.items() if any(row['ProcessId']==pid and 'sprint/scripts/'+name+'.py' in (row['CommandLine'] or '') for row in current)}
handles={pid:K.OpenProcess(0x100000,False,pid) for pid in jobs};start=time.monotonic();timedout=False
while True:
 active=[pid for pid,h in handles.items() if h and K.WaitForSingleObject(ctypes.c_void_p(h),0)==258]
 Path('sprint/data/completion_status.json').write_text(json.dumps(dict(active=[jobs[p] for p in active],seconds=time.monotonic()-start,timestamp=datetime.datetime.now().isoformat())))
 if not active:break
 if time.time()>json.loads(Path('sprint/data/wall_deadline.json').read_text())['unix']+300:
  timedout=True
  # Only terminate roots whose current command still names our known sprint script; descendants first.
  for pid in active:
   expected=jobs[pid]
   command=f"$r=Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; if ($r.CommandLine -like '*sprint/scripts/{expected}.py*') {{ function Stop-SprintTree($id) {{ Get-CimInstance Win32_Process -Filter \"ParentProcessId = $id\" | ForEach-Object {{ Stop-SprintTree $_.ProcessId }}; Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }}; Stop-SprintTree {pid} }}"
   subprocess.run(['powershell','-NoProfile','-Command',command],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  break
 time.sleep(30)
for h in handles.values():
 if h:K.CloseHandle(ctypes.c_void_p(h))
with Path('SPRINT_LOG.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' Finalization: all launched jobs ended or were explicitly TIMEBOXED at wall deadline. timedout='+str(timedout)+'\n')
for name,cmd in [('compare_last',[ 'python','sprint/scripts/compare.py']),('report',[ 'python','sprint/scripts/report.py'])]:
 with open('sprint/data/'+name+'.log','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
Path('sprint/data/finalization_complete.json').write_text(json.dumps(dict(timedout=timedout,timestamp=datetime.datetime.now().isoformat())))

