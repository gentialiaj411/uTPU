from pathlib import Path
p=Path('sprint/scripts/finalize.py');s=p.read_text();s=s.replace("23124:'argmax_bound'","33636:'argmax_bound'").replace('>3.1*3600','>2.5*3600')
s=s.replace('handles={pid:K.OpenProcess', '''snapshot=subprocess.run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process -Filter \\\"name = 'python.exe'\\\" | Select-Object ProcessId,CommandLine | ConvertTo-Json"],capture_output=True,text=True)
current=json.loads(snapshot.stdout or '[]')
if isinstance(current,dict):current=[current]
jobs={pid:name for pid,name in jobs.items() if any(row['ProcessId']==pid and 'sprint/scripts/'+name+'.py' in (row['CommandLine'] or '') for row in current)}
handles={pid:K.OpenProcess''')
p.write_text(s)
p=Path('SPRINT_STATE.md');s=p.read_text();s=s.replace('P3 DONE / NO-GO: final sound variants completed 50x120 positions, zero rows skipped in all variants, exact equality; invalid preliminary CSV excluded.', 'P3 IN-PROGRESS diversity repair: original 50 records contained only 9 unique prefixes. Reusing nine distinct results and measuring 41 additional distinct prefixes; final decision awaits repaired 6000-position sweep.');p.write_text(s)
