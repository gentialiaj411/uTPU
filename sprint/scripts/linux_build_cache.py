from pathlib import Path
p=Path('tools/decode/sim.py');s=p.read_text();s=s.replace('    obj = BUILD / (tag+"_obj")','    obj = BUILD / (tag+"_obj")\n    obj_linux = "/tmp/utpu-perf/"+tag+"_obj"')
s=s.replace('"--Mdir", linux_path(obj)', '"--Mdir", obj_linux')
s=s.replace('command = "export VERILATOR_ROOT="', 'command = "mkdir -p /tmp/utpu-perf; export VERILATOR_ROOT="')
s=s.replace('linux_path(obj/("V"+top))','obj_linux+"/V"+top');p.write_text(s)
p=Path('sprint/scripts/crosscheck.py');s=p.read_text().replace('records=[]', "records=json.loads(Path('sprint/data/crosscheck_progress.json').read_text()) if Path('sprint/data/crosscheck_progress.json').exists() else []")
s=s.replace("for gold in ['gemv15','rms','vec','attn','attn15']:\n records.append", "for gold in ['gemv15','rms','vec','attn','attn15']:\n if any(r['label']=='golden_'+gold and r['returncode']==0 for r in records):continue\n records.append")
s=s.replace("  tag=f'{label}_{backend}';records.append", "  tag=f'{label}_{backend}'\n  if any(r['label']==tag and r['returncode']==0 for r in records) and Path(f'sprint/data/{tag}.json').exists():continue\n  records.append")
p.write_text(s)
