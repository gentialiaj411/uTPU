from pathlib import Path
p=Path('sprint/scripts/finalize.py');s=p.read_text();s=s.replace("jobs={8448:'crosscheck',6420:'regress15',23124:'argmax_bound',38748:'engineering_driver'}", "new=json.loads(Path('sprint/data/restarted_jobs.json').read_text())\njobs={new['crosscheck']:'crosscheck',6420:'regress15',23124:'argmax_bound',new['engineering']:'engineering_driver'}")
s=s.replace('>4*3600','>3.5*3600');p.write_text(s)
