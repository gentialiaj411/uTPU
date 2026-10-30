from pathlib import Path
p=Path('sprint/scripts/argmax_bound.py');s=p.read_text()
start=s.index('old_prompts=json.loads(');end=s.index('from tools.llama2c.tokenizer import Tokenizer',start)
s=s[:start]+'prompts=[]\n'+s[end:]
start=s.index('records=[]\nif (D/');end=s.index('deadline=time.monotonic()',start)
s=s[:start]+"records=[]\n(D/'argmax_unique_complete.json').write_text(json.dumps(dict(status='in-progress')))\n"+s[end:]
s=s.replace(' if pi<reused:continue\n','')
s=s.replace("dict(prompts=pi+1,steps=(pi+1)*120,records=len(records))", "dict(prompts_evaluated=pi+1,steps=len(records)//36,records=len(records))")
s=s.replace('(D/"argmax_unique_complete.json").write_text(json.dumps(dict(unique_prompts=len(prompts),records=len(records))))', "complete=len(records)==50*120*36\n(D/'argmax_unique_complete.json').write_text(json.dumps(dict(status='pass' if complete else 'timeboxed',unique_prompts=len(prompts),positions=len(records)//36,records=len(records))))\nif not complete:raise SystemExit(124)")
p.write_text(s)
p=Path('sprint/scripts/report.py');s=p.read_text();s=s.replace('Nine complete measurements were reused', 'Nine complete measurements were reused')
s=s.replace('This is a screening measurement of reads and decisions, NOT a Python inference speedup.', 'This is a screening measurement of reads and decisions, NOT a Python inference speedup. The current reproduction script always evaluates all distinct prompts from scratch; the one-off diversity-repair cache is not required on a fresh checkout, and partial/timeboxed runs return nonzero with actual position counts.')
s=s.replace('Build failures preserve CalledProcessError', 'The old vvp_command field is retained for artifact compatibility; mandatory tool:simulator provenance identifies the actual backend. Reproduce using the exact Python producer commands, not that legacy field. Build failures preserve CalledProcessError')
p.write_text(s)
