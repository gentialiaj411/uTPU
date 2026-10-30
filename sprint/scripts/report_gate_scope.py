from pathlib import Path
p=Path('sprint/scripts/report.py');s=p.read_text().replace("if isinstance(v,dict) and 'rc' in v:rows.append([k,v['rc'],f\"{v.get('seconds',0):.2f}\",' '.join(v.get('command',[]))])", "if isinstance(v,dict) and 'rc' in v:rows.append([k,v['rc'] if v['rc'] is not None else 'NOT RUN',f\"{v['seconds']:.2f}\" if 'seconds' in v else 'NOT RECORDED',' '.join(v.get('command',[]))])")
needle="emit('## Diff and numbers versus audit-fixes')"
s=s.replace(needle,"unmet=load('gate_unmet_categories.json',{})\nif unmet:table(['Unmet artifact category','Dependency/gate reasons DERIVED'],sorted(unmet.items()))\nemit('These counts describe strict-gate unmet dependency reasons, not numerical mismatches. The raw complete list is in strict_gate_closed.log. Full fault sweeps that did not finish remain stale; they are queued in OWNER_RUN2 before the physical jobs.\\n')\n"+needle)
p.write_text(s)
