import sys,csv,json
from pathlib import Path
sys.path.insert(0,str(Path('tools/decode').resolve()))
import fault_inject
rows=[]
for unit,(_,rtl,faults) in fault_inject.FAULTS.items():
 source=Path(rtl).read_text()
 for name,entry in faults.items():
  if entry[0]!='rtl':continue
  rows.append(dict(unit=unit,fault=name,target=rtl,needle_matches=source.count(entry[1])))
with Path('sprint/data/fault_needle_audit.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
missing=[x for x in rows if not x['needle_matches']]
Path('sprint/data/fault_needle_missing.json').write_text(json.dumps(missing,indent=2))
print('SUMMARY audited='+str(len(rows))+' missing='+str(len(missing)))
if missing:raise SystemExit(1)
