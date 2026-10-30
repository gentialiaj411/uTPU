import json,csv
from pathlib import Path
D=Path('sprint/data');rows=[]
def extract(o,p=''):
 out={}
 if isinstance(o,dict):
  for k,v in o.items():
   if k in ['provenance','wall_seconds','vvp_command']:continue
   path=p+'/'+k
   if isinstance(v,(dict,list)):out.update(extract(v,path))
   elif k in ['summary','mutation_summary'] or any(x in k for x in ['mismatch','cycles','errors']):
    if 'first_' not in k:out[path]=v
 elif isinstance(o,list):
  for i,v in enumerate(o):out.update(extract(v,p+'/'+str(i)))
 return out
for a in D.glob('*_icarus.json'):
 b=D/(a.stem.replace('_icarus','_verilator')+'.json')
 if not b.exists():continue
 ia=json.loads(a.read_text());vb=json.loads(b.read_text());aa=extract(ia);bb=extract(vb);diff={k:[aa.get(k),bb.get(k)] for k in aa.keys()|bb.keys() if aa.get(k)!=bb.get(k)}
 rows.append(dict(unit=a.stem[:-7],icarus=ia.get('status'),verilator=vb.get('status'),identical=not diff,differences=json.dumps(diff)))
with (D/'crosscheck_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['unit','icarus','verilator','identical','differences']);w.writeheader();w.writerows(rows)
print('SUMMARY compared',len(rows),'identical',sum(x['identical'] for x in rows))
