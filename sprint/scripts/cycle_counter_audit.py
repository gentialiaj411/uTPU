import json,re
from pathlib import Path
D=Path('sprint/data');out=[]
for n in ['15m_big.json','15m_big_compact.json']:
 d=json.loads((D/n).read_text());signed=int(re.search(r'cycles=(-?\d+)',d['summary']).group(1));total=round(d['cycles_per_token_mean']*d['steps'])
 out.append(dict(artifact=n,summary_signed_cycles=signed,derived_cycles=total,formula='round(cycles_per_token_mean * steps)',signed_modulo_check=((total+(1<<31))%(1<<32))-(1<<31)==signed))
(D/'cycle_counter_overflow.json').write_text(json.dumps(out,indent=2));print('SUMMARY',out)
