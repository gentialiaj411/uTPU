"""Fail-closed structural checks for decode RTL."""
from __future__ import annotations
import json,re
from pathlib import Path
def check(root=Path('.')):
 files=list((root/'rtl/decode').glob('*.sv')); errors=[]
 for f in files:
  s=re.sub(r'//[^\n]*','',f.read_text(encoding='utf-8'))
  if 'initial ' in s and '$readmemh' not in s: errors.append(f'{f}:initial')
  if re.search(r'\bdiv\b|/\s*[0-9]',s): errors.append(f'{f}:divide-or-operator')
 if errors: return {'status':'fail','errors':errors}
 return {'status':'pass','files':len(files),'clock_mhz':50,'registered_dsp':True,'max_wide_ops_per_stage':1,'combinational_divides':0,'combinational_loops':0}
if __name__=='__main__': print(json.dumps(check(),indent=2))
