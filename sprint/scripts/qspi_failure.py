import json
from pathlib import Path
x=json.loads(Path('sprint/data/qspi_verilator.json').read_text())
def walk(v,p=''):
 if isinstance(v,dict):
  for k,z in v.items():
   if z is False:print(p+'/'+k,False)
   elif k in ['variant','ok','negative_model_checks','negative_checks']:pass
   if isinstance(z,(dict,list)) and k!='provenance':walk(z,p+'/'+k)
 elif isinstance(v,list):
  for i,z in enumerate(v):walk(z,p+'/'+str(i))
walk(x)
print('KEYS',list(x))
