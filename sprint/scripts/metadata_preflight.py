import sys,csv,hashlib,json
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools/decode'))
import numpy as np
plan=(R/'sprint/scripts/implement_metadata.py').read_text().split("p=Path('rtl/decode/top_decode.sv')")[0]
captured={}
def capture(path,text,*args,**kwargs):captured[str(path)]=text;return len(text)
with patch.object(Path,'write_text',capture):exec(compile(plan,'metadata_plan_preview','exec'),{})
assert len(captured)==1
source=next(iter(captured.values()));namespace={'__name__':'planned_metadata_readonly','__file__':str(R/'tools/decode/pack_decode_image.py')};exec(compile(source,namespace['__file__'],'exec'),namespace)
from tools.decode.pack_decode_image import tensor_bytes,load_export_15m
from tools.decode.harness import load_runtime_260k
from firmware.host.numerics_v2.runq_runtime import _linear_high,_quantize,_small_multiplier
rng=np.random.default_rng(20261030);rows=[]
for model,export in [('260k',load_runtime_260k()[0]),('15m',load_export_15m())]:
 records=[('classifier',export['classifier'])]+[(str(i)+'.'+name,record) for i,layer in enumerate(export['layers']) for name,record in layer.items() if isinstance(record,dict) and 'weight' in record]
 for name,record in records:
  old=tensor_bytes(record)[0];new,_,width,groups=namespace['tensor_bytes'](record)
  n=record['weight'].shape[0];compact=np.empty((n,groups),np.int64);shifts=np.empty_like(compact)
  tile_bytes=groups*16*3+16*(width//2 if record['qmax']==7 else width)
  for tile in range((n+15)//16):
   for group in range(groups):
    for lane in range(min(16,n-tile*16)):
     offset=tile*tile_bytes+(group*16+lane)*3
     compact[tile*16+lane,group]=int.from_bytes(new[offset:offset+2],'little')<<16;shifts[tile*16+lane,group]=new[offset+2]
  np.testing.assert_array_equal(_small_multiplier(compact),_small_multiplier(record['multiplier']))
  np.testing.assert_array_equal(shifts,record['shift'])
  for _ in range(2):
   activation=_quantize(rng.integers(-1000,1001,width,dtype=np.int32))
   np.testing.assert_array_equal(_linear_high(activation,record),_linear_high(activation,dict(record,multiplier=compact,shift=shifts)))
  rows.append(dict(model=model,tensor=name,before_bytes=len(old),after_bytes=len(new),saved_bytes=len(old)-len(new),max_effective_multiplier=int(_small_multiplier(record['multiplier']).max()),all_metadata_equal=True,integer_forward_equal=True))
with (R/'sprint/data/metadata_preflight.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(R/'sprint/data/planned_packer_normalized_sha256.txt').write_text(hashlib.sha256(source.encode()).hexdigest())
print('SUMMARY tensors='+str(len(rows))+' all_metadata_equal=1 integer_forward_equal=1 saved_15m='+str(sum(r['saved_bytes'] for r in rows if r['model']=='15m')))
