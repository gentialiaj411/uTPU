import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from tools.decode.pack_decode_image import load_export_15m
from firmware.host.numerics_v2.runq_runtime import _small_multiplier
r=load_export_15m()['classifier'];w=r['weight'].astype(np.int64).reshape(32000,9,32);m=_small_multiplier(r['multiplier']);s=r['shift'].astype(np.int64)-16;l1=np.abs(w).sum(2)
maximum=int(l1.max())*127*((1<<31)-1)*int(m.max())
assert np.all((s>=0)&(s<=63)) and np.all(m>=0) and maximum<(1<<63)
result=dict(status='pass',shift_min=int(s.min()),shift_max=int(s.max()),effective_multiplier_max=int(m.max()),weight_l1_max=int(l1.max()),max_legal_scale=(1<<31)-1,max_linear_product_bound=maximum,int64_product_safe=True,command='python sprint/scripts/bound_domain_proof.py')
Path('sprint/data/bound_domain_proof.json').write_text(json.dumps(result,indent=2))
print('SUMMARY bound_domain shifts='+str(result['shift_min'])+'..'+str(result['shift_max'])+' int64_product_safe=1')
