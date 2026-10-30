import sys,hashlib,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from bound_math import coefficient_scaled_bound
rng=np.random.default_rng(20261030)
c=rng.integers(0,1<<32,10000,dtype=np.int64);a=rng.integers(0,1<<31,10000,dtype=np.int64);x=rng.integers(0,128,10000,dtype=np.int64)
actual=coefficient_scaled_bound(c,a,x)
expected=np.array([(int(cc)*int(aa)*int(xx)+(1<<32)-1)//(1<<32)+1 for cc,aa,xx in zip(c,a,x)],dtype=np.int64)
np.testing.assert_array_equal(actual,expected)
Path('sprint/data/bound_math_property.json').write_text(json.dumps(dict(status='pass',cases=len(c),oracle='Python arbitrary-precision integer ceil division',command='python sprint/scripts/bound_math_property.py'),indent=2))
print('SUMMARY bound_limb_property cases='+str(len(c))+' errors=0')
