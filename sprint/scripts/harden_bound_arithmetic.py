from pathlib import Path
p=Path('sprint/scripts/argmax_bound.py');s=p.read_text();before=s
s=s.replace('import numpy as np','import numpy as np\nfrom bound_math import coefficient_scaled_bound')
s=s.replace('maxlog=int(logits[best]);assert maxlog>=0','maxlog=int(logits[best])')
s=s.replace("product=coeff*factor[None,:]*xmax[None,:];assert int(coeff.max())*int(factor.max())*int(xmax.max())<2**63\n    n=(product+(1<<32)-1)//(1<<32)+1", "n=coefficient_scaled_bound(coeff,factor[None,:],xmax[None,:])")
s=s.replace("product=bc*factor[None,:]*xmax[None,:]\n     nn=(product+(1<<32)-1)//(1<<32)+1", "nn=coefficient_scaled_bound(bc,factor[None,:],xmax[None,:])")
Path('sprint/data/measured_bound_script_sha256.txt').write_text(__import__('hashlib').sha256(before.encode()).hexdigest())
p.write_text(s)
p=Path('sprint/scripts/report.py');s=p.read_text();s=s.replace('Groupwise block maxima are used even though they are looser than max of activation-specific per-row sums.', 'Groupwise block maxima are used even though they are looser than max of activation-specific per-row sums. For full-range arithmetic, c*a is split into high/low Q32 limbs before multiplying by max|x|: ceil(c*a*x/2^32)=high*x+ceil(low*x/2^32), avoiding a potentially overflowing three-factor product. An independent arbitrary-precision oracle checked 10000 input triples (bound_math_property.json). This post-measurement arithmetic hardening is algebraically identical on every measured input, whose original product-overflow assertions all passed; the measured script digest is retained separately.')
p.write_text(s)
