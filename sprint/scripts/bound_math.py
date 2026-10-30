"""Integer bound helpers: no full-width c*a*x product is required."""
import numpy as np

def coefficient_scaled_bound(coefficients, scales, maxima):
    c=np.asarray(coefficients,dtype=np.int64);a=np.asarray(scales,dtype=np.int64);x=np.asarray(maxima,dtype=np.int64)
    assert np.all((c>=0)&(c<(1<<32))) and np.all((a>=0)&(a<(1<<31))) and np.all((x>=0)&(x<=127))
    product=c*a
    return (product>>32)*x+(((product&((1<<32)-1))*x+(1<<32)-1)>>32)+1
