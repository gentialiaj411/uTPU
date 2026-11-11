import numpy as np
from tools.decode.pack_decode_image import tensor_bytes
from firmware.host.numerics_v2.runq_runtime import _linear_high,_quantize
from firmware.host.numerics_v2.fixed import rshift_rne

def test_compact_metadata_preserves_integer_gemv():
    rng=np.random.default_rng(20261030)
    for trial in range(100):
        weights=rng.integers(-7,8,(16,32),dtype=np.int8)
        multiplier=rng.integers(0,1<<30,(16,1),dtype=np.int64)
        if trial%2==0:multiplier=(multiplier & ~65535)|32768
        record=dict(weight=weights,multiplier=multiplier,shift=rng.integers(32,48,(16,1),dtype=np.int64),groups=np.array([[0],[32]]),qmax=7)
        packed=tensor_bytes(record)[0]
        # Decode the actual packed bytes independently of the packer.
        expanded=np.array([int.from_bytes(packed[3*r:3*r+2],'little')<<16 for r in range(16)],dtype=np.int64)[:,None]
        shifts=np.array([packed[3*r+2] for r in range(16)],dtype=np.int64)[:,None]
        activation=_quantize(rng.integers(-100000,100001,(32,),dtype=np.int32))
        actual=_linear_high(activation,dict(record,multiplier=expanded,shift=shifts))
        expected=_linear_high(activation,record)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(rshift_rne(expanded,16),rshift_rne(multiplier,16))
