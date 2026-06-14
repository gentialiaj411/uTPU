from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class BlockedFCLoweringRequest:
    weights_int4: Any
    activations_int4: Any
    out_features: int
    in_features: int
    array_size: int
    apply_relu: bool
    apply_quant: bool
    weight_addr: int
    input_addr: int
    result_addr: int


@dataclass(frozen=True)
class Conv2DIm2ColLoweringRequest:
    input_nchw: Any
    weight_oihw: Any
    bias: Optional[Any]
    stride: Any
    padding: Any
    dilation: Any
    groups: int
    array_size: int


@dataclass(frozen=True)
class BatchedMatmulLoweringRequest:
    lhs_dynamic: Any
    rhs_dynamic: Any
    array_size: int
    apply_relu: bool
    apply_quant: bool
    weight_addr: int
    input_addr: int
    result_addr: int
