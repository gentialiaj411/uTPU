from dataclasses import dataclass
from typing import Any


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
