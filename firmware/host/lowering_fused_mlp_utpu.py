"""Fused MLP uTPU lowering with optional 2-PE schedule (simulator-validated)."""

from typing import Any, Dict, Optional

import numpy as np

from isa_encoder import IsaConfig
from program_loader import ProgramLoader


def lower_fused_mlp_program_utpu(
    fc1_weights_int4,
    fc2_weights_int4,
    input_activations_int4,
    residual_input_int4=None,
    array_size: int = 16,
    fc1_apply_relu: bool = True,
    fc2_apply_relu: bool = False,
    apply_quant: bool = True,
    result_addr: int = ProgramLoader.BUFFER_SECTION_C,
    residual_addr: int = ProgramLoader.BUFFER_SECTION_D,
    num_pe: int = 1,
    cfg: Optional[IsaConfig] = None,
) -> Dict[str, Any]:
    """Lower a two-layer fused MLP into uTPU ISA bytes.

    ``num_pe=1`` preserves the existing single-PE compressed fused schedule.
    ``num_pe=2`` emits a simulator-validated dual-PE schedule when FC1 has at
  least two K-blocks; otherwise it falls back to the 1-PE schedule.
    When ``residual_input_int4`` is provided the FC1 finalize step adds the
    residual tensor before activation. That path requires widened encoding
    support (``cfg.address_width > 9``) so the extra residual operand address
    can be encoded without perturbing the legacy byte-identical layout.
    """
    if int(num_pe) not in (1, 2):
        raise ValueError(f"num_pe must be 1 or 2, got {num_pe}")
    loader = ProgramLoader(uart=None, verbose=False, cfg=cfg)
    return loader.build_full_inference_program_compressed_fused(
        fc1_weights_int4=fc1_weights_int4,
        fc2_weights_int4=fc2_weights_int4,
        input_activations_int4=input_activations_int4,
        residual_input_int4=residual_input_int4,
        array_size=array_size,
        fc1_apply_relu=fc1_apply_relu,
        fc2_apply_relu=fc2_apply_relu,
        apply_quant=apply_quant,
        result_addr=result_addr,
        residual_addr=residual_addr,
        num_pe=int(num_pe),
    )
