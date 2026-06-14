from typing import Any, Dict, Protocol

import numpy as np

from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from cuda_blocked_fc_backend import CUDABackendLowerer
from lowering_types import BatchedMatmulLoweringRequest, BlockedFCLoweringRequest, Conv2DIm2ColLoweringRequest
from requantization import RequantParams
from utpu_batched_matmul_lowering import DEFAULT_BMM_CFG, lower_batched_matmul_utpu
from utpu_conv2d_lowering import DEFAULT_CONV_CFG, lower_conv2d_im2col_utpu


class BackendLowerer(Protocol):
    def lower_request(self, request: Any) -> Dict[str, Any]:
        ...
    def lower_graph_op(self, op: Any) -> Dict[str, Any]:
        ...


class UTPUBackendLowerer:
    def lower_request(self, request: Any) -> Dict[str, Any]:
        if isinstance(request, BlockedFCLoweringRequest):
            return self.lower_blocked_fc(request)
        if isinstance(request, BatchedMatmulLoweringRequest):
            return self.lower_batched_matmul(request)
        if isinstance(request, Conv2DIm2ColLoweringRequest):
            return self.lower_conv2d(request)
        raise TypeError(f"Unsupported uTPU lowering request type: {type(request).__name__}")

    def lower_blocked_fc(self, request: BlockedFCLoweringRequest) -> Dict[str, Any]:
        return lower_blocked_fc_program_utpu(
            weights_int4=request.weights_int4,
            activations_int4=request.activations_int4,
            out_features=request.out_features,
            in_features=request.in_features,
            array_size=request.array_size,
            apply_relu=request.apply_relu,
            apply_quant=request.apply_quant,
            weight_addr=request.weight_addr,
            input_addr=request.input_addr,
            result_addr=request.result_addr,
        )

    def lower_conv2d(self, request: Conv2DIm2ColLoweringRequest) -> Dict[str, Any]:
        lowered = lower_conv2d_im2col_utpu(
            request.input_nchw,
            request.weight_oihw,
            bias=request.bias,
            stride=request.stride,
            padding=request.padding,
            dilation=request.dilation,
            groups=request.groups,
            array_size=request.array_size,
            cfg=DEFAULT_CONV_CFG,
            requant_params=RequantParams(multiplier=1, right_shift=0, enable=True),
        )
        return {
            "mode": "utpu_conv2d_im2col",
            "mapping": lowered.mapping,
            "input_shape": list(lowered.input_shape),
            "weight_shape": list(lowered.weight_shape),
            "output_shape": list(lowered.output_shape),
            "stride": list(lowered.stride),
            "padding": list(lowered.padding),
            "groups": int(lowered.groups),
            "cfg": dict(lowered.cfg),
            "requant_params": dict(lowered.requant_params),
            "program_count": len(lowered.programs),
            "program_instruction_words": [int(p.program_instruction_words) for p in lowered.programs],
            "programs": [p.program for p in lowered.programs],
            "executable_on_current_fpga_path": True,
            "blockers": [],
        }

    def lower_batched_matmul(self, request: BatchedMatmulLoweringRequest) -> Dict[str, Any]:
        lowered = lower_batched_matmul_utpu(
            request.lhs_dynamic,
            request.rhs_dynamic,
            array_size=request.array_size,
            cfg=DEFAULT_BMM_CFG,
            requant_params=RequantParams(multiplier=1, right_shift=0, enable=True),
            apply_relu=request.apply_relu,
            weight_addr=request.weight_addr,
            input_addr=request.input_addr,
            result_addr=request.result_addr,
        )
        return {
            "mode": "utpu_batched_matmul_dynamic_dynamic",
            "input_shape_lhs": list(np.asarray(request.lhs_dynamic).shape),
            "input_shape_rhs": list(np.asarray(request.rhs_dynamic).shape),
            "output_shape": list(lowered.output_shape),
            "cfg": dict(lowered.cfg),
            "requant_params": dict(lowered.requant_params),
            "program_count": len(lowered.programs),
            "program_instruction_words": [int(p.program_instruction_words) for p in lowered.programs],
            "programs": [p.program for p in lowered.programs],
            "dynamic_operand_contract": {
                "lhs_streamed_rows": True,
                "rhs_runtime_stationary": True,
                "rhs_transposed_into_stationary_matrix": True,
            },
            "executable_on_current_fpga_path": True,
            "blockers": [],
        }

    def lower_graph_op(self, op: Any) -> Dict[str, Any]:
        return {
            "mode": "utpu_graph_op_unsupported",
            "op_kind": str(op.op),
            "generated_by_compiler": True,
            "executable_on_current_fpga_path": False,
            "blockers": [f"uTPU backend currently supports only blocked-FC linear ops; unsupported graph op '{op.op}'"],
        }


def create_backend_lowerer(name: str) -> BackendLowerer:
    n = (name or "utpu").strip().lower()
    if n == "utpu":
        return UTPUBackendLowerer()
    if n == "cuda":
        return CUDABackendLowerer()
    raise ValueError(f"Unknown backend '{name}'. Expected one of: utpu, cuda")
