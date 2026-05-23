from typing import Any, Dict, Protocol

from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from cuda_blocked_fc_backend import CUDABackendLowerer
from lowering_types import BlockedFCLoweringRequest


class BackendLowerer(Protocol):
    def lower_blocked_fc(self, request: BlockedFCLoweringRequest) -> Dict[str, Any]:
        ...
    def lower_graph_op(self, op: Any) -> Dict[str, Any]:
        ...


class UTPUBackendLowerer:
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
