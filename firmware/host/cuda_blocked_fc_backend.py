import ctypes
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from lowering_types import BlockedFCLoweringRequest
from compiler_abstractions import (
    BlockedFCProblem,
    build_blocked_fc_schedule,
    cuda_target_desc,
)


def _cuda_kernel_source() -> str:
    return r"""
extern "C" __global__
void blocked_fc_int4_kernel(
    const signed char* __restrict__ w,   // [out_padded, in_padded]
    const signed char* __restrict__ x,   // [in_padded]
    signed int* __restrict__ accum,      // [out_padded]
    int in_padded,
    int out_elems
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_elems) return;
    int acc = 0;
    const signed char* wrow = w + row * in_padded;
    for (int k = 0; k < in_padded; ++k) {
        acc += (int)wrow[k] * (int)x[k];
    }
    accum[row] = acc;
}
"""


def _leaky_relu_quantize_int4(x: np.ndarray, apply_relu: bool) -> np.ndarray:
    out = x.astype(np.float32)
    if apply_relu:
        out = np.where(out >= 0, out, out * 0.25)
    out = np.clip(np.round(out), -8, 7).astype(np.int8)
    return out


def _numpy_blocked_fc_reference(
    weights_int4,
    activations_int4,
    out_features: int,
    in_features: int,
    array_size: int,
    apply_relu: bool,
    apply_quant: bool,
) -> Dict[str, Any]:
    problem = BlockedFCProblem(out_features=out_features, in_features=in_features, array_size=array_size)
    schedule = build_blocked_fc_schedule(problem=problem, target=cuda_target_desc(array_size=array_size))
    w = np.asarray(weights_int4, dtype=np.int8)
    x = np.asarray(activations_int4, dtype=np.int8).flatten()

    w_pad = np.zeros((schedule.out_padded, schedule.in_padded), dtype=np.int8)
    x_pad = np.zeros(schedule.in_padded, dtype=np.int8)
    w_pad[:out_features, :in_features] = w
    x_pad[:in_features] = x

    accum = (w_pad.astype(np.int32) @ x_pad.astype(np.int32)).astype(np.int32)
    out = accum.copy()
    if apply_quant:
        out = _leaky_relu_quantize_int4(out, apply_relu=apply_relu).astype(np.int8)

    return {
        "schedule": schedule,
        "weights_padded": w_pad,
        "inputs_padded": x_pad,
        "accum_int32_padded": accum,
        "output_padded": out,
        "output_unpadded": out[:out_features],
    }


@dataclass(frozen=True)
class CUDAEnvironmentStatus:
    cuda_python_available: bool
    runtime_available: bool
    reason: Optional[str]


def detect_cuda_environment() -> CUDAEnvironmentStatus:
    try:
        try:
            from cuda import cuda as _cuda  # noqa: F401
            from cuda import nvrtc as _nvrtc  # noqa: F401
        except Exception:
            from cuda.bindings import driver as _cuda  # noqa: F401
            from cuda.bindings import nvrtc as _nvrtc  # noqa: F401
        # Validate that runtime dynamic libraries are discoverable.
        try:
            _nvrtc.nvrtcVersion()
        except Exception as e:
            return CUDAEnvironmentStatus(
                cuda_python_available=True,
                runtime_available=False,
                reason=f"CUDA runtime/NVRTC unavailable: {e}",
            )
    except Exception as e:
        return CUDAEnvironmentStatus(
            cuda_python_available=False,
            runtime_available=False,
            reason=f"cuda-python import failed: {e}",
        )
    return CUDAEnvironmentStatus(
        cuda_python_available=True,
        runtime_available=True,
        reason=None,
    )


class CUDABackendLowerer:
    """
    CUDA lowering metadata + kernel/runtime emission.
    Compile/launch requires cuda-python + installed NVIDIA driver/runtime.
    """

    def lower_blocked_fc(self, request: BlockedFCLoweringRequest) -> Dict[str, Any]:
        problem = BlockedFCProblem(
            out_features=request.out_features,
            in_features=request.in_features,
            array_size=request.array_size,
        )
        schedule = build_blocked_fc_schedule(problem=problem, target=cuda_target_desc(array_size=request.array_size))
        env = detect_cuda_environment()
        return {
            "mode": "cuda_blocked_fc",
            "array_size": int(request.array_size),
            "out_blocks": int(schedule.out_blocks),
            "in_blocks": int(schedule.in_blocks),
            "out_padded": int(schedule.out_padded),
            "in_padded": int(schedule.in_padded),
            "memory_scopes": {
                "weights": schedule.weight_scope.value,
                "inputs": schedule.input_scope.value,
                "accum": schedule.accum_scope.value,
            },
            "kernel_name": "blocked_fc_int4_kernel",
            "kernel_source": _cuda_kernel_source(),
            "executable_on_current_cuda_path": bool(env.runtime_available),
            "blockers": [] if env.runtime_available else [env.reason or "Unknown CUDA runtime issue"],
            "executable_on_current_fpga_path": False,
            "int32_accumulation_supported": True,
            "quantize_after_accumulation_supported": bool(request.apply_quant),
        }


class CUDABlockedFCExecutor:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CUDABlockedFCExecutor] {msg}")

    def execute(self, request: BlockedFCLoweringRequest) -> Dict[str, Any]:
        # Always produce a deterministic reference path for parity/debug.
        ref = _numpy_blocked_fc_reference(
            request.weights_int4,
            request.activations_int4,
            request.out_features,
            request.in_features,
            request.array_size,
            request.apply_relu,
            request.apply_quant,
        )

        env = detect_cuda_environment()
        if not env.runtime_available:
            return {
                "executed": False,
                "backend": "cuda",
                "reason": env.reason,
                "numpy_reference_output": ref["output_unpadded"].tolist(),
                "numpy_reference_accum_int32": ref["accum_int32_padded"][:request.out_features].tolist(),
            }
        try:
            try:
                from cuda import cuda, nvrtc
            except Exception:
                from cuda.bindings import driver as cuda
                from cuda.bindings import nvrtc

            def _check_nvrtc(err, ctx: str):
                if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    raise RuntimeError(f"{ctx} failed: {nvrtc.nvrtcGetErrorString(err)[1].decode('utf-8')}")

            def _check_cuda(err, ctx: str):
                if err != cuda.CUresult.CUDA_SUCCESS:
                    name = cuda.cuGetErrorName(err)[1].decode("utf-8")
                    desc = cuda.cuGetErrorString(err)[1].decode("utf-8")
                    raise RuntimeError(f"{ctx} failed: {name} ({desc})")

            schedule = ref["schedule"]
            w_pad = ref["weights_padded"]
            x_pad = ref["inputs_padded"]
            out_elems = schedule.out_padded

            src = _cuda_kernel_source().encode("utf-8")
            err, prog = nvrtc.nvrtcCreateProgram(src, b"blocked_fc.cu", 0, [], [])
            _check_nvrtc(err, "nvrtcCreateProgram")
            opts = [b"--std=c++11"]
            err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
                log = b""
                if log_size > 1:
                    log_buf = bytearray(log_size)
                    nvrtc.nvrtcGetProgramLog(prog, log_buf)
                    log = bytes(log_buf)
                raise RuntimeError(f"nvrtcCompileProgram failed: {log.decode('utf-8', errors='replace')}")
            err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
            _check_nvrtc(err, "nvrtcGetPTXSize")
            ptx = bytearray(ptx_size)
            err, = nvrtc.nvrtcGetPTX(prog, ptx)
            _check_nvrtc(err, "nvrtcGetPTX")
            nvrtc.nvrtcDestroyProgram(prog)

            err, = cuda.cuInit(0)
            _check_cuda(err, "cuInit")
            err, dev = cuda.cuDeviceGet(0)
            _check_cuda(err, "cuDeviceGet")
            err, ctx = cuda.cuCtxCreate(None, 0, dev)
            _check_cuda(err, "cuCtxCreate")
            try:
                err, mod = cuda.cuModuleLoadData(bytes(ptx))
                _check_cuda(err, "cuModuleLoadData")
                err, fn = cuda.cuModuleGetFunction(mod, b"blocked_fc_int4_kernel")
                _check_cuda(err, "cuModuleGetFunction")

                w_nbytes = int(w_pad.size)
                x_nbytes = int(x_pad.size)
                out_nbytes = int(out_elems * np.dtype(np.int32).itemsize)

                err, d_w = cuda.cuMemAlloc(w_nbytes)
                _check_cuda(err, "cuMemAlloc(d_w)")
                err, d_x = cuda.cuMemAlloc(x_nbytes)
                _check_cuda(err, "cuMemAlloc(d_x)")
                err, d_out = cuda.cuMemAlloc(out_nbytes)
                _check_cuda(err, "cuMemAlloc(d_out)")
                try:
                    t_h2d0 = time.perf_counter()
                    _check_cuda(cuda.cuMemcpyHtoD(d_w, w_pad.tobytes(), w_nbytes)[0], "cuMemcpyHtoD(d_w)")
                    _check_cuda(cuda.cuMemcpyHtoD(d_x, x_pad.tobytes(), x_nbytes)[0], "cuMemcpyHtoD(d_x)")
                    t_h2d1 = time.perf_counter()

                    in_padded_i32 = np.int32(schedule.in_padded)
                    out_i32 = np.zeros(out_elems, dtype=np.int32)
                    arg_w = ctypes.c_void_p(int(d_w))
                    arg_x = ctypes.c_void_p(int(d_x))
                    arg_out = ctypes.c_void_p(int(d_out))
                    arg_in = ctypes.c_int32(int(in_padded_i32))
                    arg_out_elems = ctypes.c_int32(int(out_elems))
                    kernel_args = (ctypes.c_void_p * 5)(
                        ctypes.cast(ctypes.pointer(arg_w), ctypes.c_void_p),
                        ctypes.cast(ctypes.pointer(arg_x), ctypes.c_void_p),
                        ctypes.cast(ctypes.pointer(arg_out), ctypes.c_void_p),
                        ctypes.cast(ctypes.pointer(arg_in), ctypes.c_void_p),
                        ctypes.cast(ctypes.pointer(arg_out_elems), ctypes.c_void_p),
                    )

                    threads = 128
                    blocks = int(math.ceil(out_elems / threads))
                    t0 = time.perf_counter()
                    err, = cuda.cuLaunchKernel(
                        fn,
                        blocks, 1, 1,
                        threads, 1, 1,
                        0,
                        0,
                        kernel_args,
                        0,
                    )
                    _check_cuda(err, "cuLaunchKernel")
                    _check_cuda(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize")
                    t1 = time.perf_counter()
                    kernel_ms = (t1 - t0) * 1000.0

                    t_d2h0 = time.perf_counter()
                    _check_cuda(cuda.cuMemcpyDtoH(out_i32.ctypes.data, d_out, out_nbytes)[0], "cuMemcpyDtoH(d_out)")
                    t_d2h1 = time.perf_counter()
                finally:
                    cuda.cuMemFree(d_w)
                    cuda.cuMemFree(d_x)
                    cuda.cuMemFree(d_out)
            finally:
                cuda.cuCtxDestroy(ctx)

            out_quant = out_i32.copy()
            if request.apply_quant:
                out_quant = _leaky_relu_quantize_int4(out_quant, apply_relu=request.apply_relu).astype(np.int8)

            ref_out = ref["output_padded"]
            max_abs_diff = int(np.max(np.abs(out_quant.astype(np.int32) - ref_out.astype(np.int32))))
            return {
                "executed": True,
                "backend": "cuda",
                "kernel_name": "blocked_fc_int4_kernel",
                "kernel_time_ms": float(kernel_ms),
                "h2d_time_ms": float((t_h2d1 - t_h2d0) * 1000.0),
                "d2h_time_ms": float((t_d2h1 - t_d2h0) * 1000.0),
                "transfer_time_ms": float(((t_h2d1 - t_h2d0) + (t_d2h1 - t_d2h0)) * 1000.0),
                "end_to_end_time_ms": float(((t_h2d1 - t_h2d0) + (t1 - t0) + (t_d2h1 - t_d2h0)) * 1000.0),
                "output_padded": out_quant.tolist(),
                "output_unpadded": out_quant[:request.out_features].tolist(),
                "numpy_reference_output": ref["output_unpadded"].tolist(),
                "max_abs_diff_vs_numpy_reference": max_abs_diff,
                "bit_exact_match_vs_numpy_reference": bool(max_abs_diff == 0),
            }
        except Exception as e:
            return {
                "executed": False,
                "backend": "cuda",
                "reason": str(e),
                "numpy_reference_output": ref["output_unpadded"].tolist(),
                "numpy_reference_accum_int32": ref["accum_int32_padded"][:request.out_features].tolist(),
            }
