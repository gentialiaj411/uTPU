import ctypes
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from graph_ir import GraphIR, OpKind
from lowering_types import BlockedFCLoweringRequest
from compiler_abstractions import (
    BlockedFCProblem,
    build_blocked_fc_schedule,
    cuda_target_desc,
)


DEFAULT_CUDA_SCHEDULE_PARAMS = {
    "threads_per_block": 128,
    "unroll_factor": 1,
}


def normalize_cuda_schedule_params(schedule_params: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    params = dict(DEFAULT_CUDA_SCHEDULE_PARAMS)
    if schedule_params:
        params.update(schedule_params)
    threads = int(params.get("threads_per_block", 128))
    unroll = int(params.get("unroll_factor", 1))
    if threads <= 0 or threads > 1024:
        raise ValueError(f"threads_per_block must be in 1..1024, got {threads}")
    if unroll not in (1, 2, 4, 8):
        raise ValueError(f"unroll_factor must be one of 1, 2, 4, 8, got {unroll}")
    return {
        "threads_per_block": threads,
        "unroll_factor": unroll,
    }


def _cuda_kernel_source(schedule_params: Optional[Dict[str, Any]] = None) -> str:
    params = normalize_cuda_schedule_params(schedule_params)
    unroll = params["unroll_factor"]
    if unroll == 1:
        loop_body = """
    for (int k = 0; k < in_padded; ++k) {
        acc += (int)wrow[k] * (int)x[k];
    }
"""
    else:
        terms = "\n".join(
            f"        acc += (int)wrow[k + {i}] * (int)x[k + {i}];"
            for i in range(unroll)
        )
        loop_body = f"""
    int k = 0;
    for (; k + {unroll - 1} < in_padded; k += {unroll}) {{
{terms}
    }}
    for (; k < in_padded; ++k) {{
        acc += (int)wrow[k] * (int)x[k];
    }}
"""
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
""" + loop_body + r"""
    accum[row] = acc;
}
"""


def _cuda_quantized_kernel_source(schedule_params: Optional[Dict[str, Any]] = None) -> str:
    params = normalize_cuda_schedule_params(schedule_params)
    unroll = params["unroll_factor"]
    if unroll == 1:
        loop_body = """
    for (int k = 0; k < in_padded; ++k) {
        acc += (int)wrow[k] * (int)x[k];
    }
"""
    else:
        terms = "\n".join(
            f"        acc += (int)wrow[k + {i}] * (int)x[k + {i}];"
            for i in range(unroll)
        )
        loop_body = f"""
    int k = 0;
    for (; k + {unroll - 1} < in_padded; k += {unroll}) {{
{terms}
    }}
    for (; k < in_padded; ++k) {{
        acc += (int)wrow[k] * (int)x[k];
    }}
"""
    return r"""
extern "C" __global__
void blocked_fc_int4_quant_kernel(
    const signed char* __restrict__ w,
    const signed char* __restrict__ x,
    signed char* __restrict__ y,
    int in_padded,
    int out_elems
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_elems) return;
    int acc = 0;
    const signed char* wrow = w + row * in_padded;
""" + loop_body + r"""
    if (acc > 7) acc = 7;
    if (acc < -8) acc = -8;
    y[row] = (signed char)acc;
}
"""


def _elementwise_kernel_source() -> str:
    return r"""
extern "C" __global__
void elementwise_int4_kernel(
    const signed char* __restrict__ x,
    const signed char* __restrict__ bias,
    signed char* __restrict__ y,
    int n,
    int has_bias,
    int apply_relu
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = (int)x[i];
    if (has_bias) {
        v += (int)bias[i];
    }
    if (apply_relu && v < 0) {
        v = 0;
    }
    if (v > 7) v = 7;
    if (v < -8) v = -8;
    y[i] = (signed char)v;
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
        self._cuda = None
        self._nvrtc = None
        self._ctx = None
        self._kernel_cache = {}
        self._buffer_cache = {}
        self._constant_upload_cache = set()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CUDABlockedFCExecutor] {msg}")

    def cache_stats(self) -> Dict[str, Any]:
        return {
            "context_initialized": self._ctx is not None,
            "kernel_cache_entries": len(self._kernel_cache),
            "buffer_cache_entries": len(self._buffer_cache),
        }

    def _load_cuda_bindings(self):
        if self._cuda is not None and self._nvrtc is not None:
            return self._cuda, self._nvrtc
        try:
            from cuda import cuda, nvrtc
        except Exception:
            from cuda.bindings import driver as cuda
            from cuda.bindings import nvrtc
        self._cuda = cuda
        self._nvrtc = nvrtc
        return cuda, nvrtc

    def _check_nvrtc(self, err, ctx: str):
        nvrtc = self._nvrtc
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"{ctx} failed: {nvrtc.nvrtcGetErrorString(err)[1].decode('utf-8')}")

    def _check_cuda(self, err, ctx: str):
        cuda = self._cuda
        if err != cuda.CUresult.CUDA_SUCCESS:
            name = cuda.cuGetErrorName(err)[1].decode("utf-8")
            desc = cuda.cuGetErrorString(err)[1].decode("utf-8")
            raise RuntimeError(f"{ctx} failed: {name} ({desc})")

    def _ensure_context(self) -> float:
        cuda, _ = self._load_cuda_bindings()
        if self._ctx is not None:
            if hasattr(cuda, "cuCtxSetCurrent"):
                self._check_cuda(cuda.cuCtxSetCurrent(self._ctx)[0], "cuCtxSetCurrent")
            return 0.0
        t0 = time.perf_counter()
        err, = cuda.cuInit(0)
        self._check_cuda(err, "cuInit")
        err, dev = cuda.cuDeviceGet(0)
        self._check_cuda(err, "cuDeviceGet")
        err, self._ctx = cuda.cuCtxCreate(None, 0, dev)
        self._check_cuda(err, "cuCtxCreate")
        return (time.perf_counter() - t0) * 1000.0

    def _kernel_key(self, request: BlockedFCLoweringRequest, schedule, schedule_params: Dict[str, int]) -> tuple:
        return (
            "blocked_fc_int4_kernel",
            int(request.out_features),
            int(request.in_features),
            int(schedule.in_padded),
            int(schedule.out_padded),
            int(schedule_params["threads_per_block"]),
            int(schedule_params["unroll_factor"]),
            "int4_i32",
            "cuda",
        )

    def _get_kernel(
        self,
        request: BlockedFCLoweringRequest,
        schedule,
        schedule_params: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, float, float, bool]:
        cuda, nvrtc = self._load_cuda_bindings()
        setup_ms = self._ensure_context()
        params = normalize_cuda_schedule_params(schedule_params)
        key = self._kernel_key(request, schedule, params)
        if key in self._kernel_cache:
            return self._kernel_cache[key]["fn"], 0.0, setup_ms, True

        src = _cuda_kernel_source(params).encode("utf-8")
        t_compile0 = time.perf_counter()
        err, prog = nvrtc.nvrtcCreateProgram(src, b"blocked_fc.cu", 0, [], [])
        self._check_nvrtc(err, "nvrtcCreateProgram")
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
        self._check_nvrtc(err, "nvrtcGetPTXSize")
        ptx = bytearray(ptx_size)
        err, = nvrtc.nvrtcGetPTX(prog, ptx)
        self._check_nvrtc(err, "nvrtcGetPTX")
        nvrtc.nvrtcDestroyProgram(prog)
        compile_ms = (time.perf_counter() - t_compile0) * 1000.0

        t_setup0 = time.perf_counter()
        err, mod = cuda.cuModuleLoadData(bytes(ptx))
        self._check_cuda(err, "cuModuleLoadData")
        err, fn = cuda.cuModuleGetFunction(mod, b"blocked_fc_int4_kernel")
        self._check_cuda(err, "cuModuleGetFunction")
        setup_ms += (time.perf_counter() - t_setup0) * 1000.0

        self._kernel_cache[key] = {"module": mod, "fn": fn}
        return fn, compile_ms, setup_ms, False

    def _get_elementwise_kernel(self) -> tuple[Any, float, float, bool]:
        cuda, nvrtc = self._load_cuda_bindings()
        setup_ms = self._ensure_context()
        key = ("elementwise_int4_kernel", "int4", "bias_relu", "cuda")
        if key in self._kernel_cache:
            return self._kernel_cache[key]["fn"], 0.0, setup_ms, True

        src = _elementwise_kernel_source().encode("utf-8")
        t_compile0 = time.perf_counter()
        err, prog = nvrtc.nvrtcCreateProgram(src, b"elementwise_int4.cu", 0, [], [])
        self._check_nvrtc(err, "nvrtcCreateProgram(elementwise)")
        opts = [b"--std=c++11"]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b""
            if log_size > 1:
                log_buf = bytearray(log_size)
                nvrtc.nvrtcGetProgramLog(prog, log_buf)
                log = bytes(log_buf)
            raise RuntimeError(f"nvrtcCompileProgram(elementwise) failed: {log.decode('utf-8', errors='replace')}")
        err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        self._check_nvrtc(err, "nvrtcGetPTXSize(elementwise)")
        ptx = bytearray(ptx_size)
        err, = nvrtc.nvrtcGetPTX(prog, ptx)
        self._check_nvrtc(err, "nvrtcGetPTX(elementwise)")
        nvrtc.nvrtcDestroyProgram(prog)
        compile_ms = (time.perf_counter() - t_compile0) * 1000.0

        t_setup0 = time.perf_counter()
        err, mod = cuda.cuModuleLoadData(bytes(ptx))
        self._check_cuda(err, "cuModuleLoadData(elementwise)")
        err, fn = cuda.cuModuleGetFunction(mod, b"elementwise_int4_kernel")
        self._check_cuda(err, "cuModuleGetFunction(elementwise)")
        setup_ms += (time.perf_counter() - t_setup0) * 1000.0

        self._kernel_cache[key] = {"module": mod, "fn": fn}
        return fn, compile_ms, setup_ms, False

    def _get_quantized_kernel(
        self,
        request: BlockedFCLoweringRequest,
        schedule,
        schedule_params: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, float, float, bool]:
        cuda, nvrtc = self._load_cuda_bindings()
        setup_ms = self._ensure_context()
        params = normalize_cuda_schedule_params(schedule_params)
        key = (
            "blocked_fc_int4_quant_kernel",
            int(request.out_features),
            int(request.in_features),
            int(schedule.in_padded),
            int(schedule.out_padded),
            int(params["threads_per_block"]),
            int(params["unroll_factor"]),
            "int4_i8",
            "cuda",
        )
        if key in self._kernel_cache:
            return self._kernel_cache[key]["fn"], 0.0, setup_ms, True

        src = _cuda_quantized_kernel_source(params).encode("utf-8")
        t_compile0 = time.perf_counter()
        err, prog = nvrtc.nvrtcCreateProgram(src, b"blocked_fc_quant.cu", 0, [], [])
        self._check_nvrtc(err, "nvrtcCreateProgram(quant)")
        opts = [b"--std=c++11"]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b""
            if log_size > 1:
                log_buf = bytearray(log_size)
                nvrtc.nvrtcGetProgramLog(prog, log_buf)
                log = bytes(log_buf)
            raise RuntimeError(f"nvrtcCompileProgram(quant) failed: {log.decode('utf-8', errors='replace')}")
        err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        self._check_nvrtc(err, "nvrtcGetPTXSize(quant)")
        ptx = bytearray(ptx_size)
        err, = nvrtc.nvrtcGetPTX(prog, ptx)
        self._check_nvrtc(err, "nvrtcGetPTX(quant)")
        nvrtc.nvrtcDestroyProgram(prog)
        compile_ms = (time.perf_counter() - t_compile0) * 1000.0

        t_setup0 = time.perf_counter()
        err, mod = cuda.cuModuleLoadData(bytes(ptx))
        self._check_cuda(err, "cuModuleLoadData(quant)")
        err, fn = cuda.cuModuleGetFunction(mod, b"blocked_fc_int4_quant_kernel")
        self._check_cuda(err, "cuModuleGetFunction(quant)")
        setup_ms += (time.perf_counter() - t_setup0) * 1000.0

        self._kernel_cache[key] = {"module": mod, "fn": fn}
        return fn, compile_ms, setup_ms, False

    def _get_buffer(self, name: str, nbytes: int) -> tuple[Any, float, bool]:
        cuda, _ = self._load_cuda_bindings()
        existing = self._buffer_cache.get(name)
        if existing is not None and existing["nbytes"] >= nbytes:
            return existing["ptr"], 0.0, True
        if existing is not None:
            cuda.cuMemFree(existing["ptr"])
        t0 = time.perf_counter()
        err, ptr = cuda.cuMemAlloc(nbytes)
        self._check_cuda(err, f"cuMemAlloc({name})")
        alloc_ms = (time.perf_counter() - t0) * 1000.0
        self._buffer_cache[name] = {"ptr": ptr, "nbytes": nbytes}
        return ptr, alloc_ms, False

    def _upload_constant_once(self, name: str, data: np.ndarray) -> tuple[Any, float, bool]:
        cuda, _ = self._load_cuda_bindings()
        arr = np.ascontiguousarray(data)
        ptr, setup_ms, reused = self._get_buffer(name, int(arr.nbytes))
        if name in self._constant_upload_cache:
            return ptr, setup_ms, True
        t0 = time.perf_counter()
        self._check_cuda(cuda.cuMemcpyHtoD(ptr, arr.tobytes(), int(arr.nbytes))[0], f"cuMemcpyHtoD({name})")
        upload_ms = (time.perf_counter() - t0) * 1000.0
        self._constant_upload_cache.add(name)
        return ptr, setup_ms + upload_ms, reused

    def execute(
        self,
        request: BlockedFCLoweringRequest,
        schedule_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
            cuda, _ = self._load_cuda_bindings()

            schedule = ref["schedule"]
            params = normalize_cuda_schedule_params(schedule_params)
            w_pad = ref["weights_padded"]
            x_pad = ref["inputs_padded"]
            out_elems = schedule.out_padded
            fn, compile_ms, setup_ms, kernel_cache_hit = self._get_kernel(request, schedule, params)

            w_nbytes = int(w_pad.size)
            x_nbytes = int(x_pad.size)
            out_nbytes = int(out_elems * np.dtype(np.int32).itemsize)

            d_w, alloc_w_ms, w_buffer_reused = self._get_buffer("weights", w_nbytes)
            d_x, alloc_x_ms, x_buffer_reused = self._get_buffer("inputs", x_nbytes)
            d_out, alloc_out_ms, out_buffer_reused = self._get_buffer("outputs", out_nbytes)
            setup_ms += alloc_w_ms + alloc_x_ms + alloc_out_ms

            t_h2d0 = time.perf_counter()
            self._check_cuda(cuda.cuMemcpyHtoD(d_w, w_pad.tobytes(), w_nbytes)[0], "cuMemcpyHtoD(d_w)")
            self._check_cuda(cuda.cuMemcpyHtoD(d_x, x_pad.tobytes(), x_nbytes)[0], "cuMemcpyHtoD(d_x)")
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

            threads = int(params["threads_per_block"])
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
            self._check_cuda(err, "cuLaunchKernel")
            self._check_cuda(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize")
            t1 = time.perf_counter()
            kernel_ms = (t1 - t0) * 1000.0

            t_d2h0 = time.perf_counter()
            self._check_cuda(cuda.cuMemcpyDtoH(out_i32.ctypes.data, d_out, out_nbytes)[0], "cuMemcpyDtoH(d_out)")
            t_d2h1 = time.perf_counter()

            out_quant = out_i32.copy()
            if request.apply_quant:
                out_quant = _leaky_relu_quantize_int4(out_quant, apply_relu=request.apply_relu).astype(np.int8)

            ref_out = ref["output_padded"]
            max_abs_diff = int(np.max(np.abs(out_quant.astype(np.int32) - ref_out.astype(np.int32))))
            return {
                "executed": True,
                "backend": "cuda",
                "kernel_name": "blocked_fc_int4_kernel",
                "schedule_params": dict(params),
                "compile_time_ms": float(compile_ms),
                "setup_time_ms": float(setup_ms),
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
                "kernel_cache_hit": bool(kernel_cache_hit),
                "buffer_reuse": {
                    "weights": bool(w_buffer_reused),
                    "inputs": bool(x_buffer_reused),
                    "outputs": bool(out_buffer_reused),
                },
                "cache_stats": self.cache_stats(),
            }
        except Exception as e:
            return {
                "executed": False,
                "backend": "cuda",
                "reason": str(e),
                "numpy_reference_output": ref["output_unpadded"].tolist(),
                "numpy_reference_accum_int32": ref["accum_int32_padded"][:request.out_features].tolist(),
            }

    def execute_graph_resident_int4(
        self,
        ops: list[Dict[str, Any]],
        input_int4,
        array_size: int = 16,
    ) -> Dict[str, Any]:
        env = detect_cuda_environment()
        if not env.runtime_available:
            return {
                "executed": False,
                "backend": "cuda",
                "reason": env.reason,
            }
        if not ops:
            raise ValueError("execute_graph_resident_int4 requires at least one op")

        try:
            cuda, _ = self._load_cuda_bindings()
            setup_ms = self._ensure_context()
            input_arr = np.asarray(input_int4, dtype=np.int8).reshape(-1)
            compile_ms = 0.0
            h2d_ms = 0.0
            d2h_ms = 0.0
            kernel_ms = 0.0
            h2d_count = 0
            d2h_count = 0
            linear_count = 0
            elementwise_count = 0
            op_results = []

            first_w = np.asarray(ops[0]["weights_int4"], dtype=np.int8)
            first_problem = BlockedFCProblem(
                out_features=int(first_w.shape[0]),
                in_features=int(first_w.shape[1]),
                array_size=int(array_size),
            )
            first_schedule = build_blocked_fc_schedule(
                problem=first_problem,
                target=cuda_target_desc(array_size=int(array_size)),
            )
            input_padded = np.zeros(first_schedule.in_padded, dtype=np.int8)
            input_padded[: input_arr.size] = input_arr[: first_problem.in_features]
            d_input, alloc_ms, _ = self._get_buffer("graph_input", int(input_padded.nbytes))
            setup_ms += alloc_ms
            t_h2d0 = time.perf_counter()
            self._check_cuda(cuda.cuMemcpyHtoD(d_input, input_padded.tobytes(), int(input_padded.nbytes))[0], "cuMemcpyHtoD(graph_input)")
            h2d_ms += (time.perf_counter() - t_h2d0) * 1000.0
            h2d_count += 1
            d_x = d_input

            last_out_features = None
            for idx, op in enumerate(ops):
                name = str(op.get("name", f"op{idx}"))
                w = np.asarray(op["weights_int4"], dtype=np.int8)
                out_features = int(w.shape[0])
                in_features = int(w.shape[1])
                request = BlockedFCLoweringRequest(
                    weights_int4=w,
                    activations_int4=np.zeros(in_features, dtype=np.int8),
                    out_features=out_features,
                    in_features=in_features,
                    array_size=int(array_size),
                    apply_relu=False,
                    apply_quant=True,
                    weight_addr=0,
                    input_addr=0,
                    result_addr=0,
                )
                ref = _numpy_blocked_fc_reference(
                    w,
                    np.zeros(in_features, dtype=np.int8),
                    out_features,
                    in_features,
                    int(array_size),
                    apply_relu=False,
                    apply_quant=True,
                )
                schedule = ref["schedule"]
                params = normalize_cuda_schedule_params(op.get("schedule_params"))
                fn, c_ms, s_ms, kernel_cache_hit = self._get_quantized_kernel(request, schedule, params)
                compile_ms += c_ms
                setup_ms += s_ms

                d_w, s_ms, _ = self._upload_constant_once(f"graph_{name}_weights", ref["weights_padded"])
                setup_ms += s_ms
                out_nbytes = int(schedule.out_padded)
                d_y, alloc_ms, output_reused = self._get_buffer(f"graph_{name}_output", out_nbytes)
                setup_ms += alloc_ms

                arg_w = ctypes.c_void_p(int(d_w))
                arg_x = ctypes.c_void_p(int(d_x))
                arg_y = ctypes.c_void_p(int(d_y))
                arg_in = ctypes.c_int32(int(schedule.in_padded))
                arg_out_elems = ctypes.c_int32(int(schedule.out_padded))
                kernel_args = (ctypes.c_void_p * 5)(
                    ctypes.cast(ctypes.pointer(arg_w), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(arg_x), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(arg_y), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(arg_in), ctypes.c_void_p),
                    ctypes.cast(ctypes.pointer(arg_out_elems), ctypes.c_void_p),
                )

                threads = int(params["threads_per_block"])
                blocks = int(math.ceil(schedule.out_padded / threads))
                t0 = time.perf_counter()
                err, = cuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, kernel_args, 0)
                self._check_cuda(err, f"cuLaunchKernel({name})")
                self._check_cuda(cuda.cuCtxSynchronize()[0], f"cuCtxSynchronize({name})")
                linear_kernel_ms = (time.perf_counter() - t0) * 1000.0
                kernel_ms += linear_kernel_ms
                linear_count += 1
                op_results.append(
                    {
                        "name": name,
                        "op": "linear",
                        "kernel_time_ms": float(linear_kernel_ms),
                        "compile_time_ms": float(c_ms),
                        "setup_time_ms": float(s_ms + alloc_ms),
                        "kernel_cache_hit": bool(kernel_cache_hit),
                        "schedule_params": dict(params),
                        "output_buffer_reused": bool(output_reused),
                    }
                )

                bias = op.get("bias_int4")
                apply_relu = bool(op.get("apply_relu", False))
                if bias is not None or apply_relu:
                    elem_fn, c_ms, s_ms, elem_cache_hit = self._get_elementwise_kernel()
                    compile_ms += c_ms
                    setup_ms += s_ms
                    bias_ptr = 0
                    if bias is not None:
                        bias_pad = np.zeros(schedule.out_padded, dtype=np.int8)
                        bias_arr = np.asarray(bias, dtype=np.int8).reshape(-1)
                        bias_pad[: bias_arr.size] = bias_arr
                        d_bias, s_ms, _ = self._upload_constant_once(f"graph_{name}_bias", bias_pad)
                        setup_ms += s_ms
                        bias_ptr = int(d_bias)

                    arg_x2 = ctypes.c_void_p(int(d_y))
                    arg_bias = ctypes.c_void_p(bias_ptr)
                    arg_y2 = ctypes.c_void_p(int(d_y))
                    arg_n = ctypes.c_int32(int(schedule.out_padded))
                    arg_has_bias = ctypes.c_int32(1 if bias is not None else 0)
                    arg_relu = ctypes.c_int32(1 if apply_relu else 0)
                    elem_args = (ctypes.c_void_p * 6)(
                        ctypes.cast(ctypes.pointer(arg_x2), ctypes.c_void_p),
                        ctypes.cast(ctypes.pointer(arg_bias), ctypes.c_void_p),
                        ctypes.cast(ctypes.pointer(arg_y2), ctypes.c_void_p),
                        ctypes.cast(ctypes.pointer(arg_n), ctypes.c_void_p),
                        ctypes.cast(ctypes.pointer(arg_has_bias), ctypes.c_void_p),
                        ctypes.cast(ctypes.pointer(arg_relu), ctypes.c_void_p),
                    )
                    elem_threads = 128
                    elem_blocks = int(math.ceil(schedule.out_padded / elem_threads))
                    t0 = time.perf_counter()
                    err, = cuda.cuLaunchKernel(elem_fn, elem_blocks, 1, 1, elem_threads, 1, 1, 0, 0, elem_args, 0)
                    self._check_cuda(err, f"cuLaunchKernel({name}.elementwise)")
                    self._check_cuda(cuda.cuCtxSynchronize()[0], f"cuCtxSynchronize({name}.elementwise)")
                    elem_kernel_ms = (time.perf_counter() - t0) * 1000.0
                    kernel_ms += elem_kernel_ms
                    elementwise_count += 1
                    op_results.append(
                        {
                            "name": f"{name}.elementwise",
                            "op": "bias_relu",
                            "kernel_time_ms": float(elem_kernel_ms),
                            "compile_time_ms": float(c_ms),
                            "setup_time_ms": float(s_ms),
                            "kernel_cache_hit": bool(elem_cache_hit),
                        }
                    )

                d_x = d_y
                last_out_features = out_features

            if last_out_features is None:
                raise RuntimeError("No output produced by resident graph execution")
            out = np.zeros(last_out_features, dtype=np.int8)
            t_d2h0 = time.perf_counter()
            self._check_cuda(cuda.cuMemcpyDtoH(out.ctypes.data, d_x, int(out.nbytes))[0], "cuMemcpyDtoH(graph_output)")
            d2h_ms += (time.perf_counter() - t_d2h0) * 1000.0
            d2h_count += 1

            return {
                "executed": True,
                "backend": "cuda",
                "mode": "graph_resident_int4",
                "output_unpadded": out.tolist(),
                "compile_time_ms": float(compile_ms),
                "setup_time_ms": float(setup_ms),
                "h2d_time_ms": float(h2d_ms),
                "d2h_time_ms": float(d2h_ms),
                "kernel_time_ms": float(kernel_ms),
                "transfer_time_ms": float(h2d_ms + d2h_ms),
                "end_to_end_time_ms": float(h2d_ms + kernel_ms + d2h_ms),
                "h2d_count": int(h2d_count),
                "d2h_count": int(d2h_count),
                "backend_linear_ops_executed": int(linear_count),
                "backend_elementwise_ops_executed": int(elementwise_count),
                "op_results": op_results,
                "cache_stats": self.cache_stats(),
            }
        except Exception as e:
            return {
                "executed": False,
                "backend": "cuda",
                "mode": "graph_resident_int4",
                "reason": str(e),
            }

    def execute_elementwise_int4(
        self,
        values_int4,
        bias_int4=None,
        apply_relu: bool = False,
    ) -> Dict[str, Any]:
        env = detect_cuda_environment()
        x = np.asarray(values_int4, dtype=np.int8).reshape(-1)
        bias = None if bias_int4 is None else np.asarray(bias_int4, dtype=np.int8).reshape(-1)
        if bias is not None and bias.shape[0] != x.shape[0]:
            raise ValueError(f"bias length mismatch: expected {x.shape[0]}, got {bias.shape[0]}")
        if not env.runtime_available:
            return {
                "executed": False,
                "backend": "cuda",
                "reason": env.reason,
            }
        try:
            cuda, _ = self._load_cuda_bindings()
            fn, compile_ms, setup_ms, kernel_cache_hit = self._get_elementwise_kernel()

            nbytes = int(x.size)
            d_x, alloc_x_ms, x_buffer_reused = self._get_buffer("elem_input", nbytes)
            d_y, alloc_y_ms, y_buffer_reused = self._get_buffer("elem_output", nbytes)
            d_bias = None
            bias_buffer_reused = True
            if bias is not None:
                d_bias, alloc_bias_ms, bias_buffer_reused = self._get_buffer("elem_bias", nbytes)
                setup_ms += alloc_bias_ms
            setup_ms += alloc_x_ms + alloc_y_ms

            t_h2d0 = time.perf_counter()
            self._check_cuda(cuda.cuMemcpyHtoD(d_x, x.tobytes(), nbytes)[0], "cuMemcpyHtoD(elem_x)")
            if bias is not None:
                self._check_cuda(cuda.cuMemcpyHtoD(d_bias, bias.tobytes(), nbytes)[0], "cuMemcpyHtoD(elem_bias)")
            t_h2d1 = time.perf_counter()

            out = np.zeros(x.size, dtype=np.int8)
            arg_x = ctypes.c_void_p(int(d_x))
            arg_bias = ctypes.c_void_p(int(d_bias) if d_bias is not None else 0)
            arg_y = ctypes.c_void_p(int(d_y))
            arg_n = ctypes.c_int32(int(x.size))
            arg_has_bias = ctypes.c_int32(1 if bias is not None else 0)
            arg_relu = ctypes.c_int32(1 if apply_relu else 0)
            kernel_args = (ctypes.c_void_p * 6)(
                ctypes.cast(ctypes.pointer(arg_x), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_bias), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_y), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_n), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_has_bias), ctypes.c_void_p),
                ctypes.cast(ctypes.pointer(arg_relu), ctypes.c_void_p),
            )

            threads = 128
            blocks = int(math.ceil(x.size / threads))
            t0 = time.perf_counter()
            err, = cuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, kernel_args, 0)
            self._check_cuda(err, "cuLaunchKernel(elementwise)")
            self._check_cuda(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize(elementwise)")
            t1 = time.perf_counter()

            t_d2h0 = time.perf_counter()
            self._check_cuda(cuda.cuMemcpyDtoH(out.ctypes.data, d_y, nbytes)[0], "cuMemcpyDtoH(elem_y)")
            t_d2h1 = time.perf_counter()

            return {
                "executed": True,
                "backend": "cuda",
                "kernel_name": "elementwise_int4_kernel",
                "compile_time_ms": float(compile_ms),
                "setup_time_ms": float(setup_ms),
                "kernel_time_ms": float((t1 - t0) * 1000.0),
                "h2d_time_ms": float((t_h2d1 - t_h2d0) * 1000.0),
                "d2h_time_ms": float((t_d2h1 - t_d2h0) * 1000.0),
                "output": out.tolist(),
                "kernel_cache_hit": bool(kernel_cache_hit),
                "buffer_reuse": {
                    "input": bool(x_buffer_reused),
                    "bias": bool(bias_buffer_reused),
                    "output": bool(y_buffer_reused),
                },
                "cache_stats": self.cache_stats(),
            }
        except Exception as e:
            return {
                "executed": False,
                "backend": "cuda",
                "reason": str(e),
            }


class CUDAGraphOpExecutor:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def _as_torch(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device, dtype=torch.float32)
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def run(self, graph: GraphIR, *inputs: Any) -> Dict[str, Any]:
        if self.device == "cuda" and not torch.cuda.is_available():
            return {"executed": False, "reason": "torch.cuda.is_available() is False"}

        values: Dict[str, torch.Tensor] = {}
        for name, value in zip(graph.inputs, inputs):
            values[name] = self._as_torch(value)

        for op in graph.ops:
            if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
                x = self._as_torch(values[op.inputs[0]])
                if op.attrs.get("dtype_quant") == "int4_g64":
                    packed = np.asarray(op.attrs["weight_int4_packed"], dtype=np.uint8).reshape(-1)
                    total = int(np.prod(np.asarray(op.attrs["weight_int4_shape"])))
                    q = np.zeros((packed.size * 2,), dtype=np.int8)
                    q[0::2] = (packed & 0x0F).astype(np.int8) - 8
                    q[1::2] = ((packed >> 4) & 0x0F).astype(np.int8) - 8
                    q = q[:total].reshape(tuple(op.attrs["weight_int4_shape"])).astype(np.float32)
                    scales = np.asarray(op.attrs["weight_int4_scales"], dtype=np.float32)
                    deq = np.zeros_like(q, dtype=np.float32)
                    group_size = 64
                    for o in range(deq.shape[0]):
                        for g in range(scales.shape[1]):
                            s = g * group_size
                            e = min(deq.shape[1], s + group_size)
                            deq[o, s:e] = q[o, s:e] * scales[o, g]
                    w = self._as_torch(deq)
                else:
                    w = self._as_torch(op.attrs["weight"])
                b = op.attrs.get("bias")
                y = x @ w.transpose(-1, -2)
                if b is not None:
                    y = y + self._as_torch(b)
                if op.op == OpKind.LINEAR_RELU:
                    y = torch.relu(y)
                values[op.outputs[0]] = y
                continue

            if op.op == OpKind.RELU:
                values[op.outputs[0]] = torch.relu(self._as_torch(values[op.inputs[0]]))
                continue

            if op.op == OpKind.ADD:
                lhs = self._as_torch(values[op.inputs[0]])
                rhs = self._as_torch(values[op.inputs[1]])
                values[op.outputs[0]] = lhs + rhs
                continue

            if op.op == OpKind.VIEW:
                x = self._as_torch(values[op.inputs[0]])
                raw = tuple(op.attrs.get("args", ()))
                if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                    raw = tuple(raw[0])
                values[op.outputs[0]] = x.reshape(raw)
                continue
            if op.op == OpKind.PERMUTE:
                x = self._as_torch(values[op.inputs[0]])
                raw = tuple(op.attrs.get("args", ()))
                if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                    raw = tuple(raw[0])
                values[op.outputs[0]] = x.permute(*raw)
                continue

            if op.op == OpKind.SOFTMAX:
                x = self._as_torch(values[op.inputs[0]])
                values[op.outputs[0]] = torch.softmax(x, dim=-1)
                continue

            if op.op == OpKind.LAYER_NORM:
                x = self._as_torch(values[op.inputs[0]])
                eps = float(op.attrs.get("eps", 1e-5))
                norm_kind = str(op.attrs.get("norm_kind", "rms_norm"))
                if norm_kind == "layer_norm":
                    mean = torch.mean(x, dim=-1, keepdim=True)
                    var = torch.mean((x - mean) * (x - mean), dim=-1, keepdim=True)
                    y = (x - mean) / torch.sqrt(var + eps)
                else:
                    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
                    y = x / rms
                if op.attrs.get("weight") is not None:
                    w = self._as_torch(op.attrs["weight"])
                    y = y * w
                if op.attrs.get("bias") is not None:
                    b = self._as_torch(op.attrs["bias"])
                    y = y + b
                values[op.outputs[0]] = y
                continue

            if op.op == OpKind.SCALED_DOT_PRODUCT_ATTENTION:
                q = self._as_torch(values[op.inputs[0]])
                k = self._as_torch(values[op.inputs[1]])
                v = self._as_torch(values[op.inputs[2]])
                attn_mask = None
                if len(op.inputs) > 3 and op.inputs[3] in values:
                    attn_mask = self._as_torch(values[op.inputs[3]])
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    is_causal=bool(op.attrs.get("causal_mask", False)),
                )
                values[op.outputs[0]] = out
                continue

            return {"executed": False, "reason": f"unsupported op: {op.op}"}

        outputs = [values[name].detach().cpu().numpy().astype(np.float32) for name in graph.outputs]
        return {"executed": True, "outputs": outputs if len(outputs) > 1 else outputs[0]}
