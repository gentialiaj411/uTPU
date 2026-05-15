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
