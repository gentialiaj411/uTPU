import ctypes
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

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
    "use_smem": False,
}

# Typical per-block shared memory cap (bytes); larger ``in_padded`` falls back to global x.
SMEM_INPUT_BYTES_LIMIT = 48 * 1024


def normalize_cuda_schedule_params(schedule_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(DEFAULT_CUDA_SCHEDULE_PARAMS)
    if schedule_params:
        params.update(schedule_params)
    threads = int(params.get("threads_per_block", 128))
    unroll = int(params.get("unroll_factor", 1))
    use_smem = bool(params.get("use_smem", False))
    if threads <= 0 or threads > 1024:
        raise ValueError(f"threads_per_block must be in 1..1024, got {threads}")
    if unroll not in (1, 2, 4, 8):
        raise ValueError(f"unroll_factor must be one of 1, 2, 4, 8, got {unroll}")
    out: Dict[str, Any] = {
        "threads_per_block": threads,
        "unroll_factor": unroll,
    }
    if use_smem:
        out["use_smem"] = True
    return out


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


def _cuda_smem_kernel_source(schedule_params: Optional[Dict[str, Any]] = None) -> str:
    params = normalize_cuda_schedule_params(schedule_params)
    unroll = params["unroll_factor"]
    if unroll == 1:
        loop_body = """
    for (int k = 0; k < in_padded; ++k) {
        acc += (int)wrow[k] * (int)smem_x[k];
    }
"""
    else:
        terms = "\n".join(
            f"        acc += (int)wrow[k + {i}] * (int)smem_x[k + {i}];"
            for i in range(unroll)
        )
        loop_body = f"""
    int k = 0;
    for (; k + {unroll - 1} < in_padded; k += {unroll}) {{
{terms}
    }}
    for (; k < in_padded; ++k) {{
        acc += (int)wrow[k] * (int)smem_x[k];
    }}
"""
    return r"""
extern "C" __global__
void blocked_fc_int4_smem_kernel(
    const signed char* __restrict__ w,
    const signed char* __restrict__ x,
    signed int* __restrict__ accum,
    int in_padded,
    int out_elems
) {
    extern __shared__ signed char smem_x[];
    for (int i = threadIdx.x; i < in_padded; i += blockDim.x) {
        smem_x[i] = x[i];
    }
    __syncthreads();

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


def _batched_matmul_kernel_source() -> str:
    return r"""
extern "C" __global__
void batched_matmul_fp32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int k,
    int n
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
}
"""


def _scale_softmax_kernel_source() -> str:
    return r"""
extern "C" __global__
void scaled_softmax_fp32_kernel(const float* __restrict__ x, float* __restrict__ y, int n, float scale) {
    int row = blockIdx.x;
    const float* row_in = x + row * n;
    float* row_out = y + row * n;
    float max_v = -3.402823e38f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row_in[i] * scale;
        if (v > max_v) max_v = v;
    }
    __shared__ float smem_max;
    if (threadIdx.x == 0) smem_max = max_v;
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(row_in[i] * scale - smem_max);
        row_out[i] = e;
        sum += e;
    }
    __shared__ float smem_sum;
    if (threadIdx.x == 0) smem_sum = sum;
    __syncthreads();
    float denom = smem_sum + 1e-12f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] = row_out[i] / denom;
    }
}
"""


def _linear_fp32_kernel_source() -> str:
    return r"""
extern "C" __global__
void linear_fp32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int rows,
    int in_features,
    int out_features,
    int has_bias
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= out_features) return;
    float acc = 0.0f;
    const float* xrow = x + row * in_features;
    const float* wrow = w + col * in_features;
    for (int k = 0; k < in_features; ++k) {
        acc += xrow[k] * wrow[k];
    }
    if (has_bias) acc += b[col];
    y[row * out_features + col] = acc;
}
"""


def _add_fp32_kernel_source() -> str:
    return r"""
extern "C" __global__
void add_fp32_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
"""


def _scale_fp32_kernel_source() -> str:
    return r"""
extern "C" __global__
void scale_fp32_kernel(const float* x, float* y, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}
"""


def _row_softmax_fp32_kernel_source() -> str:
    return r"""
extern "C" __global__
void row_softmax_fp32_kernel(const float* x, float* y, int rows, int cols, float scale, int causal, int q_len, int k_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int q_idx = (q_len > 0) ? (row % q_len) : 0;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;
    float max_v = -3.402823e38f;
    for (int c = 0; c < cols; ++c) {
        float v = xr[c] * scale;
        if (causal && q_len > 0 && k_len > 0 && c > q_idx) v = -1.0e9f;
        if (v > max_v) max_v = v;
    }
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = xr[c] * scale;
        if (causal && q_len > 0 && k_len > 0 && c > q_idx) v = -1.0e9f;
        float e = expf(v - max_v);
        yr[c] = e;
        sum += e;
    }
    float denom = sum + 1e-12f;
    for (int c = 0; c < cols; ++c) yr[c] = yr[c] / denom;
}
"""


def _layer_norm_fp32_kernel_source() -> str:
    return r"""
extern "C" __global__
void layer_norm_fp32_kernel(
    const float* x, const float* w, const float* b, float* y,
    int rows, int cols, float eps, int use_weight, int use_bias, int rms_only
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;
    float mean = 0.0f;
    if (!rms_only) {
        for (int c = 0; c < cols; ++c) mean += xr[c];
        mean /= (float)cols;
    }
    float var = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = rms_only ? xr[c] : (xr[c] - mean);
        var += v * v;
    }
    var /= (float)cols;
    float inv = rsqrtf(var + eps);
    for (int c = 0; c < cols; ++c) {
        float v = rms_only ? xr[c] : (xr[c] - mean);
        float out = v * inv;
        if (use_weight) out *= w[c];
        if (use_bias) out += b[c];
        yr[c] = out;
    }
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

    def lower_graph_op(self, op: Any) -> Dict[str, Any]:
        op_kind = str(op.op)
        kernel_name = None
        kernel_source = None
        if op_kind == OpKind.BATCHED_MATMUL:
            kernel_name = "batched_matmul_fp32_kernel"
            kernel_source = _batched_matmul_kernel_source()
        elif op_kind in {OpKind.SCALE, OpKind.SCALED_SOFTMAX, OpKind.SOFTMAX}:
            kernel_name = "scaled_softmax_fp32_kernel"
            kernel_source = _scale_softmax_kernel_source()
        elif op_kind in {OpKind.ADD, OpKind.LAYER_NORM}:
            kernel_name = "elementwise_int4_kernel"
            kernel_source = _elementwise_kernel_source()
        elif op_kind in {OpKind.CONV2D, OpKind.MAX_POOL2D, OpKind.ADAPTIVE_AVG_POOL2D}:
            kernel_name = "torch_cuda_conv_pool"
            kernel_source = None
        return {
            "mode": "cuda_graph_op",
            "op_kind": op_kind,
            "kernel_name": kernel_name,
            "kernel_source": kernel_source,
            "generated_by_compiler": True,
            "notes": [
                "Graph-op codegen emits CUDA kernel templates per op family.",
                "Execution uses compiler-owned CUDA graph-op kernels.",
            ],
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

    def _kernel_key(
        self,
        request: BlockedFCLoweringRequest,
        schedule,
        schedule_params: Dict[str, Any],
        *,
        kernel_name: str,
    ) -> tuple:
        return (
            kernel_name,
            int(request.out_features),
            int(request.in_features),
            int(schedule.in_padded),
            int(schedule.out_padded),
            int(schedule_params["threads_per_block"]),
            int(schedule_params["unroll_factor"]),
            bool(schedule_params.get("use_smem", False)),
            "int4_i32",
            "cuda",
        )

    def _resolve_blocked_fc_kernel(
        self,
        schedule,
        schedule_params: Dict[str, Any],
    ) -> tuple[str, str, bool]:
        """Return (kernel_name, nvrtc_source_fn_key, smem_fallback)."""
        use_smem = bool(schedule_params.get("use_smem", False))
        in_padded = int(schedule.in_padded)
        smem_bytes = in_padded * int(np.dtype(np.int8).itemsize)
        if use_smem and smem_bytes <= SMEM_INPUT_BYTES_LIMIT:
            return "blocked_fc_int4_smem_kernel", "smem", False
        return "blocked_fc_int4_kernel", "naive", bool(use_smem)

    def _get_kernel(
        self,
        request: BlockedFCLoweringRequest,
        schedule,
        schedule_params: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, float, float, bool, str, bool]:
        cuda, nvrtc = self._load_cuda_bindings()
        setup_ms = self._ensure_context()
        params = normalize_cuda_schedule_params(schedule_params)
        kernel_name, variant, smem_fallback = self._resolve_blocked_fc_kernel(schedule, params)
        key = self._kernel_key(
            request, schedule, params, kernel_name=kernel_name
        )
        if key in self._kernel_cache:
            entry = self._kernel_cache[key]
            return entry["fn"], 0.0, setup_ms, True, kernel_name, smem_fallback

        if variant == "smem":
            src_text = _cuda_smem_kernel_source(params)
        else:
            src_text = _cuda_kernel_source(params)
        src = src_text.encode("utf-8")
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
        err, fn = cuda.cuModuleGetFunction(mod, kernel_name.encode("utf-8"))
        self._check_cuda(err, "cuModuleGetFunction")
        setup_ms += (time.perf_counter() - t_setup0) * 1000.0

        self._kernel_cache[key] = {"module": mod, "fn": fn, "kernel_name": kernel_name}
        return fn, compile_ms, setup_ms, False, kernel_name, smem_fallback

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
            fn, compile_ms, setup_ms, kernel_cache_hit, kernel_name, smem_fallback = (
                self._get_kernel(request, schedule, params)
            )
            shared_mem_bytes = 0
            if kernel_name == "blocked_fc_int4_smem_kernel":
                shared_mem_bytes = int(schedule.in_padded) * int(np.dtype(np.int8).itemsize)

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
                shared_mem_bytes,
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
                "kernel_name": kernel_name,
                "smem_fallback": bool(smem_fallback),
                "shared_mem_bytes": int(shared_mem_bytes),
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
        self._cuda = None
        self._nvrtc = None
        self._ctx = None
        self._kernel_cache: Dict[str, Any] = {}

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

    def _check_cuda(self, err, ctx: str):
        cuda = self._cuda
        if err != cuda.CUresult.CUDA_SUCCESS:
            name = cuda.cuGetErrorName(err)[1].decode("utf-8")
            desc = cuda.cuGetErrorString(err)[1].decode("utf-8")
            raise RuntimeError(f"{ctx} failed: {name} ({desc})")

    def _check_nvrtc(self, err, ctx: str):
        nvrtc = self._nvrtc
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"{ctx} failed: {nvrtc.nvrtcGetErrorString(err)[1].decode('utf-8')}")

    def _ensure_context(self) -> None:
        cuda, _ = self._load_cuda_bindings()
        if self._ctx is not None:
            if hasattr(cuda, "cuCtxSetCurrent"):
                self._check_cuda(cuda.cuCtxSetCurrent(self._ctx)[0], "cuCtxSetCurrent")
            return
        self._check_cuda(cuda.cuInit(0)[0], "cuInit")
        err, dev = cuda.cuDeviceGet(0)
        self._check_cuda(err, "cuDeviceGet")
        err, self._ctx = cuda.cuCtxCreate(None, 0, dev)
        self._check_cuda(err, "cuCtxCreate")

    def _get_kernel(self, key: str, source: str, fn_name: str):
        cuda, nvrtc = self._load_cuda_bindings()
        self._ensure_context()
        if key in self._kernel_cache:
            return self._kernel_cache[key]
        src = source.encode("utf-8")
        err, prog = nvrtc.nvrtcCreateProgram(src, b"graph_op.cu", 0, [], [])
        self._check_nvrtc(err, f"nvrtcCreateProgram({key})")
        opts = [b"--std=c++11"]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b""
            if log_size > 1:
                buf = bytearray(log_size)
                nvrtc.nvrtcGetProgramLog(prog, buf)
                log = bytes(buf)
            raise RuntimeError(f"nvrtcCompileProgram({key}) failed: {log.decode('utf-8', errors='replace')}")
        err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        self._check_nvrtc(err, f"nvrtcGetPTXSize({key})")
        ptx = bytearray(ptx_size)
        err, = nvrtc.nvrtcGetPTX(prog, ptx)
        self._check_nvrtc(err, f"nvrtcGetPTX({key})")
        nvrtc.nvrtcDestroyProgram(prog)
        err, mod = cuda.cuModuleLoadData(bytes(ptx))
        self._check_cuda(err, f"cuModuleLoadData({key})")
        err, fn = cuda.cuModuleGetFunction(mod, fn_name.encode("utf-8"))
        self._check_cuda(err, f"cuModuleGetFunction({key})")
        self._kernel_cache[key] = (mod, fn)
        return mod, fn

    def _launch_unary_scale(self, x: np.ndarray, scale: float) -> np.ndarray:
        cuda, _ = self._load_cuda_bindings()
        _, fn = self._get_kernel("scale_fp32", _scale_fp32_kernel_source(), "scale_fp32_kernel")
        x = np.ascontiguousarray(x.astype(np.float32, copy=False).reshape(-1))
        out = np.zeros_like(x)
        nbytes = int(x.nbytes)
        err, d_x = cuda.cuMemAlloc(nbytes); self._check_cuda(err, "cuMemAlloc(scale_in)")
        err, d_y = cuda.cuMemAlloc(nbytes); self._check_cuda(err, "cuMemAlloc(scale_out)")
        self._check_cuda(cuda.cuMemcpyHtoD(d_x, x.tobytes(), nbytes)[0], "cuMemcpyHtoD(scale_x)")
        arg_x = ctypes.c_void_p(int(d_x)); arg_y = ctypes.c_void_p(int(d_y))
        arg_n = ctypes.c_int32(int(x.size)); arg_s = ctypes.c_float(float(scale))
        args = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.pointer(arg_x), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(arg_y), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(arg_n), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(arg_s), ctypes.c_void_p),
        )
        threads = 128
        blocks = int(math.ceil(x.size / threads))
        self._check_cuda(cuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0)[0], "cuLaunchKernel(scale)")
        self._check_cuda(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize(scale)")
        self._check_cuda(cuda.cuMemcpyDtoH(out.ctypes.data, d_y, nbytes)[0], "cuMemcpyDtoH(scale)")
        cuda.cuMemFree(d_x); cuda.cuMemFree(d_y)
        return out

    def _launch_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        cuda, _ = self._load_cuda_bindings()
        _, fn = self._get_kernel("add_fp32", _add_fp32_kernel_source(), "add_fp32_kernel")
        av = np.ascontiguousarray(a.astype(np.float32, copy=False).reshape(-1))
        bv = np.ascontiguousarray(b.astype(np.float32, copy=False).reshape(-1))
        out = np.zeros_like(av)
        nbytes = int(av.nbytes)
        err, d_a = cuda.cuMemAlloc(nbytes); self._check_cuda(err, "cuMemAlloc(add_a)")
        err, d_b = cuda.cuMemAlloc(nbytes); self._check_cuda(err, "cuMemAlloc(add_b)")
        err, d_o = cuda.cuMemAlloc(nbytes); self._check_cuda(err, "cuMemAlloc(add_out)")
        self._check_cuda(cuda.cuMemcpyHtoD(d_a, av.tobytes(), nbytes)[0], "cuMemcpyHtoD(add_a)")
        self._check_cuda(cuda.cuMemcpyHtoD(d_b, bv.tobytes(), nbytes)[0], "cuMemcpyHtoD(add_b)")
        arg_a = ctypes.c_void_p(int(d_a)); arg_b = ctypes.c_void_p(int(d_b)); arg_o = ctypes.c_void_p(int(d_o)); arg_n = ctypes.c_int32(int(av.size))
        args = (ctypes.c_void_p * 4)(ctypes.cast(ctypes.pointer(arg_a), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_b), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_o), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_n), ctypes.c_void_p))
        threads = 128; blocks = int(math.ceil(av.size / threads))
        self._check_cuda(cuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0)[0], "cuLaunchKernel(add)")
        self._check_cuda(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize(add)")
        self._check_cuda(cuda.cuMemcpyDtoH(out.ctypes.data, d_o, nbytes)[0], "cuMemcpyDtoH(add)")
        cuda.cuMemFree(d_a); cuda.cuMemFree(d_b); cuda.cuMemFree(d_o)
        return out

    def _launch_linear(self, x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
        cuda, _ = self._load_cuda_bindings()
        _, fn = self._get_kernel("linear_fp32", _linear_fp32_kernel_source(), "linear_fp32_kernel")
        rows = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
        in_features = int(x.shape[-1])
        out_features = int(w.shape[0])
        xv = np.ascontiguousarray(x.astype(np.float32, copy=False).reshape(rows, in_features))
        wv = np.ascontiguousarray(w.astype(np.float32, copy=False))
        bv = None if b is None else np.ascontiguousarray(b.astype(np.float32, copy=False).reshape(-1))
        out = np.zeros((rows, out_features), dtype=np.float32)
        x_bytes = int(xv.nbytes); w_bytes = int(wv.nbytes); o_bytes = int(out.nbytes); b_bytes = int(bv.nbytes) if bv is not None else 0
        err, d_x = cuda.cuMemAlloc(x_bytes); self._check_cuda(err, "cuMemAlloc(linear_x)")
        err, d_w = cuda.cuMemAlloc(w_bytes); self._check_cuda(err, "cuMemAlloc(linear_w)")
        err, d_o = cuda.cuMemAlloc(o_bytes); self._check_cuda(err, "cuMemAlloc(linear_o)")
        d_b = 0
        if bv is not None:
            err, d_b = cuda.cuMemAlloc(b_bytes); self._check_cuda(err, "cuMemAlloc(linear_b)")
            self._check_cuda(cuda.cuMemcpyHtoD(d_b, bv.tobytes(), b_bytes)[0], "cuMemcpyHtoD(linear_b)")
        self._check_cuda(cuda.cuMemcpyHtoD(d_x, xv.tobytes(), x_bytes)[0], "cuMemcpyHtoD(linear_x)")
        self._check_cuda(cuda.cuMemcpyHtoD(d_w, wv.tobytes(), w_bytes)[0], "cuMemcpyHtoD(linear_w)")
        arg_x = ctypes.c_void_p(int(d_x)); arg_w = ctypes.c_void_p(int(d_w)); arg_b = ctypes.c_void_p(int(d_b)); arg_o = ctypes.c_void_p(int(d_o))
        arg_rows = ctypes.c_int32(rows); arg_in = ctypes.c_int32(in_features); arg_out = ctypes.c_int32(out_features); arg_has_bias = ctypes.c_int32(1 if bv is not None else 0)
        args = (ctypes.c_void_p * 8)(ctypes.cast(ctypes.pointer(arg_x), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_w), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_b), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_o), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_rows), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_in), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_out), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_has_bias), ctypes.c_void_p))
        tx, ty = 16, 16
        bx, by = int(math.ceil(out_features / tx)), int(math.ceil(rows / ty))
        self._check_cuda(cuda.cuLaunchKernel(fn, bx, by, 1, tx, ty, 1, 0, 0, args, 0)[0], "cuLaunchKernel(linear)")
        self._check_cuda(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize(linear)")
        self._check_cuda(cuda.cuMemcpyDtoH(out.ctypes.data, d_o, o_bytes)[0], "cuMemcpyDtoH(linear)")
        cuda.cuMemFree(d_x); cuda.cuMemFree(d_w); cuda.cuMemFree(d_o)
        if d_b:
            cuda.cuMemFree(d_b)
        return out.reshape(tuple(x.shape[:-1]) + (out_features,))

    def _launch_softmax(self, x: np.ndarray, scale: float, causal: bool = False) -> np.ndarray:
        cuda, _ = self._load_cuda_bindings()
        _, fn = self._get_kernel("row_softmax_fp32", _row_softmax_fp32_kernel_source(), "row_softmax_fp32_kernel")
        cols = int(x.shape[-1]); rows = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
        xv = np.ascontiguousarray(x.astype(np.float32, copy=False).reshape(rows, cols))
        out = np.zeros_like(xv)
        nbytes = int(xv.nbytes)
        err, d_x = cuda.cuMemAlloc(nbytes); self._check_cuda(err, "cuMemAlloc(softmax_x)")
        err, d_y = cuda.cuMemAlloc(nbytes); self._check_cuda(err, "cuMemAlloc(softmax_y)")
        self._check_cuda(cuda.cuMemcpyHtoD(d_x, xv.tobytes(), nbytes)[0], "cuMemcpyHtoD(softmax_x)")
        q_len = int(x.shape[-2]) if x.ndim >= 2 else 0
        k_len = int(x.shape[-1]) if x.ndim >= 1 else 0
        arg_x = ctypes.c_void_p(int(d_x)); arg_y = ctypes.c_void_p(int(d_y))
        arg_rows = ctypes.c_int32(rows); arg_cols = ctypes.c_int32(cols); arg_scale = ctypes.c_float(scale)
        arg_causal = ctypes.c_int32(1 if causal else 0); arg_q = ctypes.c_int32(q_len); arg_k = ctypes.c_int32(k_len)
        args = (ctypes.c_void_p * 8)(ctypes.cast(ctypes.pointer(arg_x), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_y), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_rows), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_cols), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_scale), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_causal), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_q), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_k), ctypes.c_void_p))
        threads = 128; blocks = int(math.ceil(rows / threads))
        self._check_cuda(cuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0)[0], "cuLaunchKernel(softmax)")
        self._check_cuda(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize(softmax)")
        self._check_cuda(cuda.cuMemcpyDtoH(out.ctypes.data, d_y, nbytes)[0], "cuMemcpyDtoH(softmax)")
        cuda.cuMemFree(d_x); cuda.cuMemFree(d_y)
        return out.reshape(x.shape)

    def _torch_cuda_available(self) -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _nvrtc_graph_active(self) -> bool:
        return bool(detect_cuda_environment().runtime_available)

    def _launch_conv2d(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: Optional[np.ndarray],
        stride: Any,
        padding: Any,
        groups: int,
    ) -> np.ndarray:
        from graph_conv_ops import conv2d_nchw_numpy

        if not self._nvrtc_graph_active() and self._torch_cuda_available():
            import torch
            import torch.nn.functional as F

            device = torch.device("cuda")
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device=device)
            wt = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(device=device)
            bt = None
            if b is not None:
                bt = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to(device=device)
            with torch.no_grad():
                yt = F.conv2d(xt, wt, bt, stride=stride, padding=padding, groups=int(groups))
            return yt.detach().cpu().numpy().astype(np.float32, copy=False)
        return conv2d_nchw_numpy(x, w, bias=b, stride=stride, padding=padding, groups=groups)

    def _launch_max_pool2d(
        self,
        x: np.ndarray,
        kernel_size: Any,
        stride: Optional[Any],
        padding: Any,
        dilation: Any,
        ceil_mode: bool,
    ) -> np.ndarray:
        from graph_conv_ops import max_pool2d_nchw_numpy

        if not self._nvrtc_graph_active() and self._torch_cuda_available():
            import torch
            import torch.nn.functional as F

            device = torch.device("cuda")
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device=device)
            with torch.no_grad():
                yt = F.max_pool2d(
                    xt,
                    kernel_size,
                    stride=kernel_size if stride is None else stride,
                    padding=padding,
                    dilation=dilation,
                    ceil_mode=bool(ceil_mode),
                )
            return yt.detach().cpu().numpy().astype(np.float32, copy=False)
        return max_pool2d_nchw_numpy(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    def _launch_adaptive_avg_pool2d(self, x: np.ndarray, output_size: Any) -> np.ndarray:
        from graph_conv_ops import adaptive_avg_pool2d_nchw_numpy

        if not self._nvrtc_graph_active() and self._torch_cuda_available():
            import torch
            import torch.nn.functional as F

            device = torch.device("cuda")
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device=device)
            with torch.no_grad():
                yt = F.adaptive_avg_pool2d(xt, output_size)
            return yt.detach().cpu().numpy().astype(np.float32, copy=False)
        return adaptive_avg_pool2d_nchw_numpy(x, output_size=output_size)

    def _launch_layer_norm(self, x: np.ndarray, op_attrs: Dict[str, Any]) -> np.ndarray:
        cuda, _ = self._load_cuda_bindings()
        _, fn = self._get_kernel("layer_norm_fp32", _layer_norm_fp32_kernel_source(), "layer_norm_fp32_kernel")
        cols = int(x.shape[-1]); rows = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
        xv = np.ascontiguousarray(x.astype(np.float32, copy=False).reshape(rows, cols))
        out = np.zeros_like(xv)
        w = op_attrs.get("weight"); b = op_attrs.get("bias")
        use_weight = 1 if w is not None else 0
        use_bias = 1 if b is not None else 0
        rms_only = 1 if str(op_attrs.get("norm_kind", "rms_norm")) != "layer_norm" else 0
        eps = float(op_attrs.get("eps", 1e-5))
        nbytes = int(xv.nbytes)
        err, d_x = cuda.cuMemAlloc(nbytes); self._check_cuda(err, "cuMemAlloc(norm_x)")
        err, d_y = cuda.cuMemAlloc(nbytes); self._check_cuda(err, "cuMemAlloc(norm_y)")
        d_w = 0; d_b = 0
        if use_weight:
            wv = np.ascontiguousarray(np.asarray(w, dtype=np.float32).reshape(-1))
            err, d_w = cuda.cuMemAlloc(int(wv.nbytes)); self._check_cuda(err, "cuMemAlloc(norm_w)")
            self._check_cuda(cuda.cuMemcpyHtoD(d_w, wv.tobytes(), int(wv.nbytes))[0], "cuMemcpyHtoD(norm_w)")
        if use_bias:
            bv = np.ascontiguousarray(np.asarray(b, dtype=np.float32).reshape(-1))
            err, d_b = cuda.cuMemAlloc(int(bv.nbytes)); self._check_cuda(err, "cuMemAlloc(norm_b)")
            self._check_cuda(cuda.cuMemcpyHtoD(d_b, bv.tobytes(), int(bv.nbytes))[0], "cuMemcpyHtoD(norm_b)")
        self._check_cuda(cuda.cuMemcpyHtoD(d_x, xv.tobytes(), nbytes)[0], "cuMemcpyHtoD(norm_x)")
        arg_x = ctypes.c_void_p(int(d_x)); arg_w = ctypes.c_void_p(int(d_w)); arg_b = ctypes.c_void_p(int(d_b)); arg_y = ctypes.c_void_p(int(d_y))
        arg_rows = ctypes.c_int32(rows); arg_cols = ctypes.c_int32(cols); arg_eps = ctypes.c_float(eps)
        arg_use_w = ctypes.c_int32(use_weight); arg_use_b = ctypes.c_int32(use_bias); arg_rms = ctypes.c_int32(rms_only)
        args = (ctypes.c_void_p * 10)(ctypes.cast(ctypes.pointer(arg_x), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_w), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_b), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_y), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_rows), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_cols), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_eps), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_use_w), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_use_b), ctypes.c_void_p), ctypes.cast(ctypes.pointer(arg_rms), ctypes.c_void_p))
        threads = 128; blocks = int(math.ceil(rows / threads))
        self._check_cuda(cuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0)[0], "cuLaunchKernel(norm)")
        self._check_cuda(cuda.cuCtxSynchronize()[0], "cuCtxSynchronize(norm)")
        self._check_cuda(cuda.cuMemcpyDtoH(out.ctypes.data, d_y, nbytes)[0], "cuMemcpyDtoH(norm)")
        cuda.cuMemFree(d_x); cuda.cuMemFree(d_y)
        if d_w: cuda.cuMemFree(d_w)
        if d_b: cuda.cuMemFree(d_b)
        return out.reshape(x.shape)

    def _run_numpy_graph(self, graph: GraphIR, *inputs: Any) -> Dict[str, Any]:
        from graph_reference_interpreter import execute_graph_reference

        outputs = execute_graph_reference(graph, *inputs)
        return {"executed": True, "outputs": outputs, "engine": "numpy_graph_reference"}

    def _run_torch_cuda_graph(self, graph: GraphIR, *inputs: Any) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        device = torch.device("cuda")
        values: Dict[str, torch.Tensor] = {}
        for name, value in zip(graph.inputs, inputs):
            if hasattr(value, "detach"):
                values[name] = value.detach().to(device=device, dtype=torch.float32)
            else:
                values[name] = torch.as_tensor(value, dtype=torch.float32, device=device)

        with torch.no_grad():
            for op in graph.ops:
                if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
                    x = values[op.inputs[0]]
                    w = torch.as_tensor(op.attrs["weight"], dtype=torch.float32, device=device)
                    b = op.attrs.get("bias")
                    b_t = None if b is None else torch.as_tensor(b, dtype=torch.float32, device=device)
                    y = F.linear(x, w, b_t)
                    if op.op == OpKind.LINEAR_RELU:
                        y = torch.relu(y)
                    values[op.outputs[0]] = y
                    continue
                if op.op == OpKind.CONV2D:
                    x = values[op.inputs[0]]
                    w = torch.as_tensor(op.attrs["weight"], dtype=torch.float32, device=device)
                    b = op.attrs.get("bias")
                    b_t = None if b is None else torch.as_tensor(b, dtype=torch.float32, device=device)
                    values[op.outputs[0]] = F.conv2d(
                        x,
                        w,
                        b_t,
                        stride=op.attrs.get("stride", 1),
                        padding=op.attrs.get("padding", 0),
                        groups=int(op.attrs.get("groups", 1)),
                    )
                    continue
                if op.op == OpKind.MAX_POOL2D:
                    x = values[op.inputs[0]]
                    values[op.outputs[0]] = F.max_pool2d(
                        x,
                        op.attrs.get("kernel_size", 1),
                        stride=op.attrs.get("stride"),
                        padding=op.attrs.get("padding", 0),
                        dilation=op.attrs.get("dilation", 1),
                        ceil_mode=bool(op.attrs.get("ceil_mode", False)),
                    )
                    continue
                if op.op == OpKind.ADAPTIVE_AVG_POOL2D:
                    x = values[op.inputs[0]]
                    values[op.outputs[0]] = F.adaptive_avg_pool2d(
                        x, op.attrs.get("output_size", 1)
                    )
                    continue
                if op.op == OpKind.ADD:
                    values[op.outputs[0]] = values[op.inputs[0]] + values[op.inputs[1]]
                    continue
                if op.op == OpKind.RELU:
                    values[op.outputs[0]] = torch.relu(values[op.inputs[0]])
                    continue
                if op.op == OpKind.VIEW:
                    x = values[op.inputs[0]]
                    raw = tuple(op.attrs.get("args", ()))
                    if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                        raw = tuple(raw[0])
                    values[op.outputs[0]] = x.reshape(raw)
                    continue
                if op.op == OpKind.PERMUTE:
                    x = values[op.inputs[0]]
                    raw = tuple(op.attrs.get("args", ()))
                    if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                        raw = tuple(raw[0])
                    values[op.outputs[0]] = x.permute(raw)
                    continue
                return {"executed": False, "reason": f"unsupported op in torch cuda graph path: {op.op}"}

        outputs = [values[name].detach().cpu().numpy().astype(np.float32, copy=False) for name in graph.outputs]
        return {"executed": True, "outputs": outputs if len(outputs) > 1 else outputs[0], "engine": "torch_cuda_graph"}

    def run(self, graph: GraphIR, *inputs: Any) -> Dict[str, Any]:
        env = detect_cuda_environment()
        if not env.runtime_available:
            if self._torch_cuda_available():
                try:
                    return self._run_torch_cuda_graph(graph, *inputs)
                except Exception as e:
                    return {"executed": False, "reason": f"torch cuda graph path failed: {e}"}
            try:
                return self._run_numpy_graph(graph, *inputs)
            except Exception as e:
                return {"executed": False, "reason": f"numpy graph path failed: {e}"}
        try:
            self._ensure_context()
            values: Dict[str, np.ndarray] = {}
            for name, value in zip(graph.inputs, inputs):
                if hasattr(value, "detach"):
                    values[name] = value.detach().cpu().numpy().astype(np.float32)
                else:
                    values[name] = np.asarray(value, dtype=np.float32)

            for op in graph.ops:
                if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}:
                    x = values[op.inputs[0]]
                    w = np.asarray(op.attrs["weight"], dtype=np.float32)
                    b = None if op.attrs.get("bias") is None else np.asarray(op.attrs["bias"], dtype=np.float32)
                    y = self._launch_linear(x, w, b)
                    if op.op == OpKind.LINEAR_RELU:
                        y = np.maximum(y, 0.0).astype(np.float32)
                    values[op.outputs[0]] = y
                    continue
                if op.op == OpKind.ADD:
                    lhs = values[op.inputs[0]]
                    rhs = values[op.inputs[1]]
                    values[op.outputs[0]] = self._launch_add(lhs, rhs).reshape(lhs.shape)
                    continue
                if op.op == OpKind.VIEW:
                    x = values[op.inputs[0]]
                    raw = tuple(op.attrs.get("args", ()))
                    if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                        raw = tuple(raw[0])
                    values[op.outputs[0]] = x.reshape(raw).astype(np.float32, copy=False)
                    continue
                if op.op == OpKind.PERMUTE:
                    x = values[op.inputs[0]]
                    raw = tuple(op.attrs.get("args", ()))
                    if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
                        raw = tuple(raw[0])
                    values[op.outputs[0]] = np.transpose(x, axes=raw).astype(np.float32, copy=False)
                    continue
                if op.op == OpKind.BATCHED_MATMUL:
                    a = values[op.inputs[0]]
                    b = values[op.inputs[1]]
                    batch = int(np.prod(a.shape[:-2])) if a.ndim > 2 else 1
                    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
                    a2 = a.reshape(batch, m, k)
                    b2 = b.reshape(batch, k, n)
                    out = np.zeros((batch, m, n), dtype=np.float32)
                    for i in range(batch):
                        out[i] = self._launch_linear(a2[i], b2[i].T, None)
                    values[op.outputs[0]] = out.reshape(tuple(a.shape[:-2]) + (m, n))
                    continue
                if op.op == OpKind.SCALE:
                    x = values[op.inputs[0]]
                    s = float(op.attrs.get("scale", 1.0))
                    values[op.outputs[0]] = self._launch_unary_scale(x, s).reshape(x.shape)
                    continue
                if op.op == OpKind.SOFTMAX:
                    x = values[op.inputs[0]]
                    values[op.outputs[0]] = self._launch_softmax(x, 1.0, causal=False)
                    continue
                if op.op == OpKind.SCALED_SOFTMAX:
                    x = values[op.inputs[0]]
                    s = float(op.attrs.get("scale", 1.0))
                    causal = bool(op.attrs.get("causal_mask", False))
                    values[op.outputs[0]] = self._launch_softmax(x, s, causal=causal)
                    continue
                if op.op == OpKind.LAYER_NORM:
                    x = values[op.inputs[0]]
                    values[op.outputs[0]] = self._launch_layer_norm(x, op.attrs)
                    continue
                if op.op == OpKind.RELU:
                    x = values[op.inputs[0]]
                    values[op.outputs[0]] = np.maximum(x, 0.0).astype(np.float32)
                    continue
                if op.op == OpKind.CONV2D:
                    x = values[op.inputs[0]]
                    w = np.asarray(op.attrs["weight"], dtype=np.float32)
                    b = None if op.attrs.get("bias") is None else np.asarray(op.attrs["bias"], dtype=np.float32)
                    values[op.outputs[0]] = self._launch_conv2d(
                        x,
                        w,
                        b,
                        stride=op.attrs.get("stride", 1),
                        padding=op.attrs.get("padding", 0),
                        groups=int(op.attrs.get("groups", 1)),
                    )
                    continue
                if op.op == OpKind.MAX_POOL2D:
                    x = values[op.inputs[0]]
                    values[op.outputs[0]] = self._launch_max_pool2d(
                        x,
                        kernel_size=op.attrs.get("kernel_size", 1),
                        stride=op.attrs.get("stride"),
                        padding=op.attrs.get("padding", 0),
                        dilation=op.attrs.get("dilation", 1),
                        ceil_mode=bool(op.attrs.get("ceil_mode", False)),
                    )
                    continue
                if op.op == OpKind.ADAPTIVE_AVG_POOL2D:
                    x = values[op.inputs[0]]
                    values[op.outputs[0]] = self._launch_adaptive_avg_pool2d(
                        x, output_size=op.attrs.get("output_size", 1)
                    )
                    continue
                return {"executed": False, "reason": f"unsupported op: {op.op}"}

            outputs = [values[name].astype(np.float32, copy=False) for name in graph.outputs]
            return {"executed": True, "outputs": outputs if len(outputs) > 1 else outputs[0]}
        except Exception as e:
            return {"executed": False, "reason": str(e)}
