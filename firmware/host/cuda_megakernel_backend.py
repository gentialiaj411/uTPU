"""Fused CUDA region-kernel backend (Task 1 of `utpu_upgrade_plan.md`).

**Naming note.** The plan file calls this "megakernel" for plan continuity,
but per the v1 scope correction this module is honestly a **fused CUDA region
kernel** generator, not a persistent grid-wide-synchronizing kernel. It emits
ONE NVRTC kernel per legal region (`linear_with_epilogue` or
`elementwise_chain`) produced by `region_fusion.find_fusion_regions`. The
critical safety invariant is inherited from `region_fusion`: regions that
would require grid-wide synchronization (multi-Linear chains across CTAs)
are rejected upstream, so this codegen never has to reason about it.

The module is structured into three layers:

1. **Pure codegen** — `generate_kernel_source(region, graph)`. Returns a
   well-formed NVRTC source string. No CUDA dependency; testable on Windows.
2. **Execution** — `execute_region_cuda(...)`. Only runs when `cuda-python`
   is importable; otherwise raises `diff_oracle.BackendUnavailable` so the
   harness records a clean skip-with-reason.
3. **Registration** — `register_with_diff_oracle()`. Replaces the
   `cuda_megakernel` skip stub with the real runner so Task 1's benchmark
   can compare against `numpy_reference` through one harness.

The runtime path is OPT-IN. Existing op-by-op execution remains the
default; nothing wired in this module changes any existing artifact.

Honesty contract:
- No claim of beating cuBLAS.
- No claim of a persistent multi-layer megakernel.
- The kernel emits per-output-element work; the epilogue ops fold into the
  final write, so the per-thread accumulator is fully computed before the
  ReLU / ADD residual / SCALE epilogue runs.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
import diff_oracle
from diff_oracle import BackendUnavailable
from region_fusion import RegionPlan, find_fusion_regions


DEFAULT_THREADS_PER_BLOCK = 128


# ---------------------------------------------------------------------------
# Pure codegen. Returns NVRTC-ready source. Tested on the CPU host.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratedKernel:
    """A region's compiled-form description prior to NVRTC compilation.

    `source` is the C++ kernel text (a single `extern "C" __global__` entry
    point named `kernel_name`). `kernel_name` is stable across runs for a
    region with the same shape/ops, which lets a cache key over
    (kernel_name, threads_per_block, region.region_id) identify a compiled
    module.

    `external_input_layout` describes — in order — the float32 device
    buffers the kernel reads (besides weights/biases which are bound as
    constant inputs). `weights_packed` and `biases_packed` are the
    pre-flattened host-side weight / bias arrays for the root linear op
    (None for elementwise-chain regions).
    """

    region_id: str
    region_kind: str
    kernel_name: str
    source: str
    threads_per_block: int
    output_dim: int
    external_input_layout: List[Dict[str, Any]] = field(default_factory=list)
    weights_packed: Optional[np.ndarray] = None
    biases_packed: Optional[np.ndarray] = None


def _kernel_name_for(region: RegionPlan) -> str:
    # NVRTC requires a valid C identifier; region_id is already safe (letters,
    # digits, underscores), but we add a stable prefix to make it obvious in
    # `cuobjdump` output that this came from the region codegen.
    return f"utpu_fused_region__{region.region_id}"


def _resolve_op_by_name(graph: GraphIR, name: str) -> OpNode:
    for op in graph.ops:
        if op.name == name:
            return op
    raise KeyError(f"op '{name}' not found in graph '{graph.name}'")


def _generate_linear_with_epilogue_kernel(
    region: RegionPlan,
    graph: GraphIR,
    threads_per_block: int,
) -> GeneratedKernel:
    """Emit one CUDA kernel for `linear_with_epilogue`.

    Layout (one thread = one output row):
      - Each thread computes `acc = sum_k W[row, k] * x[k] (+ bias[row])`.
      - The epilogue (ReLU / ADD residual / SCALE) runs in the same thread,
        consuming externally-produced residual buffers passed by pointer.
      - Final value is written to `y[row]`.
    The per-thread accumulator is finished before the epilogue runs, so no
    cross-thread sync is required at any point.
    """
    if region.region_kind != "linear_with_epilogue":
        raise ValueError(f"_generate_linear_with_epilogue_kernel called for {region.region_kind}")
    if region.root_op_name is None:
        raise ValueError(f"linear_with_epilogue region {region.region_id} missing root_op_name")

    root = _resolve_op_by_name(graph, region.root_op_name)
    w = np.asarray(root.attrs["weight"], dtype=np.float32)
    out_features = int(w.shape[0])
    in_features = int(w.shape[1])
    bias = root.attrs.get("bias")
    bias_arr = np.asarray(bias, dtype=np.float32) if bias is not None else None
    chain_includes_root_relu = root.op == OpKind.LINEAR_RELU

    epilogue_ops = [_resolve_op_by_name(graph, name) for name in region.epilogue_op_names]

    # External input layout begins with the activation `x`. For each ADD in
    # the epilogue we add the residual external input. SCALE constants are
    # baked into the kernel source. RELU has no extra input.
    layout: List[Dict[str, Any]] = [
        {"role": "activation", "value_name": root.inputs[0], "elements": in_features, "dtype": "float32"},
    ]

    epilogue_stmts: List[str] = []
    if chain_includes_root_relu:
        epilogue_stmts.append("    if (acc < 0.0f) acc = 0.0f;  // fused linear_relu root")
    for eop in epilogue_ops:
        if eop.op == OpKind.RELU:
            epilogue_stmts.append(f"    if (acc < 0.0f) acc = 0.0f;  // {eop.name}")
            continue
        if eop.op == OpKind.SCALE:
            s = float(eop.attrs.get("scale", 1.0))
            epilogue_stmts.append(f"    acc = acc * {s:.9g}f;  // {eop.name}")
            continue
        if eop.op == OpKind.ADD:
            chain_outputs = {root.outputs[0], *(o.outputs[0] for o in epilogue_ops if o.name != eop.name)}
            external_inputs = [inp for inp in eop.inputs if inp not in chain_outputs]
            if len(external_inputs) != 1:
                raise ValueError(
                    f"region {region.region_id} ADD '{eop.name}' has {len(external_inputs)} "
                    "external inputs; expected exactly 1 (planner should have rejected this)"
                )
            ext = external_inputs[0]
            ptr_idx = len(layout)
            layout.append({"role": "residual", "value_name": ext, "elements": out_features, "dtype": "float32"})
            epilogue_stmts.append(
                f"    acc = acc + ext_{ptr_idx}[row];  // {eop.name} residual"
            )
            continue
        raise ValueError(f"region {region.region_id} contains unsupported epilogue op '{eop.op}'")

    # Build the C++ signature dynamically over the external pointers.
    sig_external = ", ".join(
        f"const float* __restrict__ ext_{i}" for i in range(len(layout))
    )
    bias_decl = ", const float* __restrict__ bias" if bias_arr is not None else ""
    bias_term = " + bias[row]" if bias_arr is not None else ""

    src = f"""
extern "C" __global__
void {_kernel_name_for(region)}(
    const float* __restrict__ weights,  // [out_features, in_features], row-major
    {bias_decl.lstrip(', ')}{', ' if bias_decl else ''}
    {sig_external},
    float* __restrict__ y,
    int in_features,
    int out_features
) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_features) return;
    float acc = 0.0f;
    const float* wrow = weights + row * in_features;
    // Activation pointer is ext_0 by layout convention.
    const float* x = ext_0;
    for (int k = 0; k < in_features; ++k) {{
        acc += wrow[k] * x[k];
    }}
    acc = acc{bias_term};
{chr(10).join(epilogue_stmts)}
    y[row] = acc;
}}
""".lstrip()

    return GeneratedKernel(
        region_id=region.region_id,
        region_kind=region.region_kind,
        kernel_name=_kernel_name_for(region),
        source=src,
        threads_per_block=int(threads_per_block),
        output_dim=out_features,
        external_input_layout=layout,
        weights_packed=np.ascontiguousarray(w, dtype=np.float32),
        biases_packed=np.ascontiguousarray(bias_arr, dtype=np.float32) if bias_arr is not None else None,
    )


def _generate_elementwise_chain_kernel(
    region: RegionPlan,
    graph: GraphIR,
    threads_per_block: int,
) -> GeneratedKernel:
    """Emit one CUDA kernel for `elementwise_chain` (RELU/ADD/SCALE chain)."""
    if region.region_kind != "elementwise_chain":
        raise ValueError(f"_generate_elementwise_chain_kernel called for {region.region_kind}")

    chain_ops = [_resolve_op_by_name(graph, name) for name in region.op_names]

    # External inputs in order: primary input (first op's input[0]), then any
    # ADD residual operands.
    primary_input_name = chain_ops[0].inputs[0]
    layout: List[Dict[str, Any]] = [
        {"role": "primary_input", "value_name": primary_input_name, "elements": -1, "dtype": "float32"},
    ]
    stmts: List[str] = ["    float v = ext_0[idx];"]
    chain_outputs_seen = set()
    for op in chain_ops:
        if op.op == OpKind.RELU:
            stmts.append(f"    if (v < 0.0f) v = 0.0f;  // {op.name}")
            chain_outputs_seen.add(op.outputs[0])
            continue
        if op.op == OpKind.SCALE:
            s = float(op.attrs.get("scale", 1.0))
            stmts.append(f"    v = v * {s:.9g}f;  // {op.name}")
            chain_outputs_seen.add(op.outputs[0])
            continue
        if op.op == OpKind.ADD:
            external_inputs = [inp for inp in op.inputs if inp not in chain_outputs_seen and inp != primary_input_name]
            if len(external_inputs) != 1:
                raise ValueError(
                    f"region {region.region_id} ADD '{op.name}' has {len(external_inputs)} external inputs (expected 1)"
                )
            ext = external_inputs[0]
            ptr_idx = len(layout)
            layout.append({"role": "residual", "value_name": ext, "elements": -1, "dtype": "float32"})
            stmts.append(f"    v = v + ext_{ptr_idx}[idx];  // {op.name}")
            chain_outputs_seen.add(op.outputs[0])
            continue
        raise ValueError(f"region {region.region_id} contains unsupported op '{op.op}' in elementwise chain")

    sig_external = ", ".join(
        f"const float* __restrict__ ext_{i}" for i in range(len(layout))
    )
    src = f"""
extern "C" __global__
void {_kernel_name_for(region)}(
    {sig_external},
    float* __restrict__ y,
    int n_elements
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
{chr(10).join(stmts)}
    y[idx] = v;
}}
""".lstrip()

    return GeneratedKernel(
        region_id=region.region_id,
        region_kind=region.region_kind,
        kernel_name=_kernel_name_for(region),
        source=src,
        threads_per_block=int(threads_per_block),
        output_dim=-1,  # determined per-input at run time
        external_input_layout=layout,
        weights_packed=None,
        biases_packed=None,
    )


def generate_kernel_source(
    region: RegionPlan,
    graph: GraphIR,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
) -> GeneratedKernel:
    """Emit the NVRTC source string for a single region. Pure codegen."""
    if threads_per_block <= 0 or threads_per_block > 1024:
        raise ValueError(f"threads_per_block must be in 1..1024, got {threads_per_block}")
    if region.region_kind == "linear_with_epilogue":
        return _generate_linear_with_epilogue_kernel(region, graph, threads_per_block)
    if region.region_kind == "elementwise_chain":
        return _generate_elementwise_chain_kernel(region, graph, threads_per_block)
    raise ValueError(
        f"v1 fused-region codegen supports 'linear_with_epilogue' / 'elementwise_chain', "
        f"got region_kind='{region.region_kind}' (single_cta_bounded_multilayer is future work)"
    )


# ---------------------------------------------------------------------------
# Per-op kernel codegen (used by the op_by_op benchmark arm). One NVRTC kernel
# per op so the fused-vs-op-by-op comparison is apples-to-apples (same kernel
# implementation quality on both sides; the only thing that varies is the
# launch count).
# ---------------------------------------------------------------------------

def _per_op_kernel_name(op_name: str) -> str:
    return f"utpu_per_op__{op_name}"


def generate_per_op_kernel_source(op: OpNode, threads_per_block: int = DEFAULT_THREADS_PER_BLOCK) -> str:
    """Emit a standalone NVRTC kernel source for one op. Pure codegen.

    The op_by_op benchmark arm uses this to launch one kernel per op so the
    measured "op_by_op time" is N kernels' worth of launches + compute over
    the same code we use inside the fused region (modulo per-launch
    boilerplate). v1 supports the same op set as fused regions:
    LINEAR/LINEAR_RELU/RELU/ADD/SCALE.
    """
    if threads_per_block <= 0 or threads_per_block > 1024:
        raise ValueError(f"threads_per_block must be in 1..1024, got {threads_per_block}")
    name = _per_op_kernel_name(op.name)
    if op.op in (OpKind.LINEAR, OpKind.LINEAR_RELU):
        bias = op.attrs.get("bias")
        bias_decl = ", const float* __restrict__ bias" if bias is not None else ""
        bias_term = " + bias[row]" if bias is not None else ""
        relu_stmt = "    if (acc < 0.0f) acc = 0.0f;" if op.op == OpKind.LINEAR_RELU else ""
        return f"""
extern "C" __global__
void {name}(
    const float* __restrict__ w,
    const float* __restrict__ x{bias_decl},
    float* __restrict__ y,
    int in_features,
    int out_features
) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_features) return;
    float acc = 0.0f;
    const float* wrow = w + row * in_features;
    for (int k = 0; k < in_features; ++k) {{
        acc += wrow[k] * x[k];
    }}
    acc = acc{bias_term};
{relu_stmt}
    y[row] = acc;
}}
""".lstrip()
    if op.op == OpKind.RELU:
        return f"""
extern "C" __global__
void {name}(const float* __restrict__ x, float* __restrict__ y, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    if (v < 0.0f) v = 0.0f;
    y[idx] = v;
}}
""".lstrip()
    if op.op == OpKind.SCALE:
        s = float(op.attrs.get("scale", 1.0))
        return f"""
extern "C" __global__
void {name}(const float* __restrict__ x, float* __restrict__ y, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = x[idx] * {s:.9g}f;
}}
""".lstrip()
    if op.op == OpKind.ADD:
        return f"""
extern "C" __global__
void {name}(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = a[idx] + b[idx];
}}
""".lstrip()
    raise ValueError(f"generate_per_op_kernel_source: op '{op.op}' not in v1 per-op set")


# ---------------------------------------------------------------------------
# CUDA execution. Only reachable when cuda-python is importable AND a CUDA
# device is available. On Windows / WSL2-without-CUDA / CI the path raises
# `BackendUnavailable`, which `diff_oracle.run_all_backends` converts to a
# clean `status="skipped"`.
# ---------------------------------------------------------------------------

def _import_cuda_python():
    """Try the cuda-python 13.x layout first, fall back to the legacy 12.x layout.

    - cuda-python >= 13.x ships `cuda.bindings.driver` + `cuda.bindings.nvrtc`
      (the `cuda-bindings` package after the cuda-python split).
    - cuda-python <= 12.x ships `cuda.cuda` + `cuda.nvrtc`.

    Both expose the same symbol names (`cuInit`, `cuMemAlloc`, `cuLaunchKernel`,
    `nvrtcCompileProgram`, …) so the rest of the backend is layout-agnostic.
    """
    last_err: Optional[Exception] = None
    try:
        from cuda.bindings import driver as cuda_mod, nvrtc as nvrtc_mod  # type: ignore[import-not-found]
        return cuda_mod, nvrtc_mod
    except Exception as e:  # noqa: BLE001
        last_err = e
    try:
        from cuda import cuda as cuda_mod, nvrtc as nvrtc_mod  # type: ignore[import-not-found]
        return cuda_mod, nvrtc_mod
    except Exception as e:  # noqa: BLE001
        raise BackendUnavailable(
            "cuda-python not importable under either layout: "
            f"`cuda.bindings.{{driver,nvrtc}}` (13.x+) failed with {last_err!r}; "
            f"`cuda.{{cuda,nvrtc}}` (12.x) failed with {e!r}. "
            "Install: `pip install cuda-python` (12.x) or `pip install cuda-bindings` (13.x+)."
        )


def _pack_kernel_args(typed_args: Sequence[Tuple[int, Any]]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Pack kernel arguments for ``cuLaunchKernel`` using cuda-python's
    documented ``(values, types)`` tuple form.

    cuda-python 13.x's ``kernelParams`` argument accepts a tuple
    ``((v0, v1, …), (t0, t1, …))`` where ``t_i`` is a ctypes type used to
    marshal ``v_i`` (``None`` means "this is already a typed CUDA pointer
    object such as ``CUdeviceptr``").

    This shape was confirmed via ``_megakernel_cuda_diagnose.py`` to be one of
    only two accepted shapes on cuda-python 13.x; the other (less portable)
    being a ``(ctypes.c_void_p * N)`` array. We pick the tuple form because
    cuda-python handles internal buffer-lifetime management itself — no need
    for the caller to retain ctypes-allocated scalars across the launch.

    Pointer arguments are passed as raw ``int`` values with type
    ``ctypes.c_void_p`` so callers don't need to hold ``CUdeviceptr`` objects
    (most of our code already stores device addresses as Python ints).

    Returns a 2-tuple ``(values, types)`` that should be passed verbatim as
    the ``kernelParams`` argument to ``cuLaunchKernel``.

    Example:
        args = _pack_kernel_args([
            (int(d_w), ctypes.c_void_p),
            (int(d_x), ctypes.c_void_p),
            (int(d_y), ctypes.c_void_p),
            (int(in_features),  ctypes.c_int),
            (int(out_features), ctypes.c_int),
        ])
        cuda_mod.cuLaunchKernel(kernel, …, args, 0)
    """
    values = tuple(v for v, _ in typed_args)
    types = tuple(t for _, t in typed_args)
    return (values, types)


def _cuda_check(result: Any, what: str) -> Tuple[Any, ...]:
    """Normalize cuda-python 12.x vs 13.x return-tuple shapes.

    cuda-python 12.x: every driver / nvrtc call returns a tuple ``(err, *vals)``;
      pure-effect calls (e.g. ``cuInit``, ``cuMemcpyHtoD``, ``cuLaunchKernel``)
      return a 1-tuple ``(err,)``.
    cuda-python 13.x: pure-effect calls return just ``err`` (a ``CUresult``);
      value-returning calls still return ``(err, *vals)``.

    This helper accepts either shape, checks the error, raises
    ``BackendUnavailable`` on failure, and returns the value-tuple
    (possibly empty). Use as:

        _cuda_check(cuda_mod.cuInit(0), "cuInit")
        (device_count,) = _cuda_check(cuda_mod.cuDeviceGetCount(), "cuDeviceGetCount")
    """
    if isinstance(result, tuple):
        err = result[0]
        vals: Tuple[Any, ...] = tuple(result[1:])
    else:
        err = result
        vals = ()
    err_int = int(err)
    if err_int != 0:
        raise BackendUnavailable(f"{what} failed: err={err_int}")
    return vals


def _ensure_cuda_context(cuda_mod) -> None:
    _cuda_check(cuda_mod.cuInit(0), "cuInit")
    (device_count,) = _cuda_check(cuda_mod.cuDeviceGetCount(), "cuDeviceGetCount")
    if device_count <= 0:
        raise BackendUnavailable(f"no CUDA devices visible: count={device_count}")


def execute_region_cuda(
    region: RegionPlan,
    graph: GraphIR,
    external_values: Dict[str, np.ndarray],
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
) -> np.ndarray:
    """Run one fused region on CUDA. NumPy-output. Raises `BackendUnavailable`
    when CUDA is not available — the diff_oracle wrapper converts that into
    `status="skipped"` automatically."""
    cuda, nvrtc = _import_cuda_python()
    _ensure_cuda_context(cuda)

    gen = generate_kernel_source(region, graph, threads_per_block=threads_per_block)

    # The full NVRTC compile + module-load + launch path is large enough that
    # we keep it in a dedicated function to make the surface easy to test
    # via mocking. v1 implements it inline below; on a real CUDA host we
    # exercise it via the subprocess benchmark.
    # NOTE: This path is INTENTIONALLY simple. It does one alloc per call —
    # the benchmark harness wraps it for warmup/timed iterations using its
    # own buffer cache so the launch isn't dominated by H2D/D2H churn.
    #
    # We use the *primary* context (cuDevicePrimaryCtxRetain), not cuCtxCreate,
    # for two reasons:
    #   1) cuda-python 13.x changed cuCtxCreate's signature to v3
    #      `(CUctxCreateParams *params, unsigned flags, CUdevice dev)`, which
    #      breaks the old 12.x 2-arg form. Primary-context retain is stable
    #      across both layouts.
    #   2) torch and most other CUDA consumers also live on the primary
    #      context. Creating a fresh context here would conflict with torch's
    #      context if anything later in the subprocess does GPU work.
    (device,) = _cuda_check(cuda.cuDeviceGet(0), "cuDeviceGet")
    (primary_ctx,) = _cuda_check(cuda.cuDevicePrimaryCtxRetain(device), "cuDevicePrimaryCtxRetain")
    _cuda_check(cuda.cuCtxSetCurrent(primary_ctx), "cuCtxSetCurrent")
    try:
        return _compile_and_launch(cuda, nvrtc, gen, region, graph, external_values, threads_per_block)
    finally:
        # Primary context is refcounted; release just decrements. Safe even if
        # torch also holds a refcount.
        try:
            cuda.cuDevicePrimaryCtxRelease(device)
        except Exception:
            pass


def _compile_and_launch(
    cuda_mod,
    nvrtc_mod,
    gen: "GeneratedKernel",
    region: RegionPlan,
    graph: GraphIR,
    external_values: Dict[str, np.ndarray],
    threads_per_block: int,
) -> np.ndarray:
    """Compile `gen.source` with NVRTC, allocate buffers, launch, copy back."""
    # NVRTC compile.
    (prog,) = _cuda_check(
        nvrtc_mod.nvrtcCreateProgram(
            gen.source.encode("utf-8"),
            f"{gen.kernel_name}.cu".encode("utf-8"),
            0,
            [],
            [],
        ),
        "nvrtcCreateProgram",
    )
    opts = [b"--gpu-architecture=compute_75", b"--use_fast_math"]
    try:
        _cuda_check(nvrtc_mod.nvrtcCompileProgram(prog, len(opts), opts), "nvrtcCompileProgram")
    except BackendUnavailable:
        # Capture the NVRTC log for diagnostics.
        try:
            (log_size,) = _cuda_check(nvrtc_mod.nvrtcGetProgramLogSize(prog), "nvrtcGetProgramLogSize")
            log_buf = b" " * int(log_size)
            nvrtc_mod.nvrtcGetProgramLog(prog, log_buf)
            log_text = log_buf.decode("utf-8", errors="replace")
        except Exception:
            log_text = "<could not retrieve NVRTC log>"
        raise BackendUnavailable(f"nvrtcCompileProgram failed; log: {log_text}")
    (ptx_size,) = _cuda_check(nvrtc_mod.nvrtcGetPTXSize(prog), "nvrtcGetPTXSize")
    ptx = b" " * int(ptx_size)
    nvrtc_mod.nvrtcGetPTX(prog, ptx)
    nvrtc_mod.nvrtcDestroyProgram(prog)

    (module,) = _cuda_check(cuda_mod.cuModuleLoadData(ptx), "cuModuleLoadData")
    (kernel,) = _cuda_check(
        cuda_mod.cuModuleGetFunction(module, gen.kernel_name.encode("utf-8")),
        "cuModuleGetFunction",
    )

    # Allocate / upload buffers.
    if region.region_kind == "linear_with_epilogue":
        out_elems = gen.output_dim
        in_elems = int(gen.weights_packed.shape[1])
        weights_host = gen.weights_packed.astype(np.float32, copy=False).reshape(-1)
        bias_host = (
            gen.biases_packed.astype(np.float32, copy=False).reshape(-1)
            if gen.biases_packed is not None
            else None
        )
        x_host = np.asarray(external_values[gen.external_input_layout[0]["value_name"]], dtype=np.float32).reshape(-1)
        residual_hosts: List[np.ndarray] = []
        for entry in gen.external_input_layout[1:]:
            buf = np.asarray(external_values[entry["value_name"]], dtype=np.float32).reshape(-1)
            if buf.size != out_elems:
                raise BackendUnavailable(
                    f"residual buffer '{entry['value_name']}' has {buf.size} elements, expected {out_elems}"
                )
            residual_hosts.append(buf)
        y_host = np.zeros(out_elems, dtype=np.float32)

        d_buffers: List[Any] = []

        def _alloc_and_copy(arr: np.ndarray):
            (ptr,) = _cuda_check(cuda_mod.cuMemAlloc(arr.nbytes), "cuMemAlloc")
            _cuda_check(cuda_mod.cuMemcpyHtoD(ptr, arr.ctypes.data, arr.nbytes), "cuMemcpyHtoD")
            d_buffers.append(ptr)
            return ptr

        try:
            d_w = _alloc_and_copy(weights_host)
            d_bias = _alloc_and_copy(bias_host) if bias_host is not None else None
            d_x = _alloc_and_copy(x_host)
            d_residuals = [_alloc_and_copy(r) for r in residual_hosts]
            (d_y,) = _cuda_check(cuda_mod.cuMemAlloc(y_host.nbytes), "cuMemAlloc(y)")
            d_buffers.append(d_y)

            # Build the kernel argument list. cuda-python 13.x's `kernelParams`
            # accepts the (values, types) tuple form (see _pack_kernel_args).
            typed: List[Tuple[int, Any]] = []
            typed.append((int(d_w), ctypes.c_void_p))
            if d_bias is not None:
                typed.append((int(d_bias), ctypes.c_void_p))
            typed.append((int(d_x), ctypes.c_void_p))
            for d_r in d_residuals:
                typed.append((int(d_r), ctypes.c_void_p))
            typed.append((int(d_y), ctypes.c_void_p))
            typed.append((int(in_elems), ctypes.c_int))
            typed.append((int(out_elems), ctypes.c_int))
            args = _pack_kernel_args(typed)

            blocks = (out_elems + threads_per_block - 1) // threads_per_block
            _cuda_check(
                cuda_mod.cuLaunchKernel(
                    kernel,
                    blocks, 1, 1,
                    threads_per_block, 1, 1,
                    0, None,
                    args, 0,
                ),
                "cuLaunchKernel",
            )
            _cuda_check(cuda_mod.cuCtxSynchronize(), "cuCtxSynchronize")
            _cuda_check(
                cuda_mod.cuMemcpyDtoH(y_host.ctypes.data, d_y, y_host.nbytes),
                "cuMemcpyDtoH",
            )
        finally:
            for ptr in d_buffers:
                try:
                    cuda_mod.cuMemFree(ptr)
                except Exception:
                    pass
            try:
                cuda_mod.cuModuleUnload(module)
            except Exception:
                pass

        # Region root may produce shape (batch, out_features). For v1 we run
        # batch=1, so reshape accordingly.
        x_full = np.asarray(external_values[gen.external_input_layout[0]["value_name"]], dtype=np.float32)
        batch_shape = tuple(x_full.shape[:-1]) if x_full.ndim >= 1 else ()
        return y_host.reshape(*batch_shape, out_elems)

    if region.region_kind == "elementwise_chain":
        primary = np.asarray(external_values[gen.external_input_layout[0]["value_name"]], dtype=np.float32)
        n_elements = int(primary.size)
        residual_hosts = []
        for entry in gen.external_input_layout[1:]:
            buf = np.asarray(external_values[entry["value_name"]], dtype=np.float32).reshape(-1)
            if buf.size != n_elements:
                raise BackendUnavailable(
                    f"residual buffer '{entry['value_name']}' has {buf.size} elements, expected {n_elements}"
                )
            residual_hosts.append(buf)
        y_host = np.zeros(n_elements, dtype=np.float32)

        d_buffers: List[Any] = []

        def _alloc_and_copy(arr: np.ndarray):
            (ptr,) = _cuda_check(cuda_mod.cuMemAlloc(arr.nbytes), "cuMemAlloc")
            _cuda_check(cuda_mod.cuMemcpyHtoD(ptr, arr.ctypes.data, arr.nbytes), "cuMemcpyHtoD")
            d_buffers.append(ptr)
            return ptr

        try:
            d_primary = _alloc_and_copy(primary.reshape(-1))
            d_residuals = [_alloc_and_copy(r) for r in residual_hosts]
            (d_y,) = _cuda_check(cuda_mod.cuMemAlloc(y_host.nbytes), "cuMemAlloc(y)")
            d_buffers.append(d_y)

            typed: List[Tuple[int, Any]] = [(int(d_primary), ctypes.c_void_p)]
            for d_r in d_residuals:
                typed.append((int(d_r), ctypes.c_void_p))
            typed.append((int(d_y), ctypes.c_void_p))
            typed.append((int(n_elements), ctypes.c_int))
            args = _pack_kernel_args(typed)
            blocks = (n_elements + threads_per_block - 1) // threads_per_block
            _cuda_check(
                cuda_mod.cuLaunchKernel(
                    kernel,
                    blocks, 1, 1,
                    threads_per_block, 1, 1,
                    0, None,
                    args, 0,
                ),
                "cuLaunchKernel",
            )
            _cuda_check(cuda_mod.cuCtxSynchronize(), "cuCtxSynchronize")
            _cuda_check(
                cuda_mod.cuMemcpyDtoH(y_host.ctypes.data, d_y, y_host.nbytes),
                "cuMemcpyDtoH",
            )
        finally:
            for ptr in d_buffers:
                try:
                    cuda_mod.cuMemFree(ptr)
                except Exception:
                    pass
            try:
                cuda_mod.cuModuleUnload(module)
            except Exception:
                pass
        return y_host.reshape(primary.shape)

    raise BackendUnavailable(f"execute_region_cuda: unsupported region_kind '{region.region_kind}'")


# ---------------------------------------------------------------------------
# diff_oracle integration. After `register_with_diff_oracle()` is called once
# per process, `run_all_backends(..., backends=("cuda_megakernel",))` will
# dispatch to the fused-region path.
# ---------------------------------------------------------------------------

def _diff_oracle_cuda_megakernel_runner(graph: GraphIR, inputs: Sequence[Any], **ctx: Any) -> np.ndarray:
    """diff_oracle.run_all_backends entry point for the cuda_megakernel backend.

    Expects:
    - `graph` is a Graph IR containing exactly one fusable region (v1
      scope: single linear_with_epilogue OR single elementwise_chain).
      Multi-region graphs are NOT v1; they raise BackendUnavailable.
    - `inputs` matches `graph.inputs` order; each is convertible to ndarray.
    """
    analysis = find_fusion_regions(graph)
    if not analysis.regions:
        raise BackendUnavailable(
            f"graph '{graph.name}' has no fusable region (rejections: "
            f"{[r.rejection_kind for r in analysis.rejections]})"
        )
    if len(analysis.regions) != 1:
        raise BackendUnavailable(
            f"graph '{graph.name}' has {len(analysis.regions)} fusable regions; "
            "v1 diff_oracle runner handles exactly one region at a time"
        )
    region = analysis.regions[0]
    # Build the external-value dict from the supplied `inputs` (positional,
    # in graph.inputs order) plus any non-graph-input values that happen to
    # already be intermediates — for v1 graphs the region's externals are
    # always a subset of graph.inputs.
    if len(inputs) != len(graph.inputs):
        raise BackendUnavailable(
            f"expected {len(graph.inputs)} positional inputs, got {len(inputs)}"
        )
    name_to_value = {name: np.asarray(val, dtype=np.float32) for name, val in zip(graph.inputs, inputs)}
    missing = [n for n in region.inputs_external if n not in name_to_value]
    if missing:
        raise BackendUnavailable(
            f"cuda_megakernel v1 requires all region externals to be graph inputs; missing {missing}"
        )
    external = {n: name_to_value[n] for n in region.inputs_external}
    threads = int(ctx.get("threads_per_block", DEFAULT_THREADS_PER_BLOCK))
    return execute_region_cuda(region, graph, external, threads_per_block=threads)


def register_with_diff_oracle() -> None:
    """Replace `diff_oracle`'s `cuda_megakernel` skip stub with the real runner.

    Idempotent; safe to call multiple times in one process.
    """
    diff_oracle.register_backend("cuda_megakernel", _diff_oracle_cuda_megakernel_runner)
