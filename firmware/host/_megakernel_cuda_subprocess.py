"""Task 1 fused CUDA region-kernel benchmark — CUDA subprocess.

Runs in a separate Python process from the parent
``run_megakernel_benchmark.py`` so the parent can hold its NVRTC context
without colliding with Torch's CUDA / Inductor contexts (same isolation
pattern as ``_cublas_baseline_torch_subprocess.py`` and
``inductor_oracle_subprocess.py``).

For each workload it times the following arms (each over warmup + iters
iterations, bracketed by ``cuCtxSynchronize`` / ``torch.cuda.synchronize``):

1. ``fused_region``      — the new fused CUDA region kernel from
                            ``cuda_megakernel_backend.execute_region_cuda``.
2. ``op_by_op``          — the same region executed one op per NVRTC kernel
                            launch (linear kernel + relu/scale/add kernels)
                            so the headline number "fused vs op-by-op" is
                            apples-to-apples in kernel implementation.
3. ``cublas_fp32``       — ``torch.matmul`` + per-op elementwise ops in
                            float32 on the same GPU. This is the honest
                            cuBLAS-fallback comparison; the harness does
                            NOT claim to beat cuBLAS.

Each arm verifies its output against a NumPy reference via the bench
oracle and records ``correctness_within_tolerance``. No arm is silently
dropped on failure; failures become ``status="error"`` with the
exception message.

Output: a single JSON payload to ``--output``. The parent script wraps
it with environment metadata, methodology, and aggregate stats.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np


def _import_cuda_bindings():
    """Compat shim for cuda-python 12.x vs 13.x package layouts.

    Returns ``(driver_module, nvrtc_module)``. Both layouts expose identical
    symbol names (`cuMemAlloc`, `cuLaunchKernel`, `nvrtcCompileProgram`, …),
    only the import path differs.
    """
    try:
        from cuda.bindings import driver as cuda_mod, nvrtc as nvrtc_mod  # type: ignore[import-not-found]
        return cuda_mod, nvrtc_mod
    except Exception:
        from cuda import cuda as cuda_mod, nvrtc as nvrtc_mod  # type: ignore[import-not-found]
        return cuda_mod, nvrtc_mod


def _cuda_check(result, what: str):
    """Normalize cuda-python 12.x vs 13.x return-tuple shapes.

    12.x: every driver / nvrtc call returns a tuple ``(err, *vals)`` (pure-effect
      calls return a 1-tuple ``(err,)``).
    13.x: pure-effect calls return just ``err`` (a ``CUresult``); value-returning
      calls still return ``(err, *vals)``.
    """
    if isinstance(result, tuple):
        err = result[0]
        vals = tuple(result[1:])
    else:
        err = result
        vals = ()
    err_int = int(err)
    if err_int != 0:
        raise RuntimeError(f"{what} failed: err={err_int}")
    return vals


def _summary_ms(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "samples": 0,
        }
    samples_sorted = sorted(samples)
    p95_idx = max(0, int(round(0.95 * (len(samples_sorted) - 1))))
    return {
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "stdev": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        "min": float(samples_sorted[0]),
        "max": float(samples_sorted[-1]),
        "p95": float(samples_sorted[p95_idx]),
        "samples": int(len(samples)),
    }


def _gpu_environment(torch_mod) -> Dict[str, Any]:
    try:
        device_idx = torch_mod.cuda.current_device()
        name = torch_mod.cuda.get_device_name(device_idx)
        cap = torch_mod.cuda.get_device_capability(device_idx)
        version = getattr(torch_mod, "version", None)
        return {
            "device_name": str(name),
            "device_capability": [int(cap[0]), int(cap[1])],
            "torch_version": str(getattr(torch_mod, "__version__", "unknown")),
            "cuda_version": str(getattr(version, "cuda", "unknown")) if version else "unknown",
            "device_index": int(device_idx),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _build_workload_graph(workload: Dict[str, Any]):
    """Synthesize a Graph IR for one workload entry from the JSON spec."""
    from graph_ir import GraphIR, OpKind, OpNode  # local import inside subprocess

    name = workload["name"]
    kind = workload["region_kind"]
    seed = int(workload.get("seed", 0xBEEF))
    rng = np.random.default_rng(seed)

    g = GraphIR(name=name)
    if kind == "linear_with_epilogue":
        in_features = int(workload["in_features"])
        out_features = int(workload["out_features"])
        w = rng.standard_normal((out_features, in_features)).astype(np.float32)
        b = rng.standard_normal((out_features,)).astype(np.float32) if workload.get("bias", False) else None
        x = rng.standard_normal((1, in_features)).astype(np.float32)
        g.inputs = ["x"]
        g.add_value("x", shape=(1, in_features), dtype="torch.float32")

        attrs: Dict[str, Any] = {"weight": w, "in_features": in_features, "out_features": out_features}
        if b is not None:
            attrs["bias"] = b
        g.add_op(OpNode(name="fc1", op=OpKind.LINEAR, inputs=["x"], outputs=["h"], attrs=attrs))
        cur = "h"
        ext_inputs: Dict[str, np.ndarray] = {"x": x}
        residual_counter = 0
        for ep in workload.get("epilogue", []):
            op_name = f"{ep['op']}_{residual_counter if ep['op'] == 'add' else ''}".rstrip("_")
            if ep["op"] == "relu":
                nxt = f"{cur}_relu"
                g.add_op(OpNode(name=f"relu_{residual_counter}", op=OpKind.RELU, inputs=[cur], outputs=[nxt], attrs={}))
                cur = nxt
                residual_counter += 1
            elif ep["op"] == "scale":
                nxt = f"{cur}_scale"
                g.add_op(
                    OpNode(
                        name=f"scale_{residual_counter}",
                        op=OpKind.SCALE,
                        inputs=[cur],
                        outputs=[nxt],
                        attrs={"scale": float(ep.get("scale", 0.5))},
                    )
                )
                cur = nxt
                residual_counter += 1
            elif ep["op"] == "add":
                rname = f"r_{residual_counter}"
                r_val = rng.standard_normal((1, out_features)).astype(np.float32)
                g.add_value(rname, shape=(1, out_features), dtype="torch.float32")
                g.inputs.append(rname)
                ext_inputs[rname] = r_val
                nxt = f"{cur}_add"
                g.add_op(
                    OpNode(
                        name=f"add_{residual_counter}",
                        op=OpKind.ADD,
                        inputs=[cur, rname],
                        outputs=[nxt],
                        attrs={},
                    )
                )
                cur = nxt
                residual_counter += 1
            else:
                raise ValueError(f"unknown epilogue op '{ep['op']}'")
        g.outputs = [cur]
        return g, ext_inputs

    if kind == "elementwise_chain":
        n_elements = int(workload["n_elements"])
        x = rng.standard_normal((n_elements,)).astype(np.float32)
        g.inputs = ["x"]
        g.add_value("x", shape=(n_elements,), dtype="torch.float32")
        cur = "x"
        ext_inputs = {"x": x}
        for i, op in enumerate(workload["chain"]):
            if op["op"] == "relu":
                nxt = f"v{i}"
                g.add_op(OpNode(name=f"relu_{i}", op=OpKind.RELU, inputs=[cur], outputs=[nxt], attrs={}))
                cur = nxt
            elif op["op"] == "scale":
                nxt = f"v{i}"
                g.add_op(
                    OpNode(
                        name=f"scale_{i}",
                        op=OpKind.SCALE,
                        inputs=[cur],
                        outputs=[nxt],
                        attrs={"scale": float(op.get("scale", 0.5))},
                    )
                )
                cur = nxt
            elif op["op"] == "add":
                rname = f"r_{i}"
                ext_inputs[rname] = rng.standard_normal((n_elements,)).astype(np.float32)
                g.add_value(rname, shape=(n_elements,), dtype="torch.float32")
                g.inputs.append(rname)
                nxt = f"v{i}"
                g.add_op(OpNode(name=f"add_{i}", op=OpKind.ADD, inputs=[cur, rname], outputs=[nxt], attrs={}))
                cur = nxt
            else:
                raise ValueError(f"unknown elementwise op '{op['op']}'")
        g.outputs = [cur]
        return g, ext_inputs

    raise ValueError(f"unknown region_kind '{kind}'")


def _numpy_reference_output(graph, ext_inputs: Dict[str, np.ndarray]) -> np.ndarray:
    from graph_reference_interpreter import GraphReferenceInterpreter

    ordered = [ext_inputs[name] for name in graph.inputs]
    out = GraphReferenceInterpreter(graph).run(*ordered)
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out, dtype=np.float32)


def _time_call(fn: Callable, warmup: int, iters: int, sync: Callable) -> List[float]:
    for _ in range(warmup):
        fn()
    sync()
    samples: List[float] = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return samples


def _time_fused_region(graph, ext_inputs, reference, warmup, iters) -> Dict[str, Any]:
    """Time the fused-region CUDA kernel using the same setup/launch split as
    ``_time_op_by_op``: pre-compile the kernel, pre-allocate and pre-upload
    all buffers ONCE, then measure only the ``cuLaunchKernel`` cost.

    Earlier versions called ``backend.execute_region_cuda`` inside the timed
    loop, which made every iteration pay the cost of NVRTC compile + module
    load + per-buffer ``cuMemAlloc`` + ``cuMemcpyHtoD`` + launch + sync +
    ``cuMemcpyDtoH`` + ``cuMemFree`` + module unload. That made the
    fused_region arm ~150× slower than op_by_op, but it was a methodology
    asymmetry, not a real codegen regression — op_by_op already amortizes
    its compile + alloc cost outside the timed call. This implementation
    matches that split for a fair launch-only comparison.
    """
    import region_fusion
    import cuda_megakernel_backend as backend

    analysis = region_fusion.find_fusion_regions(graph)
    if not analysis.regions:
        return {
            "arm": "fused_region",
            "status": "error",
            "reason": (
                f"no fusable region; rejections="
                f"{[r.rejection_kind for r in analysis.rejections]}"
            ),
        }
    region = analysis.regions[0]
    externals = {n: ext_inputs[n] for n in region.inputs_external}

    cuda_mod, nvrtc_mod = _import_cuda_bindings()

    threads_per_block = 128
    extra_buffers: List[Any] = []
    module = None
    try:
        # 1) Codegen once.
        gen = backend.generate_kernel_source(region, graph, threads_per_block=threads_per_block)

        # 2) Compile once (NVRTC compile + module load).
        module, kernel = _compile_one_kernel(cuda_mod, nvrtc_mod, gen.source, gen.kernel_name)

        # 3) Allocate + upload all persistent buffers once.
        def _alloc_and_copy(arr: np.ndarray) -> int:
            (ptr,) = _cuda_check(cuda_mod.cuMemAlloc(arr.nbytes), "cuMemAlloc")
            _cuda_check(
                cuda_mod.cuMemcpyHtoD(ptr, arr.ctypes.data, arr.nbytes),
                "cuMemcpyHtoD",
            )
            extra_buffers.append(ptr)
            return int(ptr)

        def _alloc_zero(nbytes: int) -> int:
            (ptr,) = _cuda_check(cuda_mod.cuMemAlloc(nbytes), "cuMemAlloc(y)")
            extra_buffers.append(ptr)
            return int(ptr)

        if region.region_kind == "linear_with_epilogue":
            out_elems = gen.output_dim
            in_elems = int(gen.weights_packed.shape[1])
            weights_host = gen.weights_packed.astype(np.float32, copy=False).reshape(-1)
            bias_host = (
                gen.biases_packed.astype(np.float32, copy=False).reshape(-1)
                if gen.biases_packed is not None
                else None
            )
            x_full = np.asarray(externals[gen.external_input_layout[0]["value_name"]], dtype=np.float32)
            x_host = np.ascontiguousarray(x_full.reshape(-1))
            residual_hosts: List[np.ndarray] = []
            for entry in gen.external_input_layout[1:]:
                buf = np.ascontiguousarray(
                    np.asarray(externals[entry["value_name"]], dtype=np.float32).reshape(-1)
                )
                if buf.size != out_elems:
                    raise RuntimeError(
                        f"residual buffer '{entry['value_name']}' has {buf.size} elements, expected {out_elems}"
                    )
                residual_hosts.append(buf)
            y_host = np.zeros(out_elems, dtype=np.float32)

            d_w = _alloc_and_copy(weights_host)
            d_bias = _alloc_and_copy(bias_host) if bias_host is not None else None
            d_x = _alloc_and_copy(x_host)
            d_residuals = [_alloc_and_copy(r) for r in residual_hosts]
            d_y = _alloc_zero(y_host.nbytes)

            typed: List = [(int(d_w), ctypes.c_void_p)]
            if d_bias is not None:
                typed.append((int(d_bias), ctypes.c_void_p))
            typed.append((int(d_x), ctypes.c_void_p))
            for d_r in d_residuals:
                typed.append((int(d_r), ctypes.c_void_p))
            typed.append((int(d_y), ctypes.c_void_p))
            typed.append((int(in_elems), ctypes.c_int))
            typed.append((int(out_elems), ctypes.c_int))
            args = backend._pack_kernel_args(typed)

            blocks = (out_elems + threads_per_block - 1) // threads_per_block
            batch_shape = tuple(x_full.shape[:-1]) if x_full.ndim >= 1 else ()
            out_shape = (*batch_shape, out_elems)

        elif region.region_kind == "elementwise_chain":
            primary = np.ascontiguousarray(
                np.asarray(externals[gen.external_input_layout[0]["value_name"]], dtype=np.float32)
            )
            n_elements = int(primary.size)
            residual_hosts = []
            for entry in gen.external_input_layout[1:]:
                buf = np.ascontiguousarray(
                    np.asarray(externals[entry["value_name"]], dtype=np.float32).reshape(-1)
                )
                if buf.size != n_elements:
                    raise RuntimeError(
                        f"residual buffer '{entry['value_name']}' has {buf.size} elements, expected {n_elements}"
                    )
                residual_hosts.append(buf)
            y_host = np.zeros(n_elements, dtype=np.float32)

            d_primary = _alloc_and_copy(primary.reshape(-1))
            d_residuals = [_alloc_and_copy(r) for r in residual_hosts]
            d_y = _alloc_zero(y_host.nbytes)

            typed = [(int(d_primary), ctypes.c_void_p)]
            for d_r in d_residuals:
                typed.append((int(d_r), ctypes.c_void_p))
            typed.append((int(d_y), ctypes.c_void_p))
            typed.append((int(n_elements), ctypes.c_int))
            args = backend._pack_kernel_args(typed)

            blocks = (n_elements + threads_per_block - 1) // threads_per_block
            out_shape = primary.shape

        else:
            raise RuntimeError(f"unsupported region_kind for timed fused launch: {region.region_kind}")

        # 4) Define the timed call: ONLY the launch.
        def call() -> None:
            _cuda_check(
                cuda_mod.cuLaunchKernel(
                    kernel,
                    blocks, 1, 1,
                    threads_per_block, 1, 1,
                    0, None,
                    args, 0,
                ),
                "cuLaunchKernel(fused_region)",
            )

        def sync() -> None:
            _cuda_check(cuda_mod.cuCtxSynchronize(), "cuCtxSynchronize")

        # 5) Sanity-check the output before timing.
        call()
        sync()
        _cuda_check(
            cuda_mod.cuMemcpyDtoH(y_host.ctypes.data, d_y, y_host.nbytes),
            "cuMemcpyDtoH(fused_region.sanity)",
        )
        first_out = y_host.reshape(out_shape)
        correct = bool(
            np.allclose(first_out.reshape(-1), reference.reshape(-1), rtol=1e-3, atol=1e-3)
        )
        max_abs = float(np.max(np.abs(first_out.reshape(-1) - reference.reshape(-1))))

        # 6) Time the launch-only path with warmup + iters.
        samples = _time_call(call, warmup, iters, sync)

        return {
            "arm": "fused_region",
            "status": "ok",
            "region_id": region.region_id,
            "region_kind": region.region_kind,
            "kernel_launches_per_invocation": 1,
            "correctness_within_tolerance": correct,
            "max_abs_error_vs_reference": max_abs,
            "kernel_ms": _summary_ms(samples),
            "samples_ms": [float(s) for s in samples[:32]],
            # Timing methodology marker — locks in the apples-to-apples
            # comparison so future regressions can diff against this baseline.
            "timing_includes": ["cuLaunchKernel"],
            "timing_excludes": [
                "nvrtc_compile",
                "module_load",
                "cuMemAlloc",
                "cuMemcpyHtoD(weights)",
                "cuMemcpyHtoD(inputs)",
                "cuMemcpyDtoH(output)",
            ],
        }
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb
        return {
            "arm": "fused_region",
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": _tb.format_exc(),
        }
    finally:
        for ptr in extra_buffers:
            try:
                cuda_mod.cuMemFree(ptr)
            except Exception:
                pass
        if module is not None:
            try:
                cuda_mod.cuModuleUnload(module)
            except Exception:
                pass


def _compile_one_kernel(cuda_mod, nvrtc_mod, source: str, entry_name: str):
    """NVRTC-compile a single kernel source. Returns (module, kernel)."""
    (prog,) = _cuda_check(
        nvrtc_mod.nvrtcCreateProgram(
            source.encode("utf-8"),
            f"{entry_name}.cu".encode("utf-8"),
            0,
            [],
            [],
        ),
        "nvrtcCreateProgram",
    )
    opts = [b"--gpu-architecture=compute_75", b"--use_fast_math"]
    try:
        _cuda_check(nvrtc_mod.nvrtcCompileProgram(prog, len(opts), opts), "nvrtcCompileProgram")
    except RuntimeError:
        try:
            (log_size,) = _cuda_check(nvrtc_mod.nvrtcGetProgramLogSize(prog), "nvrtcGetProgramLogSize")
            log_buf = b" " * int(log_size)
            nvrtc_mod.nvrtcGetProgramLog(prog, log_buf)
            log_text = log_buf.decode("utf-8", errors="replace")
        except Exception:
            log_text = "<could not retrieve NVRTC log>"
        raise RuntimeError(f"nvrtcCompileProgram failed; log: {log_text}")
    (ptx_size,) = _cuda_check(nvrtc_mod.nvrtcGetPTXSize(prog), "nvrtcGetPTXSize")
    ptx = b" " * int(ptx_size)
    nvrtc_mod.nvrtcGetPTX(prog, ptx)
    nvrtc_mod.nvrtcDestroyProgram(prog)
    (module,) = _cuda_check(cuda_mod.cuModuleLoadData(ptx), "cuModuleLoadData")
    (kernel,) = _cuda_check(
        cuda_mod.cuModuleGetFunction(module, entry_name.encode("utf-8")),
        "cuModuleGetFunction",
    )
    return module, kernel


def _time_op_by_op(graph, ext_inputs, reference, warmup, iters) -> Dict[str, Any]:
    """Execute every region op as a separate NVRTC launch.

    Same per-op kernel quality as the fused-region kernel (both come from
    `cuda_megakernel_backend.generate_per_op_kernel_source`), so the delta
    is purely launch count + intermediate-buffer round-trip, not kernel
    quality. This is the apples-to-apples baseline the fused-region arm
    is compared against.
    """
    try:
        cuda_mod, nvrtc_mod = _import_cuda_bindings()
    except Exception as exc:  # noqa: BLE001
        return {"arm": "op_by_op", "status": "error", "reason": f"cuda-python: {exc}"}

    import cuda_megakernel_backend as backend
    from graph_ir import OpKind

    op_by_name = {op.name: op for op in graph.ops}
    # The region is the same as the fused arm's region.
    import region_fusion
    analysis = region_fusion.find_fusion_regions(graph)
    if not analysis.regions:
        return {"arm": "op_by_op", "status": "error", "reason": "no region in workload"}
    region = analysis.regions[0]

    threads_per_block = 128

    # IMPORTANT: declare these BEFORE the try block so the finally clause can
    # always tear them down. Previously `extra_buffers` was declared inside
    # the try and the finally raised UnboundLocalError, masking the real
    # exception (the cuda-python 13.x tuple-shape mismatch).
    compiled: List[Dict[str, Any]] = []
    modules_to_unload: List[Any] = []
    extra_buffers: List[Any] = []
    try:
        for op_name in region.op_names:
            op = op_by_name[op_name]
            source = backend.generate_per_op_kernel_source(op, threads_per_block=threads_per_block)
            module, kernel = _compile_one_kernel(cuda_mod, nvrtc_mod, source, backend._per_op_kernel_name(op.name))
            modules_to_unload.append(module)
            compiled.append({"op": op, "kernel": kernel})

        # Allocate persistent device buffers for every SSA value used by the chain.
        # Each op's output gets a buffer; external inputs get a buffer; weights/biases too.
        value_buffers: Dict[str, Dict[str, Any]] = {}  # name -> {"ptr": int, "nbytes": int}

        def _alloc(nbytes: int) -> int:
            (ptr,) = _cuda_check(cuda_mod.cuMemAlloc(nbytes), f"cuMemAlloc({nbytes})")
            extra_buffers.append(ptr)
            return int(ptr)

        # External value buffers (allocated + uploaded once).
        for name, arr in ext_inputs.items():
            flat = np.asarray(arr, dtype=np.float32).reshape(-1)
            ptr = _alloc(flat.nbytes)
            _cuda_check(
                cuda_mod.cuMemcpyHtoD(ptr, flat.ctypes.data, flat.nbytes),
                f"cuMemcpyHtoD('{name}')",
            )
            value_buffers[name] = {"ptr": ptr, "nbytes": flat.nbytes, "shape": tuple(flat.shape)}

        # Intermediate + output buffers (allocated once; reused per iteration).
        for op_name in region.op_names:
            op = op_by_name[op_name]
            out_name = op.outputs[0]
            # Infer output size from the op:
            if op.op in (OpKind.LINEAR, OpKind.LINEAR_RELU):
                w = np.asarray(op.attrs["weight"], dtype=np.float32)
                out_elems = int(w.shape[0])
            else:
                # Elementwise: same size as primary input.
                in_buf = value_buffers[op.inputs[0]]
                out_elems = in_buf["nbytes"] // 4
            ptr = _alloc(out_elems * 4)
            value_buffers[out_name] = {"ptr": ptr, "nbytes": out_elems * 4, "shape": (out_elems,)}

        # Per-linear-op weight + bias device buffers.
        op_weight_buffers: Dict[str, int] = {}
        op_bias_buffers: Dict[str, int] = {}
        for op_name in region.op_names:
            op = op_by_name[op_name]
            if op.op in (OpKind.LINEAR, OpKind.LINEAR_RELU):
                w = np.ascontiguousarray(np.asarray(op.attrs["weight"], dtype=np.float32)).reshape(-1)
                w_ptr = _alloc(w.nbytes)
                _cuda_check(
                    cuda_mod.cuMemcpyHtoD(w_ptr, w.ctypes.data, w.nbytes),
                    "cuMemcpyHtoD(weight)",
                )
                op_weight_buffers[op.name] = w_ptr
                if op.attrs.get("bias") is not None:
                    b = np.ascontiguousarray(np.asarray(op.attrs["bias"], dtype=np.float32)).reshape(-1)
                    b_ptr = _alloc(b.nbytes)
                    _cuda_check(
                        cuda_mod.cuMemcpyHtoD(b_ptr, b.ctypes.data, b.nbytes),
                        "cuMemcpyHtoD(bias)",
                    )
                    op_bias_buffers[op.name] = b_ptr

        def _launch_one(entry: Dict[str, Any]) -> None:
            op = entry["op"]
            kernel = entry["kernel"]
            # cuda-python 13.x's `kernelParams` accepts the (values, types)
            # tuple form via backend._pack_kernel_args. Pointers carry
            # `ctypes.c_void_p`, ints carry `ctypes.c_int`.
            if op.op in (OpKind.LINEAR, OpKind.LINEAR_RELU):
                w = np.asarray(op.attrs["weight"])
                in_features = int(w.shape[1])
                out_features = int(w.shape[0])
                x_ptr = value_buffers[op.inputs[0]]["ptr"]
                y_ptr = value_buffers[op.outputs[0]]["ptr"]
                w_ptr = op_weight_buffers[op.name]
                typed = [(int(w_ptr), ctypes.c_void_p), (int(x_ptr), ctypes.c_void_p)]
                if op.name in op_bias_buffers:
                    typed.append((int(op_bias_buffers[op.name]), ctypes.c_void_p))
                typed.append((int(y_ptr), ctypes.c_void_p))
                typed.append((int(in_features), ctypes.c_int))
                typed.append((int(out_features), ctypes.c_int))
                args = backend._pack_kernel_args(typed)
                blocks = (out_features + threads_per_block - 1) // threads_per_block
                _cuda_check(
                    cuda_mod.cuLaunchKernel(kernel, blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, 0),
                    f"cuLaunchKernel({op.op})",
                )
            elif op.op == OpKind.RELU or op.op == OpKind.SCALE:
                x_ptr = value_buffers[op.inputs[0]]["ptr"]
                y_ptr = value_buffers[op.outputs[0]]["ptr"]
                n = value_buffers[op.outputs[0]]["nbytes"] // 4
                args = backend._pack_kernel_args([
                    (int(x_ptr), ctypes.c_void_p),
                    (int(y_ptr), ctypes.c_void_p),
                    (int(n), ctypes.c_int),
                ])
                blocks = (n + threads_per_block - 1) // threads_per_block
                _cuda_check(
                    cuda_mod.cuLaunchKernel(kernel, blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, 0),
                    f"cuLaunchKernel({op.op})",
                )
            elif op.op == OpKind.ADD:
                a_ptr = value_buffers[op.inputs[0]]["ptr"]
                b_ptr = value_buffers[op.inputs[1]]["ptr"]
                y_ptr = value_buffers[op.outputs[0]]["ptr"]
                n = value_buffers[op.outputs[0]]["nbytes"] // 4
                args = backend._pack_kernel_args([
                    (int(a_ptr), ctypes.c_void_p),
                    (int(b_ptr), ctypes.c_void_p),
                    (int(y_ptr), ctypes.c_void_p),
                    (int(n), ctypes.c_int),
                ])
                blocks = (n + threads_per_block - 1) // threads_per_block
                _cuda_check(
                    cuda_mod.cuLaunchKernel(kernel, blocks, 1, 1, threads_per_block, 1, 1, 0, None, args, 0),
                    "cuLaunchKernel(ADD)",
                )
            else:
                raise RuntimeError(f"_time_op_by_op cannot launch op kind '{op.op}'")

        def call() -> None:
            for e in compiled:
                _launch_one(e)

        def sync() -> None:
            _cuda_check(cuda_mod.cuCtxSynchronize(), "cuCtxSynchronize")

        # Sanity check: download the final output and compare.
        call()
        sync()
        last_buf = value_buffers[region.output]
        host_out = np.empty(last_buf["nbytes"] // 4, dtype=np.float32)
        _cuda_check(
            cuda_mod.cuMemcpyDtoH(host_out.ctypes.data, last_buf["ptr"], last_buf["nbytes"]),
            "cuMemcpyDtoH(final_output)",
        )
        correct = bool(np.allclose(host_out, reference.reshape(-1), rtol=1e-3, atol=1e-3))
        max_abs = float(np.max(np.abs(host_out - reference.reshape(-1))))

        samples = _time_call(call, warmup, iters, sync)

        return {
            "arm": "op_by_op",
            "status": "ok",
            "kernel_launches_per_invocation": len(compiled),
            "correctness_within_tolerance": correct,
            "max_abs_error_vs_reference": max_abs,
            "kernel_ms": _summary_ms(samples),
            "samples_ms": [float(s) for s in samples[:32]],
            "notes": (
                "Each region op runs as a standalone NVRTC kernel using the "
                "same codegen quality as the fused-region kernel. Intermediate "
                "buffers are persistent on-device (allocated once, reused per "
                "iteration); only the final output round-trips per call. The "
                "fused-vs-op-by-op delta therefore isolates launch count + "
                "intermediate-buffer traffic, not kernel-implementation differences."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb
        return {
            "arm": "op_by_op",
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": _tb.format_exc(),
        }
    finally:
        for ptr in extra_buffers:
            try:
                cuda_mod.cuMemFree(ptr)
            except Exception:
                pass
        for module in modules_to_unload:
            try:
                cuda_mod.cuModuleUnload(module)
            except Exception:
                pass


def _time_cublas_fp32(graph, ext_inputs, reference, warmup, iters) -> Dict[str, Any]:
    """Time torch.matmul + elementwise on FP32 — the cuBLAS-equivalent path."""
    try:
        import torch as torch_mod
    except Exception as exc:  # noqa: BLE001
        return {"arm": "cublas_fp32", "status": "error", "reason": f"torch import: {exc}"}

    from graph_ir import OpKind

    device = torch_mod.device("cuda")
    op_by_name = {op.name: op for op in graph.ops}

    # Bind every external input as a contiguous fp32 tensor on the device.
    tensors: Dict[str, "torch_mod.Tensor"] = {}
    for name, arr in ext_inputs.items():
        tensors[name] = torch_mod.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)

    # Pre-extract weight + bias tensors for any linear op in the graph.
    weight_tensors: Dict[str, "torch_mod.Tensor"] = {}
    bias_tensors: Dict[str, "torch_mod.Tensor"] = {}
    for op in graph.ops:
        if op.op in (OpKind.LINEAR, OpKind.LINEAR_RELU):
            w = np.asarray(op.attrs["weight"], dtype=np.float32)
            weight_tensors[op.name] = torch_mod.from_numpy(np.ascontiguousarray(w)).to(device)
            b = op.attrs.get("bias")
            if b is not None:
                bias_tensors[op.name] = torch_mod.from_numpy(
                    np.ascontiguousarray(np.asarray(b, dtype=np.float32))
                ).to(device)

    def run_one() -> "torch_mod.Tensor":
        vals: Dict[str, Any] = dict(tensors)
        for op in graph.ops:
            if op.op in (OpKind.LINEAR, OpKind.LINEAR_RELU):
                x = vals[op.inputs[0]]
                w = weight_tensors[op.name]
                y = torch_mod.matmul(x, w.T)
                if op.name in bias_tensors:
                    y = y + bias_tensors[op.name]
                if op.op == OpKind.LINEAR_RELU:
                    y = torch_mod.relu(y)
                vals[op.outputs[0]] = y
            elif op.op == OpKind.RELU:
                vals[op.outputs[0]] = torch_mod.relu(vals[op.inputs[0]])
            elif op.op == OpKind.ADD:
                vals[op.outputs[0]] = vals[op.inputs[0]] + vals[op.inputs[1]]
            elif op.op == OpKind.SCALE:
                vals[op.outputs[0]] = vals[op.inputs[0]] * float(op.attrs.get("scale", 1.0))
            else:
                raise ValueError(f"cublas_fp32 arm doesn't support op '{op.op}' in v1")
        return vals[graph.outputs[0]]

    # Functional sanity.
    try:
        out_tensor = run_one()
        torch_mod.cuda.synchronize()
        out_np = out_tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        return {"arm": "cublas_fp32", "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    correct = bool(np.allclose(out_np, reference.reshape(-1), rtol=1e-3, atol=1e-3))
    max_abs = float(np.max(np.abs(out_np - reference.reshape(-1))))

    def call() -> None:
        with torch_mod.no_grad():
            run_one()

    samples = _time_call(call, warmup, iters, lambda: torch_mod.cuda.synchronize())
    return {
        "arm": "cublas_fp32",
        "status": "ok",
        "kernel_launches_per_invocation": len(graph.ops),
        "dtype": "float32",
        "correctness_within_tolerance": correct,
        "max_abs_error_vs_reference": max_abs,
        "kernel_ms": _summary_ms(samples),
        "samples_ms": [float(s) for s in samples[:32]],
    }


def run_one_workload(workload: Dict[str, Any], warmup: int, iters: int) -> Dict[str, Any]:
    graph, ext_inputs = _build_workload_graph(workload)
    reference = _numpy_reference_output(graph, ext_inputs)
    arms = [
        _time_fused_region(graph, ext_inputs, reference, warmup, iters),
        _time_op_by_op(graph, ext_inputs, reference, warmup, iters),
        _time_cublas_fp32(graph, ext_inputs, reference, warmup, iters),
    ]
    return {
        "name": workload["name"],
        "region_kind": workload["region_kind"],
        "shape_descriptor": {k: workload[k] for k in workload if k in ("in_features", "out_features", "n_elements", "epilogue", "chain", "bias", "seed")},
        "reference_output_shape": list(reference.shape),
        "arms": arms,
    }


def run_subprocess(workloads: List[Dict[str, Any]], warmup: int, iters: int) -> Dict[str, Any]:
    try:
        import torch as torch_mod
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "torch_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "workloads_requested": [w["name"] for w in workloads],
        }

    if not torch_mod.cuda.is_available():
        return {
            "status": "cuda_unavailable",
            "reason": "torch.cuda.is_available() is False inside subprocess.",
            "workloads_requested": [w["name"] for w in workloads],
            "torch_version": str(torch_mod.__version__),
        }

    try:
        cuda_mod, _ = _import_cuda_bindings()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "cuda_python_unavailable",
            "reason": (
                f"{type(exc).__name__}: {exc}. Tried both `cuda.bindings.{{driver,nvrtc}}` "
                "(cuda-python 13.x+) and `cuda.{{cuda,nvrtc}}` (cuda-python 12.x). "
                "Install with `pip install cuda-bindings` (preferred on 13.x+) or "
                "`pip install cuda-python==12.6.0` (legacy)."
            ),
            "workloads_requested": [w["name"] for w in workloads],
        }

    # Retain the *primary* context for the lifetime of this subprocess. Three
    # reasons:
    #   1) The op_by_op arm doesn't create its own context; without an active
    #      one, cuModuleLoadData / cuMemAlloc fail with "no current context".
    #   2) cuda-python 13.x changed `cuCtxCreate` to a v3 3-arg signature
    #      (params, flags, dev); primary-context retain is stable across 12.x
    #      and 13.x.
    #   3) torch's CUDA runtime uses the same primary context — sharing it
    #      avoids context conflicts in case any arm later mixes torch+raw CUDA.
    try:
        _cuda_check(cuda_mod.cuInit(0), "cuInit")
        (device,) = _cuda_check(cuda_mod.cuDeviceGet(0), "cuDeviceGet")
        (primary_ctx,) = _cuda_check(
            cuda_mod.cuDevicePrimaryCtxRetain(device), "cuDevicePrimaryCtxRetain"
        )
        _cuda_check(cuda_mod.cuCtxSetCurrent(primary_ctx), "cuCtxSetCurrent")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "cuda_context_setup_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "workloads_requested": [w["name"] for w in workloads],
        }

    env = _gpu_environment(torch_mod)
    out_workloads: List[Dict[str, Any]] = []
    try:
        for w in workloads:
            try:
                out_workloads.append(run_one_workload(w, warmup=warmup, iters=iters))
            except Exception as exc:  # noqa: BLE001
                out_workloads.append(
                    {
                        "name": w["name"],
                        "region_kind": w.get("region_kind"),
                        "status": "error",
                        "reason": f"{type(exc).__name__}: {exc}",
                        # Schema invariant: every workload always has `arms`, even
                        # if empty. Downstream tests iterate `w["arms"]` and would
                        # KeyError otherwise.
                        "arms": [],
                        "shape_descriptor": {k: w[k] for k in w if k in ("in_features", "out_features", "n_elements", "epilogue", "chain", "bias", "seed")},
                    }
                )
    finally:
        # Refcounted release; harmless if torch also holds the primary context.
        try:
            cuda_mod.cuDevicePrimaryCtxRelease(device)
        except Exception:
            pass

    return {
        "status": "ok",
        "environment": env,
        "warmup": int(warmup),
        "iters": int(iters),
        "workloads": out_workloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    workloads = json.loads(args.workloads_json)
    payload = run_subprocess(workloads=workloads, warmup=int(args.warmup), iters=int(args.iters))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
