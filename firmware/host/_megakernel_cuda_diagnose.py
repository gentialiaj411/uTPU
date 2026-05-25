"""Diagnostic probe for cuda-python 13.x kernel-launch surface.

Runs a minimal NVRTC compile + launch using each style of `kernelParams`
known to cuda-python so we can pinpoint which packing shape the installed
version accepts. Prints PASS / FAIL per style with the exception.

Usage:
    python firmware/host/_megakernel_cuda_diagnose.py

Side effects: allocates ~16 bytes on device 0, launches a 1-block kernel
that writes one float, copies it back, frees. No persistent state.
"""

from __future__ import annotations

import sys
import traceback as _tb

import numpy as np


def _import_cuda():
    try:
        from cuda.bindings import driver as cu, nvrtc as nv
        return cu, nv, "cuda.bindings (13.x+)"
    except Exception:  # noqa: BLE001
        from cuda import cuda as cu, nvrtc as nv  # type: ignore[import-not-found]
        return cu, nv, "cuda.{cuda,nvrtc} (12.x)"


def _norm(result):
    if isinstance(result, tuple):
        err = result[0]
        vals = tuple(result[1:])
    else:
        err = result
        vals = ()
    if int(err) != 0:
        raise RuntimeError(f"CUDA call returned err={int(err)}")
    return vals


KERNEL_SRC = r"""
extern "C" __global__ void k_set(float* out, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = (float)value;
    }
}
""".strip()


def _compile_and_setup(cu, nv):
    (prog,) = _norm(nv.nvrtcCreateProgram(KERNEL_SRC.encode("utf-8"), b"k_set.cu", 0, [], []))
    opts = [b"--gpu-architecture=compute_75"]
    try:
        _norm(nv.nvrtcCompileProgram(prog, len(opts), opts))
    except Exception as e:
        try:
            (sz,) = _norm(nv.nvrtcGetProgramLogSize(prog))
            buf = b" " * int(sz)
            nv.nvrtcGetProgramLog(prog, buf)
            print("  NVRTC log:", buf.decode(errors="replace"))
        except Exception:
            pass
        raise
    (ptxsz,) = _norm(nv.nvrtcGetPTXSize(prog))
    ptx = b" " * int(ptxsz)
    nv.nvrtcGetPTX(prog, ptx)
    nv.nvrtcDestroyProgram(prog)
    (mod,) = _norm(cu.cuModuleLoadData(ptx))
    (kern,) = _norm(cu.cuModuleGetFunction(mod, b"k_set"))
    return mod, kern


def _alloc():
    cu, _, _ = _import_cuda()
    (d_out,) = _norm(cu.cuMemAlloc(4))
    return d_out


def _readback(d_out):
    cu, _, _ = _import_cuda()
    out = np.zeros(1, dtype=np.float32)
    _norm(cu.cuMemcpyDtoH(out.ctypes.data, d_out, 4))
    return float(out[0])


def _try(name, fn):
    print(f"\n[probe] style = {name}")
    try:
        result = fn()
        print(f"  PASS — readback={result}")
        return True
    except Exception as e:
        print(f"  FAIL — {type(e).__name__}: {e}")
        for line in _tb.format_exc().splitlines()[-6:]:
            print(f"    {line}")
        return False


def style_list_of_typed_buffers(cu, nv, kern):
    """Each arg is np.array([v], dtype=...). My current code uses this."""
    d_out = _alloc()
    try:
        args = [
            np.array([int(d_out)], dtype=np.uint64),
            np.array([7], dtype=np.int32),
        ]
        _norm(cu.cuLaunchKernel(kern, 1, 1, 1, 1, 1, 1, 0, None, args, None))
        _norm(cu.cuCtxSynchronize())
        return _readback(d_out)
    finally:
        cu.cuMemFree(d_out)


def style_tuple_of_typed_buffers(cu, nv, kern):
    """Same as above but tuple wrapper."""
    d_out = _alloc()
    try:
        args = (
            np.array([int(d_out)], dtype=np.uint64),
            np.array([7], dtype=np.int32),
        )
        _norm(cu.cuLaunchKernel(kern, 1, 1, 1, 1, 1, 1, 0, None, args, None))
        _norm(cu.cuCtxSynchronize())
        return _readback(d_out)
    finally:
        cu.cuMemFree(d_out)


def style_values_and_types(cu, nv, kern):
    """13.x `(values_tuple, types_tuple)` form using ctypes types."""
    import ctypes
    d_out = _alloc()
    try:
        kargs = ((d_out, 7), (None, ctypes.c_int))
        _norm(cu.cuLaunchKernel(kern, 1, 1, 1, 1, 1, 1, 0, None, kargs, 0))
        _norm(cu.cuCtxSynchronize())
        return _readback(d_out)
    finally:
        cu.cuMemFree(d_out)


def style_bare_python_ints(cu, nv, kern):
    """Old 12.x style: list of Python ints."""
    d_out = _alloc()
    try:
        args = [int(d_out), 7]
        _norm(cu.cuLaunchKernel(kern, 1, 1, 1, 1, 1, 1, 0, None, args, None))
        _norm(cu.cuCtxSynchronize())
        return _readback(d_out)
    finally:
        cu.cuMemFree(d_out)


def style_packed_void_p_array(cu, nv, kern):
    """ctypes-style: array of c_void_p pointing at typed values."""
    import ctypes
    d_out = _alloc()
    try:
        arg0 = ctypes.c_void_p(int(d_out))
        arg1 = ctypes.c_int(7)
        argv = (ctypes.c_void_p * 2)(
            ctypes.cast(ctypes.pointer(arg0), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(arg1), ctypes.c_void_p),
        )
        _norm(cu.cuLaunchKernel(kern, 1, 1, 1, 1, 1, 1, 0, None, argv, 0))
        _norm(cu.cuCtxSynchronize())
        return _readback(d_out)
    finally:
        cu.cuMemFree(d_out)


def main():
    cu, nv, layout = _import_cuda()
    print(f"[probe] cuda-python layout: {layout}")
    _norm(cu.cuInit(0))
    (dev,) = _norm(cu.cuDeviceGet(0))
    (ctx,) = _norm(cu.cuDevicePrimaryCtxRetain(dev))
    _norm(cu.cuCtxSetCurrent(ctx))
    try:
        mod, kern = _compile_and_setup(cu, nv)
        try:
            results = {
                "list-of-typed-buffers   (current code)": _try("list-of-typed-buffers", lambda: style_list_of_typed_buffers(cu, nv, kern)),
                "tuple-of-typed-buffers": _try("tuple-of-typed-buffers", lambda: style_tuple_of_typed_buffers(cu, nv, kern)),
                "(values,types)-tuple    (13.x docs)": _try("(values,types)-tuple", lambda: style_values_and_types(cu, nv, kern)),
                "bare-python-ints        (12.x legacy)": _try("bare-python-ints", lambda: style_bare_python_ints(cu, nv, kern)),
                "ctypes c_void_p array": _try("ctypes-c_void_p-array", lambda: style_packed_void_p_array(cu, nv, kern)),
            }
            print("\n[probe] SUMMARY:")
            for k, v in results.items():
                print(f"  {'PASS' if v else 'FAIL'}  {k}")
            winners = [k for k, v in results.items() if v]
            if winners:
                print(f"\n[probe] Use this kernelParams shape in our code: {winners[0]}")
            else:
                print("\n[probe] No style worked. Investigate cuda-python install:")
                import importlib.metadata
                try:
                    print("  cuda-python:", importlib.metadata.version("cuda-python"))
                except Exception:
                    pass
                try:
                    print("  cuda-bindings:", importlib.metadata.version("cuda-bindings"))
                except Exception:
                    pass
        finally:
            cu.cuModuleUnload(mod)
    finally:
        cu.cuDevicePrimaryCtxRelease(dev)


if __name__ == "__main__":
    sys.exit(main() or 0)
