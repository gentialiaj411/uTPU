"""Phase 7 — Serious CUDA baseline harness.

Compares the uTPU autotuner-selected, NVRTC-compiled blocked-FC kernel
against two "serious-library" references on a fixed shape grid:

* **cuBLAS** via ``torch.matmul`` (INT32 weights / INT32 activations /
  INT32 accumulator). This is the same accumulator dtype the uTPU
  kernel uses, so it is the apples-to-apples GEMV comparison.
* **TorchInductor** via ``torch.compile(nn.Linear, backend="inductor")``.
  This is the "what would the official PyTorch compiler do?"
  reference. Runs in FP32 — the artifact records this dtype mismatch
  explicitly so the comparison is never silently inflated.

Methodology (locked, identical for all three backends):

* Six fixed shapes: ``(M, K)`` with N=1 (FC-inference GEMV). Shape set
  is documented in :data:`SHAPES` and embedded in the artifact.
* Warmup = 10 invocations (discarded). Then iters = 50 measured
  invocations bracketed by ``torch.cuda.synchronize()``.
* Per-shape stats: mean, median, stdev, min, max, p95 (ms).
* TFLOPS computed as ``2 * M * N * K / median_ms * 1e-12`` for INT-MAC
  ops (uTPU + cuBLAS). Inductor TFLOPS is reported separately because
  the op count is the same but the dtype is not.
* GPU info recorded: device name, capability, Torch/CUDA versions,
  device index.

The harness is **GPU-gated**: on a non-CUDA host it writes a stub
artifact with ``status="cuda_unavailable"``, the full shape set that
would have been measured, and an explicit "re-run on a CUDA host"
instruction. The schema (top-level keys + shape entry shape) is
identical in both modes so :mod:`test_cublas_baseline` can lock the
contract on either host class.

Subprocess isolation: the Torch baselines run in
:mod:`_cublas_baseline_torch_subprocess` to avoid Torch/Inductor and
NVRTC fighting over the same CUDA driver context — the same pattern
used by ``inductor_oracle_subprocess.py`` for ResNet-18 Inductor
parity.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from cuda_blocked_fc_backend import (  # noqa: E402
    CUDABlockedFCExecutor,
    detect_cuda_environment,
)
from lowering_types import BlockedFCLoweringRequest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "cublas_baseline.json"
SUBPROCESS_SCRIPT = Path(__file__).with_name("_cublas_baseline_torch_subprocess.py")

# Locked shape set. M = out_features (rows of the weight matrix), K =
# in_features (columns), N = 1 (single-vector FC inference). Picked to
# span the small / medium / large regions of the calibration grid so
# the cuBLAS-launch-overhead vs cuBLAS-compute-bound regimes are both
# represented and a single "X% of cuBLAS" headline cannot be cherry-
# picked across the artifact.
SHAPES: List[Dict[str, int]] = [
    {"M": 16, "K": 16},
    {"M": 64, "K": 64},
    {"M": 128, "K": 128},
    {"M": 256, "K": 256},
    {"M": 512, "K": 256},
    {"M": 512, "K": 512},
]

ARRAY_SIZE = 16
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


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


def _int_tflops(M: int, K: int, ms: float) -> float:
    if ms <= 0.0:
        return 0.0
    ops = 2.0 * float(M) * float(K) * 1.0  # N=1
    return ops / (ms / 1000.0) / 1e12


@dataclass
class UtpuShapeResult:
    M: int
    K: int
    schedule_params: Dict[str, Any]
    kernel_ms_summary: Dict[str, float]
    samples_ms: List[float]
    int_tflops_median: float
    compile_time_ms: float
    setup_time_ms: float
    bit_exact_match_vs_numpy_reference: bool


def _run_utpu_shape(
    executor: CUDABlockedFCExecutor,
    M: int,
    K: int,
    warmup: int,
    iters: int,
    rng_seed: int,
) -> UtpuShapeResult:
    rng = np.random.default_rng(rng_seed)
    w = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
    x = rng.integers(-8, 8, size=(K,), dtype=np.int8)
    req = BlockedFCLoweringRequest(
        weights_int4=w,
        activations_int4=x,
        out_features=M,
        in_features=K,
        array_size=ARRAY_SIZE,
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )
    for _ in range(warmup):
        out = executor.execute(req)
        if not out.get("executed", False):
            raise RuntimeError(
                f"uTPU warmup failed at (M={M}, K={K}): {out.get('reason')}"
            )

    samples_ms: List[float] = []
    compile_time_ms = 0.0
    setup_time_ms = 0.0
    schedule_params: Dict[str, Any] = {}
    bit_exact = True
    for i in range(iters):
        out = executor.execute(req)
        if not out.get("executed", False):
            raise RuntimeError(
                f"uTPU iter {i} failed at (M={M}, K={K}): {out.get('reason')}"
            )
        samples_ms.append(float(out["kernel_time_ms"]))
        if i == 0:
            compile_time_ms = float(out.get("compile_time_ms", 0.0))
            setup_time_ms = float(out.get("setup_time_ms", 0.0))
            schedule_params = dict(out.get("schedule_params", {}))
        bit_exact = bit_exact and bool(out.get("bit_exact_match_vs_numpy_reference", True))

    summary = _summary_ms(samples_ms)
    return UtpuShapeResult(
        M=M,
        K=K,
        schedule_params=schedule_params,
        kernel_ms_summary=summary,
        samples_ms=[float(s) for s in samples_ms[:32]],
        int_tflops_median=_int_tflops(M, K, summary["median"]),
        compile_time_ms=compile_time_ms,
        setup_time_ms=setup_time_ms,
        bit_exact_match_vs_numpy_reference=bit_exact,
    )


def _run_torch_subprocess(
    shapes: List[Dict[str, int]],
    warmup: int,
    iters: int,
    tmp_path: Path,
) -> Dict[str, Any]:
    shapes_json = json.dumps(shapes)
    proc = subprocess.run(
        [
            sys.executable,
            str(SUBPROCESS_SCRIPT),
            "--shapes-json",
            shapes_json,
            "--output",
            str(tmp_path),
            "--warmup",
            str(int(warmup)),
            "--iters",
            str(int(iters)),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        return {
            "status": "subprocess_failed",
            "returncode": int(proc.returncode),
            "stderr": proc.stderr[-2000:],
            "stdout": proc.stdout[-2000:],
            "shapes_requested": shapes,
        }
    try:
        return json.loads(tmp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "subprocess_json_parse_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "stdout": proc.stdout[-2000:],
            "shapes_requested": shapes,
        }


def _merge_per_shape(
    utpu: UtpuShapeResult,
    torch_entry: Dict[str, Any] | None,
) -> Dict[str, Any]:
    cublas = (torch_entry or {}).get("cublas")
    inductor = (torch_entry or {}).get("inductor")

    utpu_median = utpu.kernel_ms_summary.get("median", 0.0)
    cublas_median = (cublas or {}).get("kernel_ms", {}).get("median")
    inductor_median = (inductor or {}).get("kernel_ms", {}).get("median")

    def _gap_pct(ours: float, baseline: float | None) -> float | None:
        if baseline is None or baseline <= 0.0 or ours <= 0.0:
            return None
        return float((ours - baseline) / baseline * 100.0)

    return {
        "shape": {"M": int(utpu.M), "K": int(utpu.K), "N": 1},
        "utpu": {
            "backend": "utpu_blocked_fc_nvrtc_int4",
            "dtype_W": "int8",
            "dtype_x": "int8",
            "dtype_accum": "int32",
            "dtype_out": "int4_quantized",
            "schedule_params": utpu.schedule_params,
            "kernel_ms": dict(utpu.kernel_ms_summary),
            "samples_ms": list(utpu.samples_ms),
            "int_mac_tflops_median": float(utpu.int_tflops_median),
            "compile_time_ms": float(utpu.compile_time_ms),
            "setup_time_ms": float(utpu.setup_time_ms),
            "bit_exact_match_vs_numpy_reference": bool(
                utpu.bit_exact_match_vs_numpy_reference
            ),
        },
        "cublas": cublas,
        "inductor": inductor,
        "gap_vs_cublas_pct_median": _gap_pct(utpu_median, cublas_median),
        "gap_vs_inductor_pct_median": _gap_pct(utpu_median, inductor_median),
    }


def _aggregate(per_shape: List[Dict[str, Any]]) -> Dict[str, Any]:
    cublas_gaps = [
        s["gap_vs_cublas_pct_median"]
        for s in per_shape
        if s.get("gap_vs_cublas_pct_median") is not None
    ]
    inductor_gaps = [
        s["gap_vs_inductor_pct_median"]
        for s in per_shape
        if s.get("gap_vs_inductor_pct_median") is not None
    ]
    return {
        "cublas_gap_pct_median_of_shapes": (
            float(statistics.median(cublas_gaps)) if cublas_gaps else None
        ),
        "cublas_gap_pct_mean_of_shapes": (
            float(statistics.fmean(cublas_gaps)) if cublas_gaps else None
        ),
        "cublas_gap_pct_max_of_shapes": (
            float(max(cublas_gaps)) if cublas_gaps else None
        ),
        "cublas_gap_pct_min_of_shapes": (
            float(min(cublas_gaps)) if cublas_gaps else None
        ),
        "inductor_gap_pct_median_of_shapes": (
            float(statistics.median(inductor_gaps)) if inductor_gaps else None
        ),
        "inductor_gap_pct_mean_of_shapes": (
            float(statistics.fmean(inductor_gaps)) if inductor_gaps else None
        ),
        "shapes_compared_vs_cublas": int(len(cublas_gaps)),
        "shapes_compared_vs_inductor": int(len(inductor_gaps)),
    }


def _methodology_block(shapes: List[Dict[str, int]], warmup: int, iters: int) -> Dict[str, Any]:
    return {
        "shape_set": shapes,
        "shape_set_description": (
            "Six fixed (M, K) shapes covering small (16x16) to "
            "medium-large (512x512) GEMV; N=1 in all cases (FC "
            "inference). Locked in firmware/host/run_cublas_baseline.py"
            "::SHAPES; changes to this list invalidate prior artifacts."
        ),
        "warmup_iterations": int(warmup),
        "timed_iterations": int(iters),
        "timing_protocol": (
            "Each measured iteration is bracketed by "
            "torch.cuda.synchronize() (for cuBLAS / Inductor) or "
            "cuCtxSynchronize() (for the uTPU NVRTC kernel) and timed "
            "with time.perf_counter() in ms. Warmup invocations are "
            "discarded. Reported: mean, median, stdev, min, max, p95."
        ),
        "tflops_definition": (
            "2 * M * N * K / median_kernel_ms * 1e-12. INT-MAC count "
            "for uTPU + cuBLAS (both INT32 accumulator); Inductor runs "
            "FP32 so its TFLOPS would not be comparable and is omitted."
        ),
        "dtype_caveats": [
            "uTPU kernel: INT8 inputs, INT32 accumulator, INT4-quantised "
            "output (matches the FPGA datapath). Output quantisation is "
            "applied post-cuBLAS-equivalent GEMV and is included in the "
            "uTPU kernel_time_ms.",
            "cuBLAS reference: the subprocess attempts INT32 inputs / "
            "INT32 accumulator via torch.matmul (apples-to-apples "
            "accumulator dtype with the uTPU kernel). Torch builds "
            "without an INT32 addmv/addmm CUDA kernel (observed on "
            "Torch 2.11+cu130 with the RTX 5070 sm_120) raise "
            "NotImplementedError and the subprocess falls back to FP32 "
            "inputs / FP32 accumulator via the standard cuBLAS GEMV "
            "path. The per-shape entry records the actual measured "
            "dtype in dtype_W/dtype_x/dtype_accum/dtype_out plus an "
            "explicit dtype_fallback_reason on the fallback path. The "
            "uTPU-vs-cuBLAS gap on the FP32-fallback path is not a "
            "dtype-matched comparison and the writeup flags this "
            "verbatim; no apples-to-apples INT32 cuBLAS gap is claimed "
            "until a Torch build with INT32 cuBLAS matmul (or a CuPy / "
            "cublasGemmEx wrapper) is wired in.",
            "Inductor reference: FP32 nn.Linear compiled via "
            "torch.compile(backend='inductor', fullgraph=True). Same op "
            "semantics but FP32 dtype throughout. Recorded as a "
            "framework reference, not a dtype-matched comparison; "
            "gap_vs_inductor_pct is reported but the writeup flags this "
            "caveat explicitly.",
        ],
        "isolation": (
            "cuBLAS + Inductor timings run in a separate Python process "
            "(_cublas_baseline_torch_subprocess.py) so the parent's "
            "NVRTC driver context for the uTPU kernel does not collide "
            "with Torch's CUDA / Inductor contexts (the same isolation "
            "pattern as inductor_oracle_subprocess.py)."
        ),
        "rng_seed_per_shape": "0xC0DE XOR'd with M*1009 + K for the uTPU "
                              "kernel; 0xC0DE / 0xACE inside the Torch subprocess.",
        "scope": (
            "Sim/host-measured. No physical-board claim. cuBLAS / "
            "Inductor numbers depend on the GPU + driver of the host "
            "that regenerates the artifact; uTPU numbers depend on "
            "NVRTC + cuda-python on the same host."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7 cuBLAS / Inductor baseline vs uTPU NVRTC kernel."
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="Discarded warmup invocations."
    )
    parser.add_argument(
        "--iters", type=int, default=DEFAULT_ITERS, help="Measured invocations per shape per backend."
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_JSON),
        help="Output JSON path."
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methodology = _methodology_block(SHAPES, args.warmup, args.iters)
    common_header = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "methodology": methodology,
    }

    env = detect_cuda_environment()
    if not env.runtime_available:
        payload = dict(
            common_header,
            status="cuda_unavailable",
            reason=env.reason
            or "detect_cuda_environment().runtime_available is False on this host.",
            instructions=[
                "1. Run on a CUDA-capable host with cuda-python + driver + NVRTC installed.",
                "2. Install Torch with CUDA support (and TorchInductor + gcc for the Inductor path).",
                "3. Re-execute: python firmware/host/run_cublas_baseline.py",
                "4. The artifact regenerates from the locked SHAPES set; do not edit it by hand.",
            ],
            shapes_requested=SHAPES,
            per_shape=[],
            aggregate=None,
            torch_subprocess=None,
        )
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[cublas_baseline] cuda_unavailable; wrote stub to {output_path}")
        return 0

    tmp_path = output_path.with_suffix(".torch_subprocess.json")
    torch_payload = _run_torch_subprocess(
        SHAPES, warmup=args.warmup, iters=args.iters, tmp_path=tmp_path,
    )

    executor = CUDABlockedFCExecutor(verbose=False)
    per_shape: List[Dict[str, Any]] = []
    torch_shape_index: Dict[tuple, Dict[str, Any]] = {}
    for entry in torch_payload.get("shapes", []) or []:
        sh = entry.get("shape", {})
        torch_shape_index[(int(sh.get("M", 0)), int(sh.get("K", 0)))] = entry

    for shape in SHAPES:
        M, K = int(shape["M"]), int(shape["K"])
        utpu = _run_utpu_shape(
            executor,
            M=M,
            K=K,
            warmup=args.warmup,
            iters=args.iters,
            rng_seed=0xC0DE ^ (M * 1009 + K),
        )
        per_shape.append(
            _merge_per_shape(utpu, torch_shape_index.get((M, K)))
        )

    aggregate = _aggregate(per_shape)
    payload = dict(
        common_header,
        status="ok",
        environment={
            "uTPU_cuda_runtime_available": True,
            "torch_subprocess_status": torch_payload.get("status"),
            "torch_subprocess_environment": torch_payload.get("environment"),
        },
        shapes_requested=SHAPES,
        per_shape=per_shape,
        aggregate=aggregate,
        torch_subprocess=torch_payload,
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    print(f"[cublas_baseline] wrote {output_path}")
    if aggregate.get("cublas_gap_pct_median_of_shapes") is not None:
        print(
            "[cublas_baseline] median uTPU-vs-cuBLAS gap: "
            f"{aggregate['cublas_gap_pct_median_of_shapes']:.2f}% "
            f"(across {aggregate['shapes_compared_vs_cublas']} shapes)"
        )
    if aggregate.get("inductor_gap_pct_median_of_shapes") is not None:
        print(
            "[cublas_baseline] median uTPU-vs-Inductor gap: "
            f"{aggregate['inductor_gap_pct_median_of_shapes']:.2f}% "
            "(dtype caveat applies, see methodology block)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
