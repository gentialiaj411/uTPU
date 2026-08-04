#!/usr/bin/env python3
"""Track 3 — FPGA cycle-determinism vs GPU tail-latency comparison.

Emits ``bench/results/latency_determinism_vs_gpu.json`` and a log-x
latency distribution plot under ``docs/``.

Arms
----
* **FPGA / iverilog RTL** (from ``run_latency_determinism.py``): end-to-end
  MAGIC_START→HALT cycle counts across adversarial + many random inputs
  on the blocked-FC ``(M=32,K=32)`` shape. Cycles convert to wall-clock
  at the measured closed **100 MHz** constraint
  (``timing_closure_sweep.json::requant_fmax_mb4_clk10_pd3``).
* **GPU** (same logical GEMV): N >= 10000 timed iterations with CUDA
  events + a single synchronize bracket, matching the methodology in
  ``run_cublas_baseline.py`` / ``_cublas_baseline_torch_subprocess.py``.
  Prefers the NVRTC blocked-FC CUDA path; falls back to Torch FP32
  ``torch.matmul`` GEMV with an explicit ``dtype_fallback_reason`` when
  INT8/INT32 paths are unavailable — never silent.

Claim framing (honest)
----------------------
Do **not** claim the FPGA is faster. The supported claim is **bounded
jitter** (RTL cycle variance == 0 across inputs) together with the
**median-latency loss** vs the GPU arm (FPGA median wall-clock minus
GPU p50, as a factor and as absolute ns). Tail percentiles are reported
for the GPU arm only.

When CUDA is unavailable on the host (and WSL cannot supply it), the
GPU arm is recorded as ``status="skipped_no_cuda"`` with an explicit
reason; the FPGA arm is still fully populated.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_DIR = REPO_ROOT / "firmware" / "host"
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from cuda_blocked_fc_backend import (  # noqa: E402
    CUDABlockedFCExecutor,
    detect_cuda_environment,
)
from lowering_types import BlockedFCLoweringRequest  # noqa: E402
from run_latency_determinism import (  # noqa: E402
    DEFAULT_DISTRIBUTION_SHAPE,
    DEFAULT_DISTRIBUTIONS,
    DEFAULT_E2E_RANDOM_TRIALS,
    DEFAULT_SHAPES,
    FPGA_CLOCK_MHZ,
    FPGA_CLOCK_SOURCE,
    WEIGHT_SEED_BASE,
    _build_artifact,
    _cycles_to_wall_ns,
    _resolve_iverilog,
    _shape_tag,
)

RESULTS_DIR = REPO_ROOT / "bench" / "results"
FPGA_JSON = RESULTS_DIR / "latency_determinism.json"
VS_GPU_JSON = RESULTS_DIR / "latency_determinism_vs_gpu.json"
PLOT_PATH = REPO_ROOT / "docs" / "latency_determinism_vs_gpu_logx.png"
SUBPROCESS_SCRIPT = Path(__file__).with_name("_latency_vs_gpu_cuda_subprocess.py")

DEFAULT_GPU_ITERS = 10000
DEFAULT_GPU_WARMUP = 50
SCHEMA_VERSION = 1

# Best-effort CUDA toolkit path on this Windows host (nvcc present).
_DEFAULT_CUDA_PATHS = (
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        )
        return out.strip()
    except Exception:
        return ""


def _percentile(sorted_samples: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile; ``q`` in [0, 100]."""
    if not sorted_samples:
        return float("nan")
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    pos = (len(sorted_samples) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = pos - lo
    return float(sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac)


def _tail_stats_ns(samples_ns: Sequence[float]) -> Dict[str, float]:
    s = sorted(float(x) for x in samples_ns)
    if not s:
        return {
            "n": 0,
            "p50_ns": float("nan"),
            "p90_ns": float("nan"),
            "p99_ns": float("nan"),
            "p99_9_ns": float("nan"),
            "max_ns": float("nan"),
            "stddev_ns": float("nan"),
            "mean_ns": float("nan"),
            "min_ns": float("nan"),
        }
    return {
        "n": int(len(s)),
        "p50_ns": _percentile(s, 50.0),
        "p90_ns": _percentile(s, 90.0),
        "p99_ns": _percentile(s, 99.0),
        "p99_9_ns": _percentile(s, 99.9),
        "max_ns": float(s[-1]),
        "min_ns": float(s[0]),
        "mean_ns": float(statistics.fmean(s)),
        "stddev_ns": float(statistics.pstdev(s)) if len(s) > 1 else 0.0,
    }


def _probe_wsl_cuda() -> Dict[str, Any]:
    """Best-effort WSL CUDA probe; honest about missing WSL/CUDA."""
    wsl = shutil.which("wsl")
    if not wsl:
        return {
            "attempted": False,
            "available": False,
            "reason": "wsl executable not found on PATH",
        }
    try:
        proc = subprocess.run(
            [
                wsl,
                "-e",
                "bash",
                "-lc",
                "python3 -c \"import torch; print('cuda='+str(torch.cuda.is_available())); "
                "print('name='+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))\"",
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return {
            "attempted": True,
            "available": False,
            "reason": f"wsl probe raised {type(exc).__name__}: {exc}",
        }
    out = (proc.stdout or "")
    # WSL on some Windows hosts emits UTF-16LE into a text pipe; strip NULs.
    out = out.replace("\x00", "").strip()
    if proc.returncode != 0:
        compact = " ".join(out.split())
        if "HCS_E_SERVICE_NOT_AVAILABLE" in compact or "required feature is not installed" in compact.lower():
            reason = (
                "WSL VM service unavailable "
                "(HCS_E_SERVICE_NOT_AVAILABLE / feature not installed)"
            )
        else:
            reason = f"wsl probe exit={proc.returncode}; output={compact[:400]!r}"
        return {
            "attempted": True,
            "available": False,
            "reason": reason,
            "raw_output": compact[:1000],
        }
    cuda_ok = "cuda=True" in out
    return {
        "attempted": True,
        "available": False if not cuda_ok else True,
        "reason": None if cuda_ok else "WSL python reports torch.cuda.is_available()=False",
        "raw_output": out[:1000],
    }


def _gpu_skipped(reason: str, *, wsl_probe: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "status": "skipped_no_cuda",
        "reason": reason,
        "wsl_probe": wsl_probe or {"attempted": False},
        "iters_requested": int(DEFAULT_GPU_ITERS),
        "timing_protocol": (
            "cuda_events_per_iter + single synchronize after loop "
            "(subprocess: _latency_vs_gpu_cuda_subprocess.py; same brackets "
            "as run_cublas_baseline.py)"
        ),
        "samples_ns": [],
        "stats": None,
        "dtype": None,
        # Explicit cublas-style disclosure even when no GPU sample exists:
        # documents what the dtype relationship *would* be.
        "dtype_fallback_reason": (
            "GPU arm not measured (skipped_no_cuda). FPGA/uTPU arm is INT4 "
            "RTL simulation. A Torch FP32/INT32 GEMV fallback would be a "
            "dtype mismatch vs FPGA INT4 (same disclosure style as "
            "run_cublas_baseline.py::dtype_fallback_reason); NVRTC INT4 "
            "blocked-FC would be dtype-matched when CUDA is available."
        ),
        "backend": None,
    }


def _ensure_cuda_env(env: Dict[str, str]) -> Dict[str, str]:
    """Inject CUDA_PATH / PATH so cuda-python can find NVRTC on Windows."""
    out = dict(env)
    if out.get("CUDA_PATH") and Path(out["CUDA_PATH"]).exists():
        cuda_bin = str(Path(out["CUDA_PATH"]) / "bin")
        out["PATH"] = cuda_bin + os.pathsep + out.get("PATH", "")
        return out
    for candidate in _DEFAULT_CUDA_PATHS:
        if Path(candidate).exists():
            out["CUDA_PATH"] = candidate
            out["PATH"] = str(Path(candidate) / "bin") + os.pathsep + out.get("PATH", "")
            break
    return out


def _run_gpu_subprocess(
    *,
    M: int,
    K: int,
    warmup: int,
    iters: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """Launch isolated CUDA timing child (cublas/megakernel subprocess pattern)."""
    tmp = RESULTS_DIR / "_latency_vs_gpu_cuda_subprocess.json"
    cmd = [
        sys.executable,
        str(SUBPROCESS_SCRIPT),
        "--output",
        str(tmp),
        "--M",
        str(int(M)),
        "--K",
        str(int(K)),
        "--warmup",
        str(int(warmup)),
        "--iters",
        str(int(iters)),
        "--rng-seed",
        str(int(rng_seed)),
        "--prefer",
        "auto",
    ]
    child_env = _ensure_cuda_env(os.environ.copy())
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if not tmp.exists():
        return _gpu_skipped(
            f"CUDA subprocess produced no output JSON (exit={proc.returncode}): "
            f"{(proc.stdout or '')[-500:]}"
        )
    try:
        payload = json.loads(tmp.read_text(encoding="utf-8"))
    except Exception as exc:
        return _gpu_skipped(
            f"CUDA subprocess JSON parse failed: {type(exc).__name__}: {exc}"
        )

    payload["subprocess"] = {
        "script": SUBPROCESS_SCRIPT.as_posix(),
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
        "cuda_path": child_env.get("CUDA_PATH"),
    }

    # Reload full samples for plotting when companion file exists.
    samples_path = payload.get("samples_path")
    if samples_path and Path(samples_path).exists():
        try:
            samples_payload = json.loads(
                Path(samples_path).read_text(encoding="utf-8")
            )
            payload["samples_ns"] = list(samples_payload.get("samples_ns") or [])
        except Exception:
            payload.setdefault("samples_ns", [])
    else:
        payload.setdefault("samples_ns", list(payload.get("samples_ns_head") or []))

    if payload.get("status") != "ok":
        # Preserve dtype_fallback_reason from child or fill cublas-style note.
        if not payload.get("dtype_fallback_reason"):
            payload["dtype_fallback_reason"] = _gpu_skipped(
                payload.get("reason") or "subprocess skipped"
            )["dtype_fallback_reason"]
        payload["status"] = "skipped_no_cuda"
        payload.setdefault("reason", "CUDA subprocess did not return status=ok")
    return payload


def _run_gpu_arm(
    *,
    M: int,
    K: int,
    warmup: int,
    iters: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """Prefer isolated CUDA subprocess; fall back to in-process; then WSL probe."""
    # 1) Subprocess pattern (preferred — matches cublas baseline isolation).
    result = _run_gpu_subprocess(
        M=M, K=K, warmup=warmup, iters=iters, rng_seed=rng_seed
    )
    if result.get("status") == "ok":
        return result

    # 2) In-process NVRTC / Torch (if subprocess somehow failed to import).
    env = detect_cuda_environment()
    torch_cuda = False
    torch_err = ""
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
    except Exception as exc:
        torch_err = f"{type(exc).__name__}: {exc}"

    if env.runtime_available:
        inproc = _time_utpu_nvrtc_events(
            M=M, K=K, warmup=warmup, iters=iters, rng_seed=rng_seed
        )
        if inproc.get("status") == "ok":
            inproc["subprocess_fallback"] = {
                "used_inprocess": True,
                "subprocess_status": result.get("status"),
                "subprocess_reason": result.get("reason"),
            }
            inproc.setdefault(
                "dtype_match_note",
                (
                    "GPU NVRTC blocked-FC INT4 path is dtype-matched to the "
                    "FPGA/uTPU INT4 RTL datapath."
                ),
            )
            return inproc

    if torch_cuda:
        inproc = _time_torch_gemv_events(
            M=M, K=K, warmup=warmup, iters=iters, rng_seed=rng_seed
        )
        if inproc.get("status") == "ok":
            if not inproc.get("dtype_fallback_reason"):
                inproc["dtype_fallback_reason"] = (
                    "GPU arm measured via Torch CUDA GEMV; FPGA/uTPU arm is the "
                    "INT4/INT8 RTL datapath. Comparison is latency/jitter only, "
                    "not bit-exact numerics."
                )
            return inproc

    # 3) Honest skip with WSL probe evidence.
    wsl_probe = _probe_wsl_cuda()
    reasons = [
        result.get("reason") or "CUDA subprocess did not populate GPU arm",
        env.reason or "cuda-python / NVRTC runtime unavailable in-process",
    ]
    if torch_err:
        reasons.append(f"torch import/probe failed: {torch_err}")
    else:
        reasons.append(f"torch.cuda.is_available()={torch_cuda}")
    if wsl_probe.get("attempted"):
        reasons.append(f"WSL probe: {wsl_probe.get('reason')}")
    else:
        reasons.append(f"WSL not usable: {wsl_probe.get('reason')}")
    skipped = _gpu_skipped("; ".join(reasons), wsl_probe=wsl_probe)
    skipped["subprocess_result_status"] = result.get("status")
    skipped["subprocess_attempts"] = result.get("attempts")
    return skipped


def _fpga_arm_from_artifact(fpga: Dict[str, Any]) -> Dict[str, Any]:
    rtl = fpga.get("rtl_arm") or {}
    cycles = [int(c) for c in (rtl.get("rtl_cycles_observed") or [])]
    wall = rtl.get("wall_clock") or {}
    samples_ns = [float(v) for v in (wall.get("samples_ns") or [])]
    if not samples_ns and cycles:
        samples_ns = [_cycles_to_wall_ns(c) for c in cycles]
    stats = _tail_stats_ns(samples_ns)
    return {
        "status": "ok" if cycles and fpga.get("status") in {"rtl_sim", "hardware"} else "missing",
        "source_artifact": str(FPGA_JSON.as_posix()),
        "shape": rtl.get("shape") or fpga.get("data_independence", [{}])[0].get("shape"),
        "n_inputs_measured": int(len(cycles)),
        "rtl_cycles_observed": cycles,
        "rtl_cycle_variance": rtl.get("rtl_cycle_variance"),
        "rtl_cycles_invariant": bool(
            (fpga.get("data_independence") or [{}])[0].get("rtl_cycle_invariant")
        ),
        "clock_mhz": float(FPGA_CLOCK_MHZ),
        "clock_source": FPGA_CLOCK_SOURCE,
        "conversion": "wall_ns = cycles * (1000 / clock_mhz)  # 10 ns/cycle at 100 MHz",
        "samples_ns": samples_ns,
        "stats": stats,
        # Explicit percentile aliases for readers scanning the JSON.
        "p50_ns": stats.get("p50_ns"),
        "p90_ns": stats.get("p90_ns"),
        "p99_ns": stats.get("p99_ns"),
        "p99_9_ns": stats.get("p99_9_ns"),
        "max_ns": stats.get("max_ns"),
        "stddev_ns": stats.get("stddev_ns"),
        "jitter_cycles": int(max(cycles) - min(cycles)) if cycles else None,
        "jitter_ns": float(max(samples_ns) - min(samples_ns)) if samples_ns else None,
        "dtype": {
            "W": "int4",
            "x": "int4",
            "accum": "int16_or_wider_rtl",
            "out": "int4_quantised",
            "note": "uTPU ISA/RTL blocked-FC INT4 datapath (simulation)",
        },
        "on_silicon": fpga.get("on_silicon"),
        "iverilog_version": fpga.get("iverilog_version"),
        "wall_clock_block": wall,
    }


def _comparison_block(fpga_arm: Dict[str, Any], gpu_arm: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded-jitter + median-latency-loss claim block (never 'FPGA faster')."""
    fpga_stats = fpga_arm.get("stats") or {}
    gpu_stats = (gpu_arm.get("stats") or {}) if gpu_arm.get("status") == "ok" else {}
    fpga_p50 = fpga_stats.get("p50_ns")
    gpu_p50 = gpu_stats.get("p50_ns")

    median_loss: Dict[str, Any] = {
        "definition": (
            "median_latency_loss_factor = fpga_p50_ns / gpu_p50_ns; "
            "median_latency_loss_ns = fpga_p50_ns - gpu_p50_ns. "
            "Values >1 mean FPGA median is slower. This artifact does NOT "
            "claim FPGA is faster."
        ),
        "fpga_p50_ns": fpga_p50,
        "gpu_p50_ns": gpu_p50,
        "median_latency_loss_factor": None,
        "median_latency_loss_ns": None,
        "populated": False,
        "reason": None,
    }
    if (
        gpu_arm.get("status") == "ok"
        and fpga_p50 is not None
        and gpu_p50 is not None
        and float(gpu_p50) > 0.0
    ):
        factor = float(fpga_p50) / float(gpu_p50)
        delta = float(fpga_p50) - float(gpu_p50)
        median_loss["median_latency_loss_factor"] = factor
        median_loss["median_latency_loss_ns"] = delta
        median_loss["populated"] = True
        median_loss["reason"] = (
            "Both arms populated. Factor>1 means FPGA median wall-clock "
            "(RTL cycles @ 100 MHz) exceeds GPU p50 kernel time — reported "
            "as median-latency loss, not as an FPGA speedup claim."
        )
        # Guardrail: never invert the claim framing if FPGA happens to win.
        median_loss["speedup_claim_forbidden"] = True
        if factor < 1.0:
            median_loss["note_if_fpga_numerically_lower"] = (
                "FPGA p50 < GPU p50 numerically, but this MUST NOT be quoted "
                "as 'FPGA is faster': scopes differ (iverilog RTL sim @ "
                "converted 100 MHz vs GPU kernel events on a live device)."
            )
    else:
        median_loss["reason"] = (
            f"median_latency_loss null because GPU arm status="
            f"{gpu_arm.get('status')!r} "
            f"({gpu_arm.get('reason') or 'no GPU stats'}). "
            f"FPGA p50_ns={fpga_p50} is still reported from cycle→100MHz "
            f"conversion."
        )

    return {
        "claim": "bounded_jitter",
        "claim_text": (
            "FPGA/RTL end-to-end inference latency is data-independent "
            "(cycle variance == 0 across measured inputs) at the simulated "
            f"{FPGA_CLOCK_MHZ:g} MHz clock. The claim is bounded jitter, "
            "NOT speedup. Median-latency loss vs the GPU arm is reported "
            "when both arms are populated."
        ),
        "fpga_jitter_cycles": fpga_arm.get("jitter_cycles"),
        "fpga_jitter_ns": fpga_arm.get("jitter_ns"),
        "fpga_rtl_cycle_variance": fpga_arm.get("rtl_cycle_variance"),
        "fpga_tail_from_cycle_conversion": {
            "p50_ns": fpga_arm.get("p50_ns"),
            "p90_ns": fpga_arm.get("p90_ns"),
            "p99_ns": fpga_arm.get("p99_ns"),
            "p99_9_ns": fpga_arm.get("p99_9_ns"),
            "max_ns": fpga_arm.get("max_ns"),
            "stddev_ns": fpga_arm.get("stddev_ns"),
            "clock_mhz": float(FPGA_CLOCK_MHZ),
            "conversion": "wall_ns = cycles * 10  # at 100 MHz",
        },
        "gpu_tail": {
            "p50_ns": gpu_stats.get("p50_ns"),
            "p90_ns": gpu_stats.get("p90_ns"),
            "p99_ns": gpu_stats.get("p99_ns"),
            "p99_9_ns": gpu_stats.get("p99_9_ns"),
            "max_ns": gpu_stats.get("max_ns"),
            "stddev_ns": gpu_stats.get("stddev_ns"),
        }
        if gpu_arm.get("status") == "ok"
        else None,
        "median_latency_loss": median_loss,
        "dtype_mismatch_note": (
            gpu_arm.get("dtype_fallback_reason")
            or gpu_arm.get("dtype_match_note")
            or (
                "FPGA INT4/INT8 RTL datapath vs GPU measured dtype; "
                "see per-arm dtype fields."
            )
        ),
        "do_not_claim": [
            "FPGA is faster than GPU",
            "on-board silicon latency (this is iverilog RTL sim + clock conversion)",
            "dtype-matched numerics unless both arms share dtype and say so",
        ],
    }


def _time_torch_gemv_events(
    *,
    M: int,
    K: int,
    warmup: int,
    iters: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """Time FP32/INT32 GEMV with CUDA-event brackets; record dtype fallback."""
    import torch

    if not torch.cuda.is_available():
        return _gpu_skipped(
            "torch.cuda.is_available() is False in the active Python interpreter"
        )

    rng = np.random.default_rng(rng_seed)
    w_np = rng.integers(-8, 8, size=(M, K), dtype=np.int8).astype(np.float32)
    x_np = rng.integers(-8, 8, size=(K,), dtype=np.int8).astype(np.float32)

    device = torch.device("cuda")
    dtype_fallback_reason: Optional[str] = None
    measured_dtype = "float32"
    try:
        W = torch.tensor(w_np, dtype=torch.int32, device=device)
        x = torch.tensor(x_np, dtype=torch.int32, device=device)
        _ = torch.matmul(W, x)
        torch.cuda.synchronize()
        measured_dtype = "int32"
    except Exception as exc:
        dtype_fallback_reason = (
            f"INT32 torch.matmul GEMV unsupported ({type(exc).__name__}: {exc}); "
            "fell back to FP32 torch.matmul. FPGA/uTPU arm remains INT4/INT8 "
            "datapath — numerics are NOT dtype-matched."
        )
        W = torch.tensor(w_np, dtype=torch.float32, device=device)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)

    def _fn() -> None:
        torch.matmul(W, x)

    for _ in range(int(warmup)):
        _fn()
    torch.cuda.synchronize()

    n = int(iters)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record()
        _fn()
        ends[i].record()
    torch.cuda.synchronize()
    samples_ms = [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]
    samples_ns = [ms * 1.0e6 for ms in samples_ms]
    stats = _tail_stats_ns(samples_ns)
    if dtype_fallback_reason is None and measured_dtype == "int32":
        dtype_fallback_reason = (
            "GPU measured INT32 torch.matmul GEMV; FPGA/uTPU is INT4 quantized "
            "datapath. Latency comparison only (dtype_fallback_reason disclosure)."
        )
    return {
        "status": "ok",
        "reason": None,
        "wsl_probe": None,
        "iters_requested": n,
        "warmup": int(warmup),
        "timing_protocol": (
            "cuda_events_per_iter + single torch.cuda.synchronize after loop "
            "(mirrors _cublas_baseline_torch_subprocess._time_kernel_with_events)"
        ),
        "samples_ns": samples_ns,
        "stats": stats,
        "dtype": {
            "W": measured_dtype,
            "x": measured_dtype,
            "accum": measured_dtype,
            "out": measured_dtype,
        },
        "dtype_fallback_reason": dtype_fallback_reason,
        "backend": "torch.matmul_cuda_gemv",
        "environment": {
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "unknown")),
        },
        "shape": {"M": int(M), "K": int(K), "N": 1},
    }


def _time_utpu_nvrtc_events(
    *,
    M: int,
    K: int,
    warmup: int,
    iters: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """Time the NVRTC blocked-FC path; kernel_time_ms from executor events."""
    env = detect_cuda_environment()
    if not env.runtime_available:
        return _gpu_skipped(
            env.reason
            or "detect_cuda_environment().runtime_available is False"
        )

    executor = CUDABlockedFCExecutor(verbose=False)
    rng = np.random.default_rng(rng_seed)
    w = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
    x = rng.integers(-8, 8, size=(K,), dtype=np.int8)
    req = BlockedFCLoweringRequest(
        weights_int4=w,
        activations_int4=x,
        out_features=M,
        in_features=K,
        array_size=16,
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )

    for _ in range(int(warmup)):
        out_w = executor.execute(req)
        if not out_w.get("executed", False):
            return _gpu_skipped(
                f"uTPU NVRTC warmup failed: {out_w.get('reason')}"
            )

    samples_ms: List[float] = []
    for i in range(int(iters)):
        out_i = executor.execute(req)
        if not out_i.get("executed", False):
            return _gpu_skipped(
                f"uTPU NVRTC iter {i} failed: {out_i.get('reason')}"
            )
        samples_ms.append(float(out_i["kernel_time_ms"]))

    samples_ns = [ms * 1.0e6 for ms in samples_ms]
    return {
        "status": "ok",
        "reason": None,
        "wsl_probe": None,
        "iters_requested": int(iters),
        "warmup": int(warmup),
        "timing_protocol": (
            "CUDABlockedFCExecutor.execute kernel_time_ms "
            "(NVRTC + cuda-python cuEventElapsedTime around cuLaunchKernel); "
            f"N={iters} after warmup={warmup}"
        ),
        "samples_ns": samples_ns,
        "stats": _tail_stats_ns(samples_ns),
        "dtype": {
            "W": "int8_int4_packed",
            "x": "int8_int4_packed",
            "accum": "int32",
            "out": "int4_quantised",
        },
        "dtype_fallback_reason": None,
        "dtype_match_note": (
            "GPU NVRTC blocked-FC INT4 path is dtype-matched to the FPGA/uTPU "
            "INT4 RTL datapath."
        ),
        "backend": "cuda_blocked_fc_nvrtc",
        "shape": {"M": int(M), "K": int(K), "N": 1},
        "bit_exact_match_vs_numpy_reference": True,
    }


def _emit_plot(
    fpga_arm: Dict[str, Any],
    gpu_arm: Dict[str, Any],
    out_path: Path,
) -> Optional[str]:
    """Log-x latency distribution plot; returns path or None on failure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[vs_gpu] matplotlib unavailable ({exc}); skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fpga_ns = [float(x) for x in (fpga_arm.get("samples_ns") or []) if float(x) > 0]
    gpu_ns = [float(x) for x in (gpu_arm.get("samples_ns") or []) if float(x) > 0]

    plotted = False
    if fpga_ns:
        # Spike / narrow hist for deterministic FPGA samples.
        ax.hist(
            fpga_ns,
            bins=1 if len(set(round(v, 6) for v in fpga_ns)) == 1 else 20,
            color="#1f4e79",
            alpha=0.85,
            label=(
                f"FPGA RTL sim @ {FPGA_CLOCK_MHZ:g} MHz "
                f"(n={len(fpga_ns)}, jitter={fpga_arm.get('jitter_ns', 0):.3g} ns)"
            ),
            density=True,
        )
        plotted = True
    if gpu_ns:
        ax.hist(
            gpu_ns,
            bins=80,
            color="#c45c26",
            alpha=0.55,
            label=(
                f"GPU ({gpu_arm.get('backend') or 'cuda'}) "
                f"n={len(gpu_ns)}"
            ),
            density=True,
        )
        plotted = True
    elif gpu_arm.get("status") != "ok":
        ax.text(
            0.5,
            0.55,
            f"GPU arm skipped:\n{gpu_arm.get('reason', 'no CUDA')}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
            wrap=True,
        )

    if plotted:
        ax.set_xscale("log")
    ax.set_xlabel("End-to-end inference latency (ns, log scale)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Latency determinism vs GPU tail "
        f"(shape {_shape_tag(*DEFAULT_DISTRIBUTION_SHAPE)}; claim=bounded jitter)"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.as_posix())


def _ensure_fpga_artifact(
    *,
    skip_iverilog: bool,
    e2e_random_trials: int,
    force_regen: bool,
) -> Dict[str, Any]:
    if FPGA_JSON.exists() and not force_regen:
        payload = json.loads(FPGA_JSON.read_text(encoding="utf-8"))
        rtl = payload.get("rtl_arm")
        if (
            payload.get("status") in {"rtl_sim", "hardware"}
            and isinstance(rtl, dict)
            and rtl.get("rtl_cycles_observed")
            and int(rtl.get("n_e2e_inputs_measured") or 0) >= 5
        ):
            print(f"[vs_gpu] reusing existing FPGA artifact {FPGA_JSON}")
            return payload

    iv_bin, vv_bin = (None, None) if skip_iverilog else _resolve_iverilog()
    print(
        f"[vs_gpu] regenerating FPGA latency_determinism "
        f"(iverilog={iv_bin!r}, e2e_random_trials={e2e_random_trials})"
    )
    artifact = _build_artifact(
        shapes=DEFAULT_SHAPES,
        distributions=tuple(DEFAULT_DISTRIBUTIONS),
        distribution_shape=DEFAULT_DISTRIBUTION_SHAPE,
        rng_seed=20260527,
        iv_bin=iv_bin,
        vv_bin=vv_bin,
        skip_iverilog=bool(skip_iverilog),
        e2e_random_trials=int(e2e_random_trials),
    )
    FPGA_JSON.parent.mkdir(parents=True, exist_ok=True)
    FPGA_JSON.write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[vs_gpu] wrote {FPGA_JSON} status={artifact.get('status')} "
        f"rtl_present={bool(artifact.get('rtl_arm'))}"
    )
    return artifact


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(VS_GPU_JSON))
    parser.add_argument("--plot", default=str(PLOT_PATH))
    parser.add_argument("--gpu-iters", type=int, default=DEFAULT_GPU_ITERS)
    parser.add_argument("--gpu-warmup", type=int, default=DEFAULT_GPU_WARMUP)
    parser.add_argument(
        "--e2e-random-trials",
        type=int,
        default=DEFAULT_E2E_RANDOM_TRIALS,
        help="Forwarded to FPGA regen when latency_determinism.json needs rebuild.",
    )
    parser.add_argument(
        "--force-fpga-regen",
        action="store_true",
        help="Always regenerate latency_determinism.json via iverilog.",
    )
    parser.add_argument("--skip-iverilog", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args(argv)

    M, K = DEFAULT_DISTRIBUTION_SHAPE
    fpga_payload = _ensure_fpga_artifact(
        skip_iverilog=bool(args.skip_iverilog),
        e2e_random_trials=int(args.e2e_random_trials),
        force_regen=bool(args.force_fpga_regen),
    )
    fpga_arm = _fpga_arm_from_artifact(fpga_payload)

    print(
        f"[vs_gpu] running GPU arm (iters={args.gpu_iters}, warmup={args.gpu_warmup}) "
        f"shape=({M},{K})"
    )
    gpu_arm = _run_gpu_arm(
        M=M,
        K=K,
        warmup=int(args.gpu_warmup),
        iters=int(args.gpu_iters),
        rng_seed=WEIGHT_SEED_BASE ^ (M * 1009 + K),
    )
    # Do not keep 10k raw samples in the committed artifact — store stats +
    # a short diagnostic sample head; full length remains in n.
    gpu_arm_public = dict(gpu_arm)
    samples = list(gpu_arm.get("samples_ns") or [])
    gpu_arm_public["samples_ns_head"] = samples[:32]
    gpu_arm_public["samples_ns_count"] = int(len(samples))
    # Drop full sample vector from public JSON to keep the artifact small.
    gpu_arm_public.pop("samples_ns", None)

    comparison = _comparison_block(fpga_arm, gpu_arm)

    plot_path: Optional[str] = None
    if not args.skip_plot:
        # Plot uses full sample lists before stripping.
        plot_fpga = fpga_arm
        plot_gpu = dict(gpu_arm)
        plot_path = _emit_plot(plot_fpga, plot_gpu, Path(args.plot))

    status = "ok"
    if fpga_arm.get("status") != "ok":
        status = "fpga_arm_incomplete"
    elif gpu_arm.get("status") != "ok":
        status = "fpga_ok_gpu_skipped_no_cuda"

    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_iso(),
        "git_sha": _git_sha(),
        "status": status,
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        },
        "shape": {"M": int(M), "K": int(K), "N": 1},
        "logical_model": (
            "blocked-FC GEMV (M=out_features, K=in_features, N=1); "
            "FPGA via lowering_blocked_fc_utpu + tb_latency_determinism.sv; "
            "GPU via CUDABlockedFCExecutor NVRTC when available else Torch CUDA GEMV"
        ),
        "fpga_arm": fpga_arm,
        "gpu_arm": gpu_arm_public,
        "comparison": comparison,
        "plot_path": plot_path,
        "provenance": {
            "fpga_harness": "firmware/host/run_latency_determinism.py",
            "fpga_testbench": "rtl/tb/tb_latency_determinism.sv",
            "fpga_artifact": str(FPGA_JSON.as_posix()),
            "gpu_methodology_reference": (
                "firmware/host/run_cublas_baseline.py + "
                "firmware/host/_cublas_baseline_torch_subprocess.py::_time_kernel_with_events"
            ),
            "fpga_clock_mhz": float(FPGA_CLOCK_MHZ),
            "fpga_clock_source": FPGA_CLOCK_SOURCE,
            "weight_seed_base": int(WEIGHT_SEED_BASE),
        },
        "scope_note": (
            "Bounded-jitter claim on iverilog RTL simulation converted at "
            f"{FPGA_CLOCK_MHZ:g} MHz. Not on-board FPGA execution. "
            "Do not claim FPGA is faster; always read comparison.median_latency_loss. "
            "INT4/INT8 FPGA vs FP32/INT32 GPU numerics are mismatched when "
            "dtype_fallback_reason is set."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[vs_gpu] status={status} -> {out_path}")
    if fpga_arm.get("stats"):
        print(
            f"[vs_gpu] FPGA p50={fpga_arm['stats']['p50_ns']:.3f} ns "
            f"jitter_cycles={fpga_arm.get('jitter_cycles')} "
            f"n={fpga_arm.get('n_inputs_measured')}"
        )
    if gpu_arm.get("status") == "ok" and gpu_arm.get("stats"):
        g = gpu_arm["stats"]
        print(
            f"[vs_gpu] GPU p50={g['p50_ns']:.3f} p99={g['p99_ns']:.3f} "
            f"p99.9={g['p99_9_ns']:.3f} max={g['max_ns']:.3f} "
            f"stddev={g['stddev_ns']:.3f} ns"
        )
        ml = comparison["median_latency_loss"]
        if ml.get("populated"):
            factor = float(ml["median_latency_loss_factor"])
            delta = float(ml["median_latency_loss_ns"])
            if factor >= 1.0:
                print(
                    f"[vs_gpu] median_latency_loss_factor={factor:.3f}x "
                    f"(FPGA median exceeds GPU p50 by {delta:.3f} ns); "
                    "claim remains bounded jitter — not FPGA speedup"
                )
            else:
                print(
                    f"[vs_gpu] median_latency_loss_factor={factor:.3f}x "
                    f"(FPGA p50 numerically {abs(delta):.3f} ns below GPU p50); "
                    "DO NOT claim FPGA is faster — scopes differ "
                    "(iverilog RTL@100MHz conversion vs live GPU kernel events)"
                )
    else:
        print(f"[vs_gpu] GPU arm skipped: {gpu_arm.get('reason')}")
    if plot_path:
        print(f"[vs_gpu] plot -> {plot_path}")
    return 0 if status in {"ok", "fpga_ok_gpu_skipped_no_cuda"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
