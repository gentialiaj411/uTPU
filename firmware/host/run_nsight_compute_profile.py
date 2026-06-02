"""Task 1 v1.1 — Nsight Compute occupancy / bottleneck profile harness.

Wraps ``_megakernel_cuda_subprocess.run_one_workload`` for the
``fused_region`` arm of a single locked workload, launches it under
``ncu --set full --target-processes all --export <path>``, then exports
the resulting ``.ncu-rep`` to CSV via ``ncu --import ... --csv`` and
distills the per-kernel metrics into ``bench/results/nsight_compute_profile.json``.

Distillation philosophy:
    The full ``--set full`` output is hundreds of metrics per kernel. The
    artifact records the **3 metrics that actually answer the
    "occupancy-bound vs memory-bound vs compute-bound" question** for a
    phone-screen-friendly summary:

      * ``sm__throughput.avg.pct_of_peak_sustained_elapsed``        — SM
                                                                       compute throughput as a % of theoretical peak.
      * ``gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`` —
                                                                       memory subsystem throughput as a % of peak.
      * ``sm__warps_active.avg.pct_of_peak_sustained_active``       — achieved
                                                                       occupancy.

    Bound classification rule (locked):
      * ``compute_bound``    iff sm_throughput  >= 0.80 and mem_throughput < 0.80
      * ``memory_bound``     iff mem_throughput >= 0.80 and sm_throughput < 0.80
      * ``launch_overhead_bound`` iff both throughputs < 0.40 (kernel too short)
      * ``balanced``         otherwise.

    These thresholds are documented in ``methodology.bound_classification``.

Gracefully degrades on hosts where ``ncu`` is not on PATH or CUDA is
unavailable — emits a stub artifact with
``status="nsight_compute_unavailable"`` and full ``regen_instructions``
so the schema-lock test always has a populated artifact to read.

NOT a primary resume bullet. Provides expert-interview armor for the
megakernel Bullet 2: an engineer asking "is this fused win compute-bound
or memory-bound?" gets a direct answer with measured metrics, not hand-waving.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_DIR = REPO_ROOT / "firmware" / "host"
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "nsight_compute_profile.json"

# The 4 locked workloads we profile — same names as in `run_megakernel_benchmark.WORKLOADS`.
# `ncu --set full` is HEAVY (~5-10s per kernel launch); we don't profile every
# workload, we profile the 4 that anchor the Bullet 2 claim.
LOCKED_WORKLOAD_NAMES: List[str] = [
    "linear_relu_256",
    "linear_relu_add_residual_256",
    "linear_scale_relu_add_512",
    "elementwise_relu_scale_add_4096",
]

# Bound-classification thresholds (locked in METHODOLOGY for the schema-lock test).
COMPUTE_BOUND_THRESHOLD_PCT = 80.0
MEMORY_BOUND_THRESHOLD_PCT = 80.0
LAUNCH_OVERHEAD_THRESHOLD_PCT = 40.0

# Metric column names ncu emits in `--csv` mode (exact names from
# CUDA 13.x Nsight Compute). If the user has a different ncu version that
# uses slightly different metric names, the harness records all column
# names in `methodology.metric_columns_observed` so the schema-lock test
# can distinguish "harness alignment issue" from "kernel actually didn't
# emit the metric".
METRIC_SM_THROUGHPUT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_MEM_THROUGHPUT = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_OCCUPANCY = "sm__warps_active.avg.pct_of_peak_sustained_active"


METHODOLOGY: Dict[str, Any] = {
    "ncu_command": (
        "ncu --set full --target-processes all --replay-mode kernel "
        "--launch-skip-before-match 0 --launch-count 1 --export <rep>"
    ),
    "csv_export_command": "ncu --import <rep> --csv --page raw",
    "workloads_profiled": LOCKED_WORKLOAD_NAMES,
    "arm_profiled": "fused_region",
    "metrics_extracted": [
        METRIC_SM_THROUGHPUT,
        METRIC_MEM_THROUGHPUT,
        METRIC_OCCUPANCY,
    ],
    "bound_classification": {
        "compute_bound_iff": (
            f"sm_throughput >= {COMPUTE_BOUND_THRESHOLD_PCT}% AND "
            f"mem_throughput < {MEMORY_BOUND_THRESHOLD_PCT}%"
        ),
        "memory_bound_iff": (
            f"mem_throughput >= {MEMORY_BOUND_THRESHOLD_PCT}% AND "
            f"sm_throughput < {COMPUTE_BOUND_THRESHOLD_PCT}%"
        ),
        "launch_overhead_bound_iff": (
            f"BOTH sm_throughput < {LAUNCH_OVERHEAD_THRESHOLD_PCT}% AND "
            f"mem_throughput < {LAUNCH_OVERHEAD_THRESHOLD_PCT}%"
        ),
        "balanced_iff": "otherwise",
    },
    "scope": (
        "Profiles the fused_region kernel ONLY (the kernel under the Bullet 2 "
        "headline claim); the op_by_op / cuda_graphs_op_by_op / cublas_fp32 arms "
        "are NOT profiled because their kernels are either identical to fused "
        "per-op codegen (op_by_op) or external library kernels (cublas)."
    ),
    "honest_caveats": (
        "Nsight Compute serializes kernel replay (`--replay-mode kernel`) which "
        "inflates absolute time; the ncu-reported percentages are still valid "
        "but the absolute timings differ from the megakernel benchmark's "
        "warmup+iters timings. The artifact does NOT use ncu timings as a "
        "headline number; ncu is used ONLY for bound classification."
    ),
    "stub_behavior": (
        "If ncu is not on PATH OR CUDA is unavailable OR the subprocess "
        "errors, the artifact lands with status='nsight_compute_unavailable' "
        "and full regen_instructions. No timings are fabricated under any "
        "failure mode."
    ),
}


def _ncu_on_path() -> Optional[str]:
    """Return the path to the ncu binary if found, else None."""
    candidate = shutil.which("ncu") or shutil.which("ncu.exe")
    if candidate:
        return candidate
    # Common install locations on Windows + Linux.
    for guess in [
        "/usr/local/cuda/bin/ncu",
        "/opt/nvidia/nsight-compute/2024.1/ncu",
        "/opt/nvidia/nsight-compute/2025.1/ncu",
        "C:/Program Files/NVIDIA Corporation/Nsight Compute 2024.1/ncu.exe",
        "C:/Program Files/NVIDIA Corporation/Nsight Compute 2025.1/ncu.exe",
    ]:
        if os.path.exists(guess):
            return guess
    return None


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _host_env() -> Dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }


def _build_profile_subprocess_script(workload_name: str) -> str:
    """Inline Python script that, when executed by ncu, runs ONE fused_region
    invocation of the given workload. We keep this short so ncu's
    `--launch-count 1` captures exactly one kernel launch.
    """
    return f"""
import json, sys, os
sys.path.insert(0, {str(HOST_DIR)!r})
from run_megakernel_benchmark import WORKLOADS
from _megakernel_cuda_subprocess import (
    _build_workload_graph,
    _numpy_reference_output,
    _time_fused_region,
    _import_cuda_bindings,
    _cuda_check,
)
w = next(x for x in WORKLOADS if x["name"] == {workload_name!r})
graph, ext = _build_workload_graph(w)
ref = _numpy_reference_output(graph, ext)
cuda_mod, _ = _import_cuda_bindings()
_cuda_check(cuda_mod.cuInit(0), "cuInit")
(device,) = _cuda_check(cuda_mod.cuDeviceGet(0), "cuDeviceGet")
(ctx,) = _cuda_check(cuda_mod.cuDevicePrimaryCtxRetain(device), "cuDevicePrimaryCtxRetain")
_cuda_check(cuda_mod.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
# warmup=0 + iters=1 so ncu's --launch-count 1 captures exactly one launch.
out = _time_fused_region(graph, ext, ref, warmup=0, iters=1)
sys.stdout.write(json.dumps({{"workload": {workload_name!r}, "arm_result": out}}, default=str))
cuda_mod.cuDevicePrimaryCtxRelease(device)
"""


def _run_ncu_for_workload(ncu_bin: str, workload_name: str) -> Dict[str, Any]:
    """Run ncu against a one-shot subprocess that fires the fused_region
    kernel for `workload_name`. Returns a dict with raw + distilled metrics,
    or an error description.
    """
    with tempfile.TemporaryDirectory(prefix="ncu_profile_") as tmpdir:
        rep_path = Path(tmpdir) / f"{workload_name}.ncu-rep"
        script_path = Path(tmpdir) / f"{workload_name}.py"
        script_path.write_text(_build_profile_subprocess_script(workload_name), encoding="utf-8")

        cmd = [
            ncu_bin,
            "--set", "full",
            "--target-processes", "all",
            "--replay-mode", "kernel",
            "--launch-skip-before-match", "0",
            "--launch-count", "1",
            "--export", str(rep_path),
            sys.executable, str(script_path),
        ]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT)
            )
        except subprocess.TimeoutExpired:
            return {"workload": workload_name, "status": "error", "reason": "ncu timed out after 600s"}
        if completed.returncode != 0:
            return {
                "workload": workload_name,
                "status": "error",
                "reason": (completed.stderr or completed.stdout or "").strip()[-1500:],
                "returncode": int(completed.returncode),
            }
        if not rep_path.exists():
            return {
                "workload": workload_name,
                "status": "error",
                "reason": f"ncu reported success but {rep_path} not found",
            }

        # Export the .ncu-rep to CSV so we can parse it.
        csv_cmd = [ncu_bin, "--import", str(rep_path), "--csv", "--page", "raw"]
        try:
            csv_completed = subprocess.run(
                csv_cmd, capture_output=True, text=True, timeout=300
            )
        except subprocess.TimeoutExpired:
            return {"workload": workload_name, "status": "error", "reason": "ncu --import timed out"}
        if csv_completed.returncode != 0 or not csv_completed.stdout:
            return {
                "workload": workload_name,
                "status": "error",
                "reason": f"ncu --import failed: {csv_completed.stderr[-1500:]}",
            }

        return _distill_csv(workload_name, csv_completed.stdout)


def _distill_csv(workload_name: str, csv_text: str) -> Dict[str, Any]:
    """Parse ncu --csv --page raw output and pull the three classification
    metrics. The first row is metric names (one column per metric); subsequent
    rows are per-kernel-invocation values.
    """
    reader = csv.reader(csv_text.splitlines())
    rows = list(reader)
    if not rows or len(rows) < 2:
        return {"workload": workload_name, "status": "error", "reason": "ncu CSV had no data rows"}
    header = [c.strip() for c in rows[0]]
    # ncu CSV often has metadata header rows before the column header; find
    # the first row that contains a known metric name.
    metric_columns = {METRIC_SM_THROUGHPUT, METRIC_MEM_THROUGHPUT, METRIC_OCCUPANCY}
    header_row_idx = 0
    for i, row in enumerate(rows[:10]):
        if any(col.strip() in metric_columns for col in row):
            header = [c.strip() for c in row]
            header_row_idx = i
            break
    data_rows = rows[header_row_idx + 1:]
    if not data_rows:
        return {"workload": workload_name, "status": "error", "reason": "no kernel rows below header"}

    col_idx = {name: header.index(name) if name in header else None for name in metric_columns}
    sm_pct = _safe_float(data_rows[0], col_idx[METRIC_SM_THROUGHPUT])
    mem_pct = _safe_float(data_rows[0], col_idx[METRIC_MEM_THROUGHPUT])
    occ_pct = _safe_float(data_rows[0], col_idx[METRIC_OCCUPANCY])

    bound = _classify_bound(sm_pct, mem_pct)

    return {
        "workload": workload_name,
        "status": "ok",
        "kernel_count_in_report": len(data_rows),
        "sm_throughput_pct": sm_pct,
        "memory_throughput_pct": mem_pct,
        "achieved_occupancy_pct": occ_pct,
        "bottleneck_classification": bound,
        "metric_columns_observed": header,
    }


def _safe_float(row: List[str], idx: Optional[int]) -> Optional[float]:
    if idx is None or idx >= len(row):
        return None
    raw = row[idx].strip().replace(",", "")
    if not raw or raw.lower() in ("n/a", "nan", "inf", "-inf"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _classify_bound(sm_pct: Optional[float], mem_pct: Optional[float]) -> str:
    if sm_pct is None or mem_pct is None:
        return "unclassified"
    if sm_pct < LAUNCH_OVERHEAD_THRESHOLD_PCT and mem_pct < LAUNCH_OVERHEAD_THRESHOLD_PCT:
        return "launch_overhead_bound"
    if sm_pct >= COMPUTE_BOUND_THRESHOLD_PCT and mem_pct < MEMORY_BOUND_THRESHOLD_PCT:
        return "compute_bound"
    if mem_pct >= MEMORY_BOUND_THRESHOLD_PCT and sm_pct < COMPUTE_BOUND_THRESHOLD_PCT:
        return "memory_bound"
    return "balanced"


def build_artifact(workloads: List[str]) -> Dict[str, Any]:
    artifact: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "methodology": METHODOLOGY,
        "workloads_requested": workloads,
        "environment": _host_env(),
    }

    ncu_bin = _ncu_on_path()
    if ncu_bin is None:
        artifact["status"] = "nsight_compute_unavailable"
        artifact["reason"] = (
            "ncu (Nsight Compute CLI) not on PATH and not at the standard "
            "install locations. Install via the CUDA Toolkit and re-run."
        )
        artifact["per_workload"] = [
            {"workload": w, "status": "skipped", "reason": "ncu not found"}
            for w in workloads
        ]
        artifact["aggregate"] = _empty_aggregate(workloads)
        artifact["regen_instructions"] = (
            "On WSL2 + RTX 5070 with CUDA toolkit installed: "
            "`python firmware/host/run_nsight_compute_profile.py`. "
            "Requires ncu in PATH (typically /usr/local/cuda/bin or via "
            "the Nsight Compute desktop install)."
        )
        return artifact

    artifact["ncu_bin"] = ncu_bin
    try:
        artifact["ncu_version"] = subprocess.check_output(
            [ncu_bin, "--version"], text=True, timeout=10
        ).strip().splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        artifact["ncu_version"] = f"unknown: {exc}"

    per_workload: List[Dict[str, Any]] = []
    for w in workloads:
        per_workload.append(_run_ncu_for_workload(ncu_bin, w))

    artifact["status"] = "ok" if all(r.get("status") == "ok" for r in per_workload) else "partial"
    artifact["per_workload"] = per_workload
    artifact["aggregate"] = _compute_aggregate(per_workload)
    artifact["regen_instructions"] = (
        "On WSL2 + RTX 5070: `python firmware/host/run_nsight_compute_profile.py`. "
        "Runs ncu --set full once per locked workload (~30-60s each); writes the "
        "distilled artifact and discards the raw .ncu-rep files."
    )
    return artifact


def _empty_aggregate(workloads: List[str]) -> Dict[str, Any]:
    return {
        "workloads_profiled_ok": 0,
        "workloads_requested_total": len(workloads),
        "sm_throughput_pct_median": None,
        "memory_throughput_pct_median": None,
        "achieved_occupancy_pct_median": None,
        "bottleneck_classification_counts": {},
        "primary_bottleneck_class": None,
    }


def _compute_aggregate(per_workload: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [r for r in per_workload if r.get("status") == "ok"]
    sm_vals = [r["sm_throughput_pct"] for r in ok_rows if r.get("sm_throughput_pct") is not None]
    mem_vals = [r["memory_throughput_pct"] for r in ok_rows if r.get("memory_throughput_pct") is not None]
    occ_vals = [r["achieved_occupancy_pct"] for r in ok_rows if r.get("achieved_occupancy_pct") is not None]
    classes = [r["bottleneck_classification"] for r in ok_rows if r.get("bottleneck_classification")]
    counts: Dict[str, int] = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    primary = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
    return {
        "workloads_profiled_ok": len(ok_rows),
        "workloads_requested_total": len(per_workload),
        "sm_throughput_pct_median": float(np.median(sm_vals)) if sm_vals else None,
        "memory_throughput_pct_median": float(np.median(mem_vals)) if mem_vals else None,
        "achieved_occupancy_pct_median": float(np.median(occ_vals)) if occ_vals else None,
        "bottleneck_classification_counts": counts,
        "primary_bottleneck_class": primary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_JSON))
    parser.add_argument(
        "--workload",
        action="append",
        default=None,
        help="Workload name to profile (default: all 4 locked workloads).",
    )
    parser.add_argument(
        "--skip-ncu",
        action="store_true",
        help="Emit a stub artifact with status=nsight_compute_unavailable "
        "without trying to run ncu. Useful for CI on hosts where ncu is absent.",
    )
    args = parser.parse_args()

    workloads = args.workload or LOCKED_WORKLOAD_NAMES

    if args.skip_ncu:
        artifact = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "methodology": METHODOLOGY,
            "workloads_requested": workloads,
            "environment": _host_env(),
            "status": "nsight_compute_unavailable",
            "reason": "--skip-ncu requested; stub artifact emitted for hostless CI",
            "per_workload": [
                {"workload": w, "status": "skipped", "reason": "--skip-ncu"}
                for w in workloads
            ],
            "aggregate": _empty_aggregate(workloads),
            "regen_instructions": (
                "Re-run without --skip-ncu on a WSL2 + CUDA + ncu host."
            ),
        }
    else:
        artifact = build_artifact(workloads)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[run_nsight_compute_profile] wrote {out_path} with status={artifact['status']}"
    )


if __name__ == "__main__":
    main()
