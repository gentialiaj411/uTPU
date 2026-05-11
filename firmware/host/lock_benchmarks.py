import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "benchmarks"


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, cwd=REPO_ROOT)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: List[float]) -> Dict[str, float]:
    sorted_vals = sorted(values)
    return {
        "min": float(sorted_vals[0]),
        "median": float(statistics.median(sorted_vals)),
        "max": float(sorted_vals[-1]),
    }


def _extract_float(d: Dict[str, Any], path: List[str]) -> float:
    cur: Any = d
    for k in path:
        cur = cur[k]
    return float(cur)


def _ensure_dirs(runs: int) -> None:
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    for i in range(1, runs + 1):
        (BENCH_ROOT / f"run_{i:02d}").mkdir(parents=True, exist_ok=True)


def _copy_report_to_run(report: Path, run_dir: Path, dst_name: str) -> None:
    if not report.exists():
        raise FileNotFoundError(f"Missing report: {report}")
    (run_dir / dst_name).write_text(report.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and lock uTPU benchmark suite.")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    _ensure_dirs(args.runs)

    # Canonical shapes for CUDA blocked FC benchmarking.
    shapes = [
        ("small", 10, 9),
        ("medium", 9, 196),
        ("representative_mlp", 64, 256),
    ]

    for i in range(1, args.runs + 1):
        run_dir = BENCH_ROOT / f"run_{i:02d}"

        _run([
            sys.executable,
            "firmware/host/block_runtime_analysis.py",
            "--num-samples",
            "100",
            "--output-json",
            "build/reports/block_runtime_metrics.json",
            "--output-md",
            "build/reports/block_runtime_report.md",
        ])
        _copy_report_to_run(
            REPO_ROOT / "build/reports/block_runtime_metrics.json",
            run_dir,
            "block_runtime_metrics.json",
        )

        for shape_name, m, k in shapes:
            _run([
                sys.executable,
                "firmware/host/benchmark_cuda_blocked_fc.py",
                "--m",
                str(m),
                "--k",
                str(k),
                "--iters",
                "40",
                "--warmup",
                "8",
                "--output-json",
                "build/reports/cuda_blocked_fc_benchmark.json",
            ])
            _copy_report_to_run(
                REPO_ROOT / "build/reports/cuda_blocked_fc_benchmark.json",
                run_dir,
                f"cuda_blocked_fc_{shape_name}.json",
            )

        _run([sys.executable, "firmware/host/test_fused_full_inference_program.py"])
        _copy_report_to_run(
            REPO_ROOT / "build/reports/fused_full_inference_metrics.json",
            run_dir,
            "fused_full_inference_metrics.json",
        )

        _run([
            sys.executable,
            "firmware/host/run_rtl_fused_sim.py",
            "--output-json",
            "build/reports/rtl_fused_sim_metrics.json",
            "--output-md",
            "build/reports/rtl_fused_sim_report.md",
        ])
        _copy_report_to_run(
            REPO_ROOT / "build/reports/rtl_fused_sim_metrics.json",
            run_dir,
            "rtl_fused_sim_metrics.json",
        )

    # Summaries from raw outputs
    block_acc = []
    block_diff = []

    cuda_kernel = {"small": [], "medium": [], "representative_mlp": []}
    cuda_transfer = {"small": [], "medium": [], "representative_mlp": []}
    cuda_vs_cublas = {"small": [], "medium": [], "representative_mlp": []}

    fused_bram_words = []
    rtl_cycles = []
    rtl_pass = []

    for i in range(1, args.runs + 1):
        run_dir = BENCH_ROOT / f"run_{i:02d}"

        block = _read_json(run_dir / "block_runtime_metrics.json")
        block_acc.append(_extract_float(block, ["equivalence", "array_block_accuracy_pct"]))
        block_diff.append(_extract_float(block, ["equivalence", "max_abs_logit_diff"]))

        for shape in ["small", "medium", "representative_mlp"]:
            cu = _read_json(run_dir / f"cuda_blocked_fc_{shape}.json")
            cuda_kernel[shape].append(_extract_float(cu, ["timing_ms", "kernel_avg"]))
            cuda_transfer[shape].append(_extract_float(cu, ["transfer_overhead_pct_of_e2e"]))
            kv = cu.get("kernel_speed_vs_cublas_pct")
            cuda_vs_cublas[shape].append(float(kv) if kv is not None else float("nan"))

        fused = _read_json(run_dir / "fused_full_inference_metrics.json")
        fused_bram_words.append(_extract_float(fused, ["new_fused_full_inference_words"]))

        rtl = _read_json(run_dir / "rtl_fused_sim_metrics.json")
        rtl_pass.append(bool(rtl.get("rtl_sim_passed", False)))
        cyc = rtl.get("total_cycles")
        rtl_cycles.append(float(cyc) if cyc is not None else -1.0)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": args.runs,
        "block_runtime_correctness": {
            "array_block_accuracy_pct": _stats(block_acc),
            "max_abs_logit_diff": _stats(block_diff),
        },
        "cuda_blocked_fc": {
            shape: {
                "kernel_avg_ms": _stats(cuda_kernel[shape]),
                "transfer_overhead_pct": _stats(cuda_transfer[shape]),
                "kernel_vs_cublas_pct": _stats([x for x in cuda_vs_cublas[shape] if x == x]),
            }
            for shape in ["small", "medium", "representative_mlp"]
        },
        "fused_inference_program_bram_words": _stats(fused_bram_words),
        "rtl_fused_sim": {
            "all_runs_passed": all(rtl_pass),
            "pass_count": sum(1 for x in rtl_pass if x),
            "cycle_counts": _stats(rtl_cycles),
        },
    }

    (BENCH_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
