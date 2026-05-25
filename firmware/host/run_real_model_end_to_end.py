"""Benchmark ResNet-18 end-to-end through the uTPU CUDA graph compiler path."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


RTOL = 1e-3
ATOL = 1e-3


def _within_tolerance(ref: np.ndarray, out: np.ndarray) -> Tuple[bool, float, float]:
    ref = np.asarray(ref, dtype=np.float32)
    out = np.asarray(out, dtype=np.float32)
    abs_err = np.max(np.abs(ref - out))
    denom = np.maximum(np.abs(ref), 1e-12)
    rel_err = float(np.max(abs_err / denom))
    ok = bool(np.allclose(ref, out, rtol=RTOL, atol=ATOL))
    return ok, float(abs_err), rel_err


def _collect_host_info(torch_module: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
        "pytorch_version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_available": bool(torch_module.cuda.is_available()),
    }
    if info["cuda_available"]:
        info["cuda_version"] = str(getattr(torch_module.version, "cuda", "unknown"))
        info["gpu_name"] = str(torch_module.cuda.get_device_name(0))
    return info


def _run_inductor_oracle_subprocess(
    seeds: Tuple[int, ...],
    input_size: int,
    weights_path: str,
) -> Dict[int, Dict[str, Any]]:
    """Collect per-seed Inductor outputs in a fresh process before NVRTC driver context init."""
    worker = os.path.join(os.path.dirname(__file__), "inductor_oracle_subprocess.py")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        out_path = tmp.name
    try:
        cmd = [
            sys.executable,
            worker,
            "--output",
            out_path,
            "--input-size",
            str(int(input_size)),
            "--seeds",
            ",".join(str(int(s)) for s in seeds),
            "--weights",
            weights_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 and not os.path.isfile(out_path):
            reason = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
            return {
                int(s): {"status": "skipped", "reason": reason, "output": None}
                for s in seeds
            }
        with open(out_path, encoding="utf-8") as f:
            payload = json.load(f)
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass

    by_seed: Dict[int, Dict[str, Any]] = {}
    for case in payload.get("cases", []):
        seed = int(case["seed"])
        if case.get("status") == "pass":
            by_seed[seed] = {
                "status": "pass",
                "output": np.asarray(case["output"], dtype=np.float32),
                "reason": None,
            }
        else:
            by_seed[seed] = {
                "status": "skipped",
                "output": None,
                "reason": case.get("reason", "inductor subprocess failed"),
            }
    for seed in seeds:
        by_seed.setdefault(
            int(seed),
            {"status": "skipped", "output": None, "reason": "missing inductor subprocess case"},
        )
    return by_seed


def run_real_model_benchmark(
    output_json_path: str = "bench/results/real_model_end_to_end.json",
    input_size: int = 224,
    warmup: int = 2,
    iters: int = 5,
    seeds: Tuple[int, ...] = (0, 1, 42),
) -> Dict[str, Any]:
    import torch
    import torchvision.models as models

    from pytorch_compiler import compile_model

    # Avoid cuDNN on newer GPUs when the installed PyTorch build lacks matching cuDNN kernels.
    torch.backends.cudnn.enabled = False

    cuda_available = bool(torch.cuda.is_available())
    device = torch.device("cuda" if cuda_available else "cpu")
    trace_model = models.resnet18(weights=None).cpu().eval()

    weights_path = tempfile.mktemp(suffix=".pt")
    torch.save(trace_model.state_dict(), weights_path)

    trace_gen = torch.Generator(device="cpu").manual_seed(int(seeds[0]))
    trace_x = torch.randn(
        1, 3, input_size, input_size, generator=trace_gen, device="cpu", dtype=torch.float32
    )
    compiled_result = compile_model(
        trace_model,
        (trace_x,),
        target="cuda",
        apply_quant=False,
        strict=False,
    )
    if not compiled_result.ok:
        raise RuntimeError(f"compile_model failed: {compiled_result.summary()}")

    run_model = models.resnet18(weights=None).to(device).eval()
    run_model.load_state_dict(trace_model.state_dict())
    if compiled_result.runtime is not None:
        compiled_result.runtime.reference_model = run_model

    eager_by_seed: Dict[int, Any] = {}
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        x = torch.randn(
            1, 3, input_size, input_size, generator=gen, device=device, dtype=torch.float32
        )
        with torch.no_grad():
            eager_by_seed[int(seed)] = run_model(x)

    inductor_by_seed: Optional[Dict[int, Dict[str, Any]]] = None
    try:
        if cuda_available:
            inductor_by_seed = _run_inductor_oracle_subprocess(
                seeds=seeds,
                input_size=input_size,
                weights_path=weights_path,
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    finally:
        try:
            os.remove(weights_path)
        except OSError:
            pass

    cases: List[Dict[str, Any]] = []
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        x = torch.randn(
            1, 3, input_size, input_size, generator=gen, device=device, dtype=torch.float32
        )
        inputs = (x,)
        eager = eager_by_seed[int(seed)]

        inductor_entry = (
            inductor_by_seed.get(int(seed), {"status": "skipped", "output": None, "reason": "no oracle"})
            if inductor_by_seed is not None
            else {"status": "skipped", "output": None, "reason": "CUDA not available"}
        )
        inductor_status = str(inductor_entry["status"])
        inductor_reason = inductor_entry.get("reason")
        inductor_ok = inductor_status == "pass"
        inductor_abs = 0.0
        inductor_rel = 0.0
        inductor_out = inductor_entry.get("output")

        t0 = time.perf_counter()
        with torch.no_grad():
            ours = compiled_result(*inputs, mode="compiled")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if inductor_status == "pass" and inductor_out is not None:
            inductor_ok, inductor_abs, inductor_rel = _within_tolerance(
                inductor_out, ours.detach().cpu().numpy()
            )
            if not inductor_ok:
                inductor_status = "fail"

        eager_ok, eager_abs, eager_rel = _within_tolerance(
            eager.detach().cpu().numpy(), ours.detach().cpu().numpy()
        )

        cases.append(
            {
                "seed": int(seed),
                "input_shape": [1, 3, input_size, input_size],
                "latency_ms": float(latency_ms),
                "eager_pytorch": {
                    "within_tolerance": eager_ok,
                    "max_abs_error": eager_abs,
                    "max_rel_error": eager_rel,
                },
                "torch_compile_inductor": {
                    "status": inductor_status,
                    "within_tolerance": inductor_ok if inductor_status == "pass" else None,
                    "max_abs_error": inductor_abs if inductor_status == "pass" else None,
                    "max_rel_error": inductor_rel if inductor_status == "pass" else None,
                    "reason": inductor_reason,
                },
            }
        )

    summary = compiled_result.summary() if compiled_result is not None else {}
    exec_report = compiled_result.execution_report() if compiled_result is not None else {}
    report: Dict[str, Any] = {
        "model": "resnet18",
        "backend": "cuda",
        "execution_backend": (
            "cuda_graph_executor" if cuda_available else "numpy_graph_reference_fallback"
        ),
        "cuda_available": cuda_available,
        "host": _collect_host_info(torch),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "input_size": int(input_size),
        "graph_op_count": int(summary.get("graph_op_count", 0)),
        "runtime_op_count": int(summary.get("runtime_op_count", 0)),
        "cases": cases,
        "all_cases_within_tolerance_vs_inductor": all(
            c["torch_compile_inductor"]["status"] == "pass"
            and c["torch_compile_inductor"]["within_tolerance"]
            for c in cases
        ),
        "all_cases_within_tolerance_vs_eager": all(c["eager_pytorch"]["within_tolerance"] for c in cases),
    }

    parent = os.path.dirname(output_json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ResNet-18 end-to-end CUDA benchmark")
    parser.add_argument(
        "--output",
        default="bench/results/real_model_end_to_end.json",
        help="JSON artifact path (relative to repo root)",
    )
    args = parser.parse_args()
    report = run_real_model_benchmark(output_json_path=args.output)
    print(
        json.dumps(
            {
                "ok_eager": report["all_cases_within_tolerance_vs_eager"],
                "ok_inductor": report["all_cases_within_tolerance_vs_inductor"],
                "execution_backend": report["execution_backend"],
                "cuda_available": report["cuda_available"],
                "model": report["model"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
