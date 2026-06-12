from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from run_rtl_batched_gemm_sim import run_rtl_batched_gemm_sim


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "build" / "reports" / "rtl_batched_gemm_sweep.json"
CASES: Tuple[Tuple[int, int, int], ...] = (
    (16, 16, 1),
    (16, 16, 4),
    (16, 16, 16),
    (16, 16, 32),
    (16, 16, 64),
    (32, 32, 1),
    (32, 32, 16),
    (64, 64, 1),
    (64, 64, 16),
)


def run_sweep(output_json: str = str(OUTPUT_JSON)) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    rtl_available = True
    for out_features, in_features, batch_size in CASES:
        stem = f"batched_sweep_o{out_features}_i{in_features}_b{batch_size}"
        metrics = run_rtl_batched_gemm_sim(
            output_json=str(REPO_ROOT / "build" / "reports" / f"{stem}.json"),
            out_features=out_features,
            in_features=in_features,
            batch_size=batch_size,
            stem=stem,
        )
        rows.append(
            {
                "shape": {"out_features": out_features, "in_features": in_features},
                "batch_size": batch_size,
                "rtl_sim_executed": bool(metrics["rtl_sim_executed"]),
                "rtl_sim_passed": bool(metrics["rtl_sim_passed"]),
                "expected_fetch_bytes": len(metrics["expected_fetch_bytes"]),
                "perf_cycle_counter": metrics["perf_cycle_counter"],
                "perf_busy_counter": metrics["perf_busy_counter"],
            }
        )
        if not metrics["rtl_sim_executed"]:
            rtl_available = False
            break

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "status": "ok" if rtl_available else "rtl_unavailable",
        "cases": rows,
        "aggregate": {
            "rtl_available": rtl_available,
            "all_cases_passed": rtl_available and all(row["rtl_sim_passed"] for row in rows),
        },
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    payload = run_sweep()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
