import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOST_DIR = os.path.join(REPO_ROOT, "firmware", "host")
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from cuda_autotuner import tune_many_shapes


OUT_JSON = os.path.join(REPO_ROOT, "build", "reports", "autotuner_best_shape.json")


def _write_json(payload: dict) -> None:
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> int:
    shapes = [
        ("m32_k128", 32, 128),
        ("m32_k256", 32, 256),
        ("m64_k128", 64, 128),
        ("m64_k256", 64, 256),
        ("m128_k128", 128, 128),
        ("m128_k256", 128, 256),
    ]
    report = tune_many_shapes(shapes, warmup=1, iters=3)
    results = report.get("results", [])

    if not any(bool(r.get("executed")) for r in results):
        payload = {
            "result": "cuda_unavailable",
            "reason": results[0].get("reason", "no executed shapes") if results else "no results",
            "shapes_tested": [name for name, _, _ in shapes],
        }
        _write_json(payload)
        print(json.dumps(payload, sort_keys=True))
        return 0

    measured = []
    for r in results:
        fixed = r.get("fixed_latency_ms")
        best = r.get("best_latency_ms")
        if fixed is None or best is None or float(fixed) <= 0.0:
            continue
        pct = ((float(fixed) - float(best)) / float(fixed)) * 100.0
        shape = r.get("shape", {})
        measured.append(
            {
                "shape_name": r.get("shape_name"),
                "M": int(shape.get("M")),
                "K": int(shape.get("K")),
                "default_kernel_ms": float(fixed),
                "best_kernel_ms": float(best),
                "pct_improvement": float(pct),
            }
        )

    if not measured:
        payload = {"result": "no_meaningful_win", "reason": "no valid latency pairs", "shapes_tested": len(results)}
        _write_json(payload)
        print(json.dumps(payload, sort_keys=True))
        return 0

    best_shape = max(measured, key=lambda x: x["pct_improvement"])
    if best_shape["pct_improvement"] < 1.0:
        payload = {
            "result": "no_meaningful_win",
            "best_observed_pct_improvement": best_shape["pct_improvement"],
            "best_shape": {"M": best_shape["M"], "K": best_shape["K"]},
            "measurements": measured,
        }
        _write_json(payload)
        print(json.dumps(payload, sort_keys=True))
        return 0

    payload = {
        "result": "meaningful_win",
        "best_shape": {"M": best_shape["M"], "K": best_shape["K"]},
        "default_ms": best_shape["default_kernel_ms"],
        "best_ms": best_shape["best_kernel_ms"],
        "pct_improvement": best_shape["pct_improvement"],
        "measurements": measured,
    }
    _write_json(payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
