import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOST_DIR = os.path.join(REPO_ROOT, "firmware", "host")
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from cuda_autotuner import DEFAULT_CACHE_PATH, default_search_space, tune_many_shapes


def main():
    shapes = [
        ("tiny_fc1", 3, 4),
        ("tiny_fc2", 2, 3),
        ("fc1_like_small_fc1", 128, 64),
        ("shared_64x128_linear", 64, 128),
        ("fc2_like_small_fc2", 16, 64),
        ("stress_linear", 128, 256),
    ]
    report = tune_many_shapes(shapes, warmup=2, iters=5, cache_path=DEFAULT_CACHE_PATH)

    print("CUDA blocked-FC autotune")
    print("========================")
    print(f"search_space={json.dumps(default_search_space().schema(), sort_keys=True)}")
    print("shape,fixed_kernel_ms,best_kernel_ms,improvement_pct,best_schedule,max_abs_error,executed")
    for r in report["results"]:
        print(
            f"{r['shape_name']},{r['fixed_latency_ms']},{r['best_latency_ms']},"
            f"{r['improvement_pct']},{r['best_schedule']},{r['max_abs_error']},{r['executed']}"
        )
    print(f"\nwrote {DEFAULT_CACHE_PATH}")


if __name__ == "__main__":
    main()
