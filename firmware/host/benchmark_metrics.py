import json
import os
import time

import numpy as np
import torch

from tiled_inference import TiledInferenceEngine, get_default_paths


def main():
    weights_dir, model_path, data_dir = get_default_paths()

    test_images = np.load(os.path.join(data_dir, "mnist_14x14_test.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    engine = TiledInferenceEngine(weights_dir, model_path, verbose=False)

    t0 = time.perf_counter()
    tiled_acc, tiled_correct, total = engine.evaluate(test_images, test_labels)
    t1 = time.perf_counter()

    # PyTorch reference accuracy
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../software/model"))
    from qat_model import MNISTNet

    model = MNISTNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    pt_correct = 0
    with torch.no_grad():
        for i in range(total):
            pt_input = torch.tensor(test_images[i], dtype=torch.float32).unsqueeze(0)
            if model(pt_input).argmax().item() == test_labels[i]:
                pt_correct += 1
    pt_acc = pt_correct / total

    # Throughput on subset for a stable latency estimate.
    sample_n = min(1000, total)
    s0 = time.perf_counter()
    for i in range(sample_n):
        engine.predict(test_images[i])
    s1 = time.perf_counter()
    subset_dt = s1 - s0

    fc1_bin = os.path.join(weights_dir, "fc1_weight.bin")
    fc2_bin = os.path.join(weights_dir, "fc2_weight.bin")

    metrics = {
        "dataset_samples": int(total),
        "pytorch_accuracy_pct": round(pt_acc * 100.0, 2),
        "tiled_accuracy_pct": round(tiled_acc * 100.0, 2),
        "accuracy_delta_pct": round(abs(pt_acc - tiled_acc) * 100.0, 4),
        "tiled_eval_seconds": round(t1 - t0, 4),
        "latency_ms_per_sample_subset": round((subset_dt * 1000.0) / sample_n, 4),
        "throughput_samples_per_sec_subset": round(sample_n / subset_dt, 2),
        "fc1_weight_bin_bytes": os.path.getsize(fc1_bin) if os.path.exists(fc1_bin) else None,
        "fc2_weight_bin_bytes": os.path.getsize(fc2_bin) if os.path.exists(fc2_bin) else None,
    }

    metrics["total_weight_bin_bytes"] = (
        (metrics["fc1_weight_bin_bytes"] or 0) + (metrics["fc2_weight_bin_bytes"] or 0)
    )

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "software_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
