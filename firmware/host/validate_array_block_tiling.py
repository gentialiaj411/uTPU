import argparse
import json
import os

import numpy as np

from tiled_inference import TiledInferenceEngine, get_default_paths


def evaluate(engine, images, labels):
    correct = 0
    logits_all = []
    for i in range(len(labels)):
        pred, logits = engine.predict(images[i])
        logits_all.append(logits.astype(np.float32))
        if pred == labels[i]:
            correct += 1
    return correct / len(labels), np.stack(logits_all, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Validate legacy 2x2 vs array-block tiling")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--array-size", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    weights_dir, model_path, data_dir = get_default_paths()
    test_images = np.load(os.path.join(data_dir, "mnist_14x14_test.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    n = min(args.num_samples, len(test_labels))
    images = test_images[:n]
    labels = test_labels[:n]

    legacy = TiledInferenceEngine(
        weights_dir, model_path, verbose=False, tiling_mode="legacy_2x2", array_size=args.array_size
    )
    block = TiledInferenceEngine(
        weights_dir, model_path, verbose=False, tiling_mode="array_block", array_size=args.array_size
    )

    legacy_acc, legacy_logits = evaluate(legacy, images, labels)
    block_acc, block_logits = evaluate(block, images, labels)

    diff = np.abs(legacy_logits - block_logits)
    max_abs_logit_diff = float(diff.max())

    # Sample-level run stats from one forward pass.
    block.predict(images[0])
    legacy.predict(images[0])
    block_stats = block.get_last_run_stats()
    legacy_stats = legacy.get_last_run_stats()

    results = {
        "num_samples": int(n),
        "legacy_accuracy_pct": round(legacy_acc * 100.0, 4),
        "array_block_accuracy_pct": round(block_acc * 100.0, 4),
        "accuracy_delta_pct": round((block_acc - legacy_acc) * 100.0, 4),
        "max_abs_logit_diff": max_abs_logit_diff,
        "legacy_stats": legacy_stats,
        "array_block_stats": block_stats,
        "invocations": {
            "fc1_legacy_2x2_runs": int(block_stats["fc1"]["legacy_2x2_runs"]),
            "fc1_array_block_runs": int(block_stats["fc1"]["array_block_runs"]),
            "fc2_legacy_2x2_runs": int(block_stats["fc2"]["legacy_2x2_runs"]),
            "fc2_array_block_runs": int(block_stats["fc2"]["array_block_runs"]),
            "total_legacy_2x2_runs": int(block_stats["totals"]["legacy_2x2_runs"]),
            "total_array_block_runs": int(block_stats["totals"]["array_block_runs"]),
            "estimated_reduction_vs_legacy": float(block_stats["totals"]["estimated_run_reduction_vs_legacy"]),
        },
    }

    md = []
    md.append("# Array Block Tiling Milestone Report")
    md.append("")
    md.append("## What changed")
    md.append("- Added `tiling_mode` support in `TiledInferenceEngine`: `legacy_2x2` and `array_block`.")
    md.append("- Added ARRAY_SIZE-aware block matmul path that pads to ARRAY_SIZE on input/output dims.")
    md.append("- Added run-count metrics for FC1, FC2, and total legacy-vs-block invocation counts.")
    md.append("")
    md.append("## Files changed")
    md.append("- firmware/host/tiled_inference.py")
    md.append("- firmware/host/validate_array_block_tiling.py")
    md.append("")
    md.append("## Old vs new invocation counts (per inference)")
    inv = results["invocations"]
    md.append(f"- FC1 legacy 2x2 runs: {inv['fc1_legacy_2x2_runs']}")
    md.append(f"- FC1 array-block runs: {inv['fc1_array_block_runs']}")
    md.append(f"- FC2 legacy 2x2 runs: {inv['fc2_legacy_2x2_runs']}")
    md.append(f"- FC2 array-block runs: {inv['fc2_array_block_runs']}")
    md.append(f"- Total legacy 2x2 runs: {inv['total_legacy_2x2_runs']}")
    md.append(f"- Total array-block runs: {inv['total_array_block_runs']}")
    md.append(f"- Estimated reduction vs legacy: {inv['estimated_reduction_vs_legacy']:.4f}x")
    md.append("")
    md.append("## Correctness comparison")
    md.append(f"- Samples compared: {results['num_samples']}")
    md.append(f"- Legacy accuracy: {results['legacy_accuracy_pct']}%")
    md.append(f"- Array-block accuracy: {results['array_block_accuracy_pct']}%")
    md.append(f"- Accuracy delta (array-block - legacy): {results['accuracy_delta_pct']} percentage points")
    md.append(f"- Max absolute logit difference: {results['max_abs_logit_diff']}")
    md.append("")
    md.append("## Remaining blockers for real FPGA execution")
    md.append("- This milestone is software/compiler simulation only.")
    md.append("- FPGA runtime still executes host-driven 2x2 tile RPCs (`execute2x2MatMul`) and does not yet issue ARRAY_SIZE block programs.")
    md.append("- RTL/control path would need new program semantics to consume block-level schedules end-to-end.")
    md_text = "\n".join(md) + "\n"

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md_text)

    print(json.dumps(results, indent=2))
    print("\n" + md_text)


if __name__ == "__main__":
    main()
