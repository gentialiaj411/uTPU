import argparse
import json
import os
import sys

import numpy as np

from program_loader import ProgramLoader
from tiled_inference import TiledInferenceEngine, get_default_paths


FUSED_BLOCKED_WORDS = 1017


def compute_unfused_vs_fused_comparison() -> dict:
    weights_dir, model_path, data_dir = get_default_paths()
    eng = TiledInferenceEngine(weights_dir, model_path, tiling_mode="array_block", verbose=False)
    loader = ProgramLoader(uart=None, verbose=False)
    a = eng.array_size

    imgs = np.load(os.path.join(data_dir, "mnist_14x14_test.npy"))
    x0 = eng.preprocess_image(imgs[0]).astype(np.int8)

    fc1_un = loader.build_fc_layer_block_program(
        eng.fc1_weight,
        x0,
        out_features=eng.fc1_weight.shape[0],
        in_features=eng.fc1_weight.shape[1],
        array_size=a,
        apply_relu=True,
        apply_quant=True,
    )
    fc1_mid = eng.fc_layer(x0, eng.fc1_weight, eng.fc1_scale, apply_relu=True)[0]
    fc2_in = np.clip(np.round(fc1_mid), -8, 7).astype(np.int8)
    fc2_un = loader.build_fc_layer_block_program(
        eng.fc2_weight,
        fc2_in,
        out_features=eng.fc2_weight.shape[0],
        in_features=eng.fc2_weight.shape[1],
        array_size=a,
        apply_relu=False,
        apply_quant=True,
    )

    fc1_words = int(fc1_un["program_instruction_words"])
    fc2_words = int(fc2_un["program_instruction_words"])
    unfused_total = fc1_words + fc2_words

    pct_reduction = ((float(unfused_total) - float(FUSED_BLOCKED_WORDS)) / float(unfused_total)) * 100.0
    return {
        "method": "real_program_loader_unfused_block_programs_fc1_plus_fc2",
        "array_size": int(a),
        "fc1_unfused_words": fc1_words,
        "fc2_unfused_words": fc2_words,
        "unfused_total_words": int(unfused_total),
        "fused_words": int(FUSED_BLOCKED_WORDS),
        "percent_reduction_unfused_to_fused": pct_reduction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure unfused standalone BRAM words via real lowering path and compare to fused=1017."
    )
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    args = parser.parse_args()

    report = compute_unfused_vs_fused_comparison()
    if int(report["unfused_total_words"]) <= FUSED_BLOCKED_WORDS:
        print("baseline_invalid: unfused total not larger than fused")
        sys.exit(1)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(
        f"unfused_total={unfused_total} fused={FUSED_BLOCKED_WORDS} "
        f"pct_reduction={pct_reduction:.2f}%"
    )


if __name__ == "__main__":
    main()
