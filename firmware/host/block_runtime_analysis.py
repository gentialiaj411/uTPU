import argparse
import json
import os

import numpy as np

from program_loader import ProgramLoader
from tiled_inference import TiledInferenceEngine, get_default_paths


def compare_modes(images, labels, legacy_engine, block_engine):
    max_abs_logit_diff = 0.0
    legacy_correct = 0
    block_correct = 0
    for i in range(len(labels)):
        l_pred, l_logits = legacy_engine.predict(images[i])
        b_pred, b_logits = block_engine.predict(images[i])
        if l_pred == labels[i]:
            legacy_correct += 1
        if b_pred == labels[i]:
            block_correct += 1
        diff = float(np.max(np.abs(l_logits - b_logits)))
        if diff > max_abs_logit_diff:
            max_abs_logit_diff = diff
    return {
        "num_samples": int(len(labels)),
        "legacy_accuracy_pct": round((legacy_correct / len(labels)) * 100.0, 4),
        "array_block_accuracy_pct": round((block_correct / len(labels)) * 100.0, 4),
        "accuracy_delta_pct": round(((block_correct - legacy_correct) / len(labels)) * 100.0, 4),
        "max_abs_logit_diff": max_abs_logit_diff,
    }


def theoretical_full_layer_words(fc1_block_ops, fc2_block_ops, array_size):
    # Lower bound estimate with a future ISA that can preload tensors and only issue:
    # LOADWEI + LOADIN + RUN per block op, then FETCH final layer once.
    # Fetch cost for final ARRAY_SIZE output vector is 2 fetches per packed 16-bit word.
    fetch_final = (array_size // 4) * 2
    return (fc1_block_ops + fc2_block_ops) * 3 + fetch_final + 1  # +HALT


def main():
    parser = argparse.ArgumentParser(description="Blocked FC runtime/codegen analysis")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output-json", type=str, default="build/reports/block_runtime_metrics.json")
    parser.add_argument("--output-md", type=str, default="build/reports/block_runtime_report.md")
    args = parser.parse_args()

    weights_dir, model_path, data_dir = get_default_paths()
    images = np.load(os.path.join(data_dir, "mnist_14x14_test.npy"))
    labels = np.load(os.path.join(data_dir, "test_labels.npy"))
    n = min(args.num_samples, len(labels))
    images = images[:n]
    labels = labels[:n]

    legacy_engine = TiledInferenceEngine(weights_dir, model_path, tiling_mode="legacy_2x2", verbose=False)
    block_engine = TiledInferenceEngine(weights_dir, model_path, tiling_mode="array_block", verbose=False)

    eq = compare_modes(images, labels, legacy_engine, block_engine)
    block_engine.predict(images[0])
    run_stats = block_engine.get_last_run_stats()
    array_size = int(run_stats["array_size"])

    loader = ProgramLoader(uart=None, verbose=False)

    fc1_build = loader.build_fc_layer_block_program(
        weights_int4=block_engine.fc1_weight,
        activations_int4=block_engine.preprocess_image(images[0]),
        out_features=block_engine.fc1_weight.shape[0],
        in_features=block_engine.fc1_weight.shape[1],
        array_size=array_size,
        apply_relu=False,
        apply_quant=True,
    )
    # FC2 input from array_block engine to keep scale path identical.
    fc1_out_logits = block_engine.fc_layer(
        block_engine.preprocess_image(images[0]),
        block_engine.fc1_weight,
        block_engine.fc1_scale,
        apply_relu=True,
    )[0]
    fc2_build = loader.build_fc_layer_block_program(
        weights_int4=block_engine.fc2_weight,
        activations_int4=np.clip(np.round(fc1_out_logits), -8, 7).astype(np.int8),
        out_features=block_engine.fc2_weight.shape[0],
        in_features=block_engine.fc2_weight.shape[1],
        array_size=array_size,
        apply_relu=False,
        apply_quant=True,
    )

    fc1_est = loader.estimate_fc_layer_instruction_words(
        out_features=block_engine.fc1_weight.shape[0],
        in_features=block_engine.fc1_weight.shape[1],
        array_size=array_size,
    )
    fc2_est = loader.estimate_fc_layer_instruction_words(
        out_features=block_engine.fc2_weight.shape[0],
        in_features=block_engine.fc2_weight.shape[1],
        array_size=array_size,
    )
    fc1_seg = loader.build_fc_layer_block_program_segmented(
        weights_int4=block_engine.fc1_weight,
        activations_int4=block_engine.preprocess_image(images[0]),
        out_features=block_engine.fc1_weight.shape[0],
        in_features=block_engine.fc1_weight.shape[1],
        array_size=array_size,
        apply_relu=False,
        apply_quant=True,
        max_words_per_segment=1024,
    )
    fc2_seg = loader.build_fc_layer_block_program_segmented(
        weights_int4=block_engine.fc2_weight,
        activations_int4=np.clip(np.round(fc1_out_logits), -8, 7).astype(np.int8),
        out_features=block_engine.fc2_weight.shape[0],
        in_features=block_engine.fc2_weight.shape[1],
        array_size=array_size,
        apply_relu=False,
        apply_quant=True,
        max_words_per_segment=1024,
    )

    total_block_program_words = fc1_build["program_instruction_words"] + fc2_build["program_instruction_words"]
    total_legacy_words = fc1_est["legacy_2x2_instruction_words"] + fc2_est["legacy_2x2_instruction_words"]
    total_block_ops = fc1_build["block_ops"] + fc2_build["block_ops"]
    theoretical_words = theoretical_full_layer_words(fc1_build["block_ops"], fc2_build["block_ops"], array_size)

    metrics = {
        "array_size": array_size,
        "equivalence": eq,
        "fc1": {
            "shape": list(block_engine.fc1_weight.shape),
            "block_count": int(fc1_build["block_ops"]),
            "generated_instruction_words": int(fc1_build["program_instruction_words"]),
            "fits_instruction_bram": bool(fc1_build["fits_instruction_bram"]),
            "legacy_instruction_words": int(fc1_est["legacy_2x2_instruction_words"]),
            "segmented": {
                "segment_count": int(fc1_seg["segment_count"]),
                "instruction_words_per_segment": [int(x) for x in fc1_seg["segment_words"]],
                "fits_each_segment": bool(fc1_seg["fits_each_segment"]),
                "first_segment_has_acc_clear": bool(fc1_seg["first_segment_has_acc_clear"]),
                "later_segments_have_acc_clear": bool(fc1_seg["later_segments_have_acc_clear"]),
                "final_segment_has_finalize_fetch": bool(fc1_seg["final_segment_has_finalize_fetch"]),
                "earlier_segments_have_finalize_fetch": bool(fc1_seg["earlier_segments_have_finalize_fetch"]),
                "accumulator_persistence_required": bool(fc1_seg["accumulator_persistence_required"]),
            },
        },
        "fc2": {
            "shape": list(block_engine.fc2_weight.shape),
            "block_count": int(fc2_build["block_ops"]),
            "generated_instruction_words": int(fc2_build["program_instruction_words"]),
            "fits_instruction_bram": bool(fc2_build["fits_instruction_bram"]),
            "legacy_instruction_words": int(fc2_est["legacy_2x2_instruction_words"]),
            "segmented": {
                "segment_count": int(fc2_seg["segment_count"]),
                "instruction_words_per_segment": [int(x) for x in fc2_seg["segment_words"]],
                "fits_each_segment": bool(fc2_seg["fits_each_segment"]),
            },
        },
        "totals": {
            "block_count": int(total_block_ops),
            "legacy_instruction_words": int(total_legacy_words),
            "block_program_instruction_words": int(total_block_program_words),
            "theoretical_full_layer_instruction_words": int(theoretical_words),
            "instruction_bram_words": 1024,
            "full_block_program_fits_instruction_bram": bool(total_block_program_words <= 1024),
            "legacy_tile_rpc_count": int(run_stats["totals"]["legacy_2x2_runs"]),
            "array_block_rpc_count": int(total_block_ops),
            "estimated_uart_transactions_avoided_vs_legacy": int(
                run_stats["totals"]["legacy_2x2_runs"] - total_block_ops
            ),
            "segmented_upload_start_phases_fc1": int(fc1_seg["segment_count"]),
            "segmented_upload_start_phases_fc2": int(fc2_seg["segment_count"]),
            "segmented_upload_start_phases_total": int(fc1_seg["segment_count"] + fc2_seg["segment_count"]),
            "legacy_upload_start_phases_total": int(run_stats["totals"]["legacy_2x2_runs"]),
            "segmented_phase_reduction_vs_legacy": int(
                run_stats["totals"]["legacy_2x2_runs"] - (fc1_seg["segment_count"] + fc2_seg["segment_count"])
            ),
        },
        "fpga_support": {
            "int32_accumulation_supported": bool(
                fc1_build.get("int32_accumulation_supported", False)
            ),
            "quantize_after_accumulation_supported": bool(
                fc1_build.get("quantize_after_accumulation_supported", False)
            ),
            "execute_fc_layer_blocked_hardware_supported": bool(
                fc1_build["executable_on_current_fpga_path"] and fc2_build["executable_on_current_fpga_path"]
            ),
            "executable_on_current_fpga_path": bool(
                fc1_build["executable_on_current_fpga_path"] and fc2_build["executable_on_current_fpga_path"]
            ),
            "fc1_blockers": fc1_build["blockers"],
            "fc2_blockers": fc2_build["blockers"],
        },
    }

    report = []
    report.append("# Block Runtime Report")
    report.append("")
    report.append("## What changed")
    report.append("- Added `ProgramLoader.execute_fc_layer_blocked(...)` API.")
    report.append("- Added ARRAY_SIZE-aware block-program generation for one FC layer.")
    report.append("- Added ISA-level instruction count estimation for legacy vs block mode.")
    report.append("")
    report.append("## Metrics")
    report.append(f"- array_size: {metrics['array_size']}")
    report.append(f"- FC1 block count: {metrics['fc1']['block_count']}")
    report.append(f"- FC2 block count: {metrics['fc2']['block_count']}")
    report.append(f"- total block count: {metrics['totals']['block_count']}")
    report.append(f"- FC1 generated instruction words: {metrics['fc1']['generated_instruction_words']}")
    report.append(f"- FC2 generated instruction words: {metrics['fc2']['generated_instruction_words']}")
    report.append(f"- total block-program instruction words: {metrics['totals']['block_program_instruction_words']}")
    report.append(f"- full block program fits 1024-word BRAM: {metrics['totals']['full_block_program_fits_instruction_bram']}")
    report.append(f"- estimated UART transactions avoided vs legacy 515-tile RPC: {metrics['totals']['estimated_uart_transactions_avoided_vs_legacy']}")
    report.append(f"- FC1 segmented upload/start phases: {metrics['totals']['segmented_upload_start_phases_fc1']}")
    report.append(f"- FC2 segmented upload/start phases: {metrics['totals']['segmented_upload_start_phases_fc2']}")
    report.append(f"- total segmented upload/start phases: {metrics['totals']['segmented_upload_start_phases_total']}")
    report.append(f"- phase reduction vs legacy 515: {metrics['totals']['segmented_phase_reduction_vs_legacy']}")
    report.append("")
    report.append("## Numerical equivalence")
    report.append(f"- samples: {metrics['equivalence']['num_samples']}")
    report.append(f"- legacy accuracy: {metrics['equivalence']['legacy_accuracy_pct']}%")
    report.append(f"- array_block accuracy: {metrics['equivalence']['array_block_accuracy_pct']}%")
    report.append(f"- max_abs_logit_diff: {metrics['equivalence']['max_abs_logit_diff']}")
    report.append("")
    report.append("## FPGA execution status")
    report.append(f"- int32_accumulation_supported: {metrics['fpga_support']['int32_accumulation_supported']}")
    report.append(f"- quantize_after_accumulation_supported: {metrics['fpga_support']['quantize_after_accumulation_supported']}")
    report.append(f"- execute_fc_layer_blocked_hardware_supported: {metrics['fpga_support']['execute_fc_layer_blocked_hardware_supported']}")
    report.append(f"- executable on current FPGA path: {metrics['fpga_support']['executable_on_current_fpga_path']}")
    if metrics["fpga_support"]["fc1_blockers"]:
        report.append("- blockers:")
        for b in metrics["fpga_support"]["fc1_blockers"]:
            report.append(f"  - {b}")
    report_text = "\n".join(report) + "\n"

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(json.dumps(metrics, indent=2))
    print("\n" + report_text)


if __name__ == "__main__":
    main()
