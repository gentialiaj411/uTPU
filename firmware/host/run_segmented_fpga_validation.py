import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from tiled_inference import TiledInferenceEngine, get_default_paths
from program_loader import ProgramLoader

try:
    from uart_driver import UARTDriver
except Exception:
    UARTDriver = None


def _write_outputs(metrics: Dict[str, Any], out_json: str, out_md: str) -> None:
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    md = []
    md.append("# Segmented FPGA Validation Report")
    md.append("")
    md.append("## Summary")
    md.append(f"- num_samples: {metrics['num_samples']}")
    md.append(f"- array_size: {metrics['array_size']}")
    md.append(f"- uart_available: {metrics['uart_available']}")
    md.append(f"- hardware_executed: {metrics['hardware_executed']}")
    md.append("")
    md.append("## Segmentation")
    md.append(f"- fc1_segment_count: {metrics['fc1_segment_count']}")
    md.append(f"- fc2_segment_count: {metrics['fc2_segment_count']}")
    md.append(f"- fc1_segment_words: {metrics['fc1_segment_words']}")
    md.append(f"- fc2_segment_words: {metrics['fc2_segment_words']}")
    md.append(f"- segmented_phase_count: {metrics['segmented_phase_count']}")
    md.append(f"- legacy_tile_rpc_count: {metrics['legacy_tile_rpc_count']}")
    md.append(f"- all_segments_fit_bram: {metrics['all_segments_fit_bram']}")
    md.append("")
    md.append("## Accuracy")
    md.append(f"- software_accuracy_pct: {metrics['software_accuracy_pct']}")
    md.append(f"- fpga_accuracy_pct: {metrics['fpga_accuracy_pct']}")
    md.append(f"- prediction_match_rate_pct: {metrics['prediction_match_rate_pct']}")
    md.append(f"- max_abs_output_diff: {metrics['max_abs_output_diff']}")
    md.append("")
    md.append("## Timing / IO")
    md.append(f"- total_wall_time_ms: {metrics['total_wall_time_ms']}")
    md.append(f"- upload_time_ms: {metrics['upload_time_ms']}")
    md.append(f"- execution_wait_time_ms: {metrics['execution_wait_time_ms']}")
    md.append(f"- fetch_time_ms: {metrics['fetch_time_ms']}")
    md.append(f"- total_uart_bytes_sent: {metrics['total_uart_bytes_sent']}")
    md.append(f"- total_uart_bytes_received: {metrics['total_uart_bytes_received']}")
    md.append("")
    if metrics["mismatch_examples"]:
        md.append("## Mismatch Examples")
        for ex in metrics["mismatch_examples"]:
            md.append(
                f"- sample={ex['sample_index']}, sw_pred={ex['software_pred']}, fpga_pred={ex['fpga_pred']}, "
                f"max_abs_diff={ex['max_abs_diff']}"
            )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmented blocked FPGA validation")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=str, default="build/reports/segmented_fpga_validation_metrics.json")
    parser.add_argument("--output-md", type=str, default="build/reports/segmented_fpga_validation_report.md")
    parser.add_argument("--segment-wait-ms", type=float, default=40.0)
    args = parser.parse_args()

    weights_dir, model_path, data_dir = get_default_paths()
    images = np.load(os.path.join(data_dir, "mnist_14x14_test.npy"))
    labels = np.load(os.path.join(data_dir, "test_labels.npy"))
    n = min(args.num_samples, len(labels))
    images = images[:n]
    labels = labels[:n]

    sw_engine = TiledInferenceEngine(weights_dir, model_path, tiling_mode="array_block", verbose=False)
    sw_engine.predict(images[0])
    sw_stats = sw_engine.get_last_run_stats()
    array_size = int(sw_stats["array_size"])

    uart_available = bool((not args.dry_run) and args.port and UARTDriver is not None)
    uart = None
    loader = None
    hardware_executed = False
    if uart_available:
        try:
            uart = UARTDriver(args.port, baud=args.baud)
            loader = ProgramLoader(uart, verbose=False)
            loader.resetChip()
            hardware_executed = True
        except Exception:
            uart_available = False
            hardware_executed = False
            uart = None
            loader = ProgramLoader(None, verbose=False)
    else:
        loader = ProgramLoader(None, verbose=False)

    # Build representative segmentation metadata.
    x0 = sw_engine.preprocess_image(images[0]).astype(np.int8)
    fc1_seg = loader.build_fc_layer_block_program_segmented(
        sw_engine.fc1_weight,
        x0,
        out_features=sw_engine.fc1_weight.shape[0],
        in_features=sw_engine.fc1_weight.shape[1],
        array_size=array_size,
        apply_relu=True,
        apply_quant=True,
        max_words_per_segment=1024,
    )
    fc1_sw = sw_engine.fc_layer(x0, sw_engine.fc1_weight, sw_engine.fc1_scale, apply_relu=True)[0]
    fc2_in = np.clip(np.round(fc1_sw), -8, 7).astype(np.int8)
    fc2_seg = loader.build_fc_layer_block_program_segmented(
        sw_engine.fc2_weight,
        fc2_in,
        out_features=sw_engine.fc2_weight.shape[0],
        in_features=sw_engine.fc2_weight.shape[1],
        array_size=array_size,
        apply_relu=False,
        apply_quant=True,
        max_words_per_segment=1024,
    )

    mismatch_examples: List[Dict[str, Any]] = []
    sw_correct = 0
    hw_correct = 0
    pred_match = 0
    max_abs_output_diff = 0.0

    total_wall_start = time.perf_counter()
    total_upload_ms = 0.0
    total_exec_wait_ms = 0.0
    total_fetch_ms = 0.0
    total_uart_sent = 0
    total_uart_recv = 0

    if hardware_executed:
        for i in range(n):
            sw_pred, sw_logits = sw_engine.predict(images[i])
            sw_correct += int(sw_pred == labels[i])

            x = sw_engine.preprocess_image(images[i]).astype(np.int8)
            fc1_res = loader.execute_fc_layer_blocked(
                sw_engine.fc1_weight,
                x,
                out_features=sw_engine.fc1_weight.shape[0],
                in_features=sw_engine.fc1_weight.shape[1],
                array_size=array_size,
                apply_relu=True,
                apply_quant=True,
                allow_segmentation=True,
                max_words_per_segment=1024,
                timeout=args.segment_wait_ms / 1000.0,
            )
            fc1_out = np.array(fc1_res.get("output_int4_padded", [0] * array_size), dtype=np.int8)[:sw_engine.fc1_weight.shape[0]]

            fc2_res = loader.execute_fc_layer_blocked(
                sw_engine.fc2_weight,
                fc1_out,
                out_features=sw_engine.fc2_weight.shape[0],
                in_features=sw_engine.fc2_weight.shape[1],
                array_size=array_size,
                apply_relu=False,
                apply_quant=True,
                allow_segmentation=True,
                max_words_per_segment=1024,
                timeout=args.segment_wait_ms / 1000.0,
            )
            hw_logits = np.array(fc2_res.get("output_int4_padded", [0] * array_size), dtype=np.float32)[:sw_engine.fc2_weight.shape[0]]
            hw_pred = int(np.argmax(hw_logits))

            hw_correct += int(hw_pred == labels[i])
            pred_match += int(hw_pred == sw_pred)
            diff = float(np.max(np.abs(hw_logits - sw_logits)))
            max_abs_output_diff = max(max_abs_output_diff, diff)

            if hw_pred != sw_pred and len(mismatch_examples) < 5:
                mismatch_examples.append({
                    "sample_index": int(i),
                    "software_pred": int(sw_pred),
                    "fpga_pred": int(hw_pred),
                    "label": int(labels[i]),
                    "max_abs_diff": diff,
                    "software_logits": sw_logits.tolist(),
                    "fpga_logits": hw_logits.tolist(),
                })

            for res in (fc1_res, fc2_res):
                t = res.get("timing", {})
                io = res.get("io", {})
                total_upload_ms += float(t.get("upload_time_ms", 0.0))
                total_exec_wait_ms += float(t.get("execution_wait_time_ms", 0.0))
                total_fetch_ms += float(t.get("fetch_time_ms", 0.0))
                total_uart_sent += int(io.get("bytes_sent", 0))
                total_uart_recv += int(io.get("bytes_received", 0))
    else:
        # Dry-run or no UART path: compute software-only accuracy.
        for i in range(n):
            sw_pred, _ = sw_engine.predict(images[i])
            sw_correct += int(sw_pred == labels[i])

    total_wall_ms = (time.perf_counter() - total_wall_start) * 1000.0

    software_accuracy_pct = round((sw_correct / n) * 100.0, 4) if n else 0.0
    fpga_accuracy_pct = round((hw_correct / n) * 100.0, 4) if (n and hardware_executed) else None
    prediction_match_rate_pct = round((pred_match / n) * 100.0, 4) if (n and hardware_executed) else None

    metrics = {
        "num_samples": int(n),
        "array_size": int(array_size),
        "legacy_tile_rpc_count": int(sw_stats["totals"]["legacy_2x2_runs"]),
        "segmented_phase_count": int(fc1_seg["segment_count"] + fc2_seg["segment_count"]),
        "fc1_segment_count": int(fc1_seg["segment_count"]),
        "fc2_segment_count": int(fc2_seg["segment_count"]),
        "fc1_segment_words": [int(x) for x in fc1_seg["segment_words"]],
        "fc2_segment_words": [int(x) for x in fc2_seg["segment_words"]],
        "all_segments_fit_bram": bool(fc1_seg["fits_each_segment"] and fc2_seg["fits_each_segment"]),
        "uart_available": bool(uart_available),
        "hardware_executed": bool(hardware_executed),
        "total_uart_bytes_sent": int(total_uart_sent),
        "total_uart_bytes_received": int(total_uart_recv),
        "total_wall_time_ms": round(total_wall_ms, 4),
        "upload_time_ms": round(total_upload_ms, 4),
        "execution_wait_time_ms": round(total_exec_wait_ms, 4),
        "fetch_time_ms": round(total_fetch_ms, 4),
        "software_accuracy_pct": software_accuracy_pct,
        "fpga_accuracy_pct": fpga_accuracy_pct,
        "prediction_match_rate_pct": prediction_match_rate_pct,
        "max_abs_output_diff": float(max_abs_output_diff) if hardware_executed else None,
        "mismatch_examples": mismatch_examples,
    }

    _write_outputs(metrics, args.output_json, args.output_md)
    print(json.dumps(metrics, indent=2))

    if uart is not None:
        uart.close()


if __name__ == "__main__":
    main()
