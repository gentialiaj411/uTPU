#!/usr/bin/env python3
"""Train and verify buffer-resident INT8 MLP deployments with the ISA simulator.

The simulator check is deliberately against a separate NumPy integer reference,
not the graph reference interpreter.  Full-dataset runs write a reproducible
JSON result under bench/results.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from board_config import BoardConfig
from buffer_resident_deployment import (
    deploy_model, encode_buffer_fill, integer_reference, quantize_input, run_isa_inference,
)
from isa_simulator import UTPUISASimulator
from run_real_model_accelerator import FloatMLP, _load_dataset, _train_float_model


class FashionMLP(nn.Module):
    """Three-layer, bias-free MLP within the same supported FX op whitelist."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(196, 128, bias=False)
        self.act1 = nn.LeakyReLU(0.25)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.act2 = nn.LeakyReLU(0.25)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 196)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def _read_idx(path: Path, *, images: bool) -> np.ndarray:
    import gzip
    import struct

    with gzip.open(path, "rb") as f:
        data = f.read()
    if images:
        magic, count, rows, cols = struct.unpack(">IIII", data[:16])
        if magic != 2051:
            raise ValueError(f"invalid IDX image magic in {path}")
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols).copy()
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"invalid IDX label magic in {path}")
    return np.frombuffer(data, dtype=np.uint8, offset=8, count=count).astype(np.int64)


def load_fashion(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load official Fashion-MNIST IDX files, resizing 28x28 to 14x14 by area."""
    paths = {
        "train_images": root / "train-images-idx3-ubyte.gz",
        "train_labels": root / "train-labels-idx1-ubyte.gz",
        "test_images": root / "t10k-images-idx3-ubyte.gz",
        "test_labels": root / "t10k-labels-idx1-ubyte.gz",
    }
    missing = [p for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Fashion-MNIST data missing; place official IDX .gz files in " + str(root))
    xtr = _read_idx(paths["train_images"], images=True)
    ytr = _read_idx(paths["train_labels"], images=False)
    xte = _read_idx(paths["test_images"], images=True)
    yte = _read_idx(paths["test_labels"], images=False)
    def resize(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)
        return F.interpolate(t, size=(14, 14), mode="area").squeeze(1).numpy().reshape(-1, 196)
    return resize(xtr), ytr, resize(xte), yte


def train_fashion(xtr: np.ndarray, ytr: np.ndarray) -> FashionMLP:
    torch.manual_seed(42)
    np.random.seed(42)
    model = FashionMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    xb, yb = torch.from_numpy(xtr), torch.from_numpy(ytr)
    for _ in range(12):
        for indices in torch.randperm(len(xb)).split(128):
            logits = model(xb[indices])
            loss = F.cross_entropy(logits, yb[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def _accuracy_logits(model: nn.Module, images: np.ndarray, labels: np.ndarray) -> float:
    model.eval()
    pred = []
    with torch.no_grad():
        for start in range(0, len(images), 512):
            pred.extend(model(torch.from_numpy(images[start:start + 512])).argmax(1).tolist())
    return float(np.mean(np.asarray(pred) == labels))


def evaluate(name: str, model: nn.Module, xtr: np.ndarray, xte: np.ndarray, yte: np.ndarray,
             *, limit: int | None = None) -> Dict[str, Any]:
    n = len(xte) if limit is None else min(len(xte), int(limit))
    board = BoardConfig.boolean_xc7s50()
    deployment = deploy_model(model, board, torch.from_numpy(xtr))
    sim = UTPUISASimulator(array_size=board.array_size,
                           buffer_size=board.buffer_size,
                           cfg=deployment.config,
                           accumulator_data_width=32)
    for i, word in enumerate(deployment.weight_image_words):
        sim.buffer[i] = int(word)
    predictions = np.empty(n, dtype=np.int64)
    mismatch_count = 0
    within_1 = True
    sample_rows = []
    cycles = []
    started = time.perf_counter()
    for i in range(n):
        image_q = quantize_input(xte[i], deployment.input_scale)
        expected = integer_reference(image_q, deployment)
        actual, cycle_count = run_isa_inference(image_q, deployment, sim)
        mismatch = not np.array_equal(actual, expected)
        mismatch_count += int(mismatch)
        within_1 = within_1 and not mismatch
        predictions[i] = int(np.argmax(actual))
        cycles.append(int(cycle_count))
        if i < 20:
            sample_rows.append({
                "index": i,
                "label": int(yte[i]),
                "predicted_class": int(predictions[i]),
                "expected_int8": expected.tolist(),
                "actual_int8": actual.tolist(),
                "non_saturated_outputs": int(np.count_nonzero((expected > -127) & (expected < 127))),
                "bit_exact": not mismatch,
                "cycles": int(cycle_count),
            })
    elapsed = time.perf_counter() - started
    float_acc = _accuracy_logits(model, xte[:n], yte[:n])
    first = deployment.layers[0]
    input_padded = ((first.input_features + 7) // 8) * 8
    input_fill_bytes = 5 + input_padded
    weight_fill_bytes = 5 + 2 * len(deployment.weight_image_words)
    control_upload_bytes = 3 + len(deployment.control_program)
    cold_tx = weight_fill_bytes + control_upload_bytes + input_fill_bytes + 1  # A2 start
    steady_tx = input_fill_bytes + 2  # A6 repeat + A2 start
    uart_baud = 115200
    return {
        "schema_version": 1,
        "model": name,
        "dataset": "MNIST" if name == "mnist_mlp" else "Fashion-MNIST",
        "evaluated_images": n,
        "full_test_set": n == len(xte),
        "deployment": {
            "board": board.as_dict(),
            "layers": [{"name": l.name, "input_features": l.input_features,
                        "output_features": l.output_features, "weight_words": l.weight_words,
                        "control_words": l.control_words} for l in deployment.layers],
            "weight_words": len(deployment.weight_image_words),
            "control_words": deployment.control_words,
            "activation_scratch_words": deployment.fit_report["activation_scratch_words"],
            "fits": deployment.fit_report["fits"],
            "bram_budget": deployment.fit_report["bram_budget"],
        },
        "float_accuracy": float_acc,
        "int8_accuracy": float(np.mean(predictions == yte[:n])),
        "isa_vs_independent_integer_reference": {
            "all_output_bytes_bit_exact": mismatch_count == 0,
            "mismatched_images": mismatch_count,
            "checked_output_bytes": n * deployment.output_features,
        },
        "cycles_per_inference": {
            "cold_device_cycles": int(cycles[0]) if cycles else None,
            "cold_upload_and_first_input_bytes": int(cold_tx),
            "cold_host_uart_seconds_at_115200_8n1": float(cold_tx * 10 / uart_baud),
            "steady_state_mean": float(np.mean(cycles)) if cycles else None,
            "steady_state_min": int(min(cycles)) if cycles else None,
            "steady_state_max": int(max(cycles)) if cycles else None,
            "steady_state_input_start_bytes": int(steady_tx),
            "steady_state_host_uart_seconds_per_image_at_115200_8n1": float(steady_tx * 10 / uart_baud),
            "full_10k_host_to_fpga_bytes": int(weight_fill_bytes + control_upload_bytes
                                                   + len(xte) * input_fill_bytes + len(xte)
                                                   + max(0, len(xte) - 1)),
            "full_10k_host_to_fpga_uart_seconds_at_115200_8n1": float(
                (weight_fill_bytes + control_upload_bytes + len(xte) * input_fill_bytes
                 + len(xte) + max(0, len(xte) - 1)) * 10 / uart_baud
            ),
            "note": "Device cycles are ISA perf/cycle counters; A5 fill is outside START-to-HALT counters, so UART load time is reported separately. Current top supports configurable UART_BAUD.",
        },
        "simulator_elapsed_seconds": elapsed,
        "samples_first_20": sample_rows,
    }


def run_rtl_uart_replay(model_name: str, xte: np.ndarray, yte: np.ndarray,
                        deployment: Any, *, sample_count: int = 20) -> Dict[str, Any]:
    """Replay A5 weights + control-only code + inputs through Boolean RTL UART."""
    from run_rtl_fused_sim import _resolve_iverilog_tools

    n = min(int(sample_count), len(xte))
    simulator = UTPUISASimulator(array_size=8, buffer_size=32768,
                                 cfg=deployment.config, accumulator_data_width=32)
    for address, word in enumerate(deployment.weight_image_words):
        simulator.buffer[address] = int(word)
    input_chunks = []
    expected_chunks = []
    sample_info = []
    for i in range(n):
        image_q = quantize_input(xte[i], deployment.input_scale)
        expected = integer_reference(image_q, deployment)
        actual, cycles = run_isa_inference(image_q, deployment, simulator)
        if not np.array_equal(expected, actual):
            raise AssertionError(f"ISA vs independent integer mismatch at {model_name} sample {i}")
        padded_n = max(16, ((deployment.output_features + 7) // 8) * 8)
        output = bytearray(padded_n)
        output[:deployment.output_features] = bytes(int(v) & 0xff for v in expected)
        expected_chunks.append(bytes(output))
        input_chunks.append(deployment.input_fill_command(image_q))
        sample_info.append({"index": i, "label": int(yte[i]),
                            "predicted_class": int(np.argmax(expected)),
                            "non_saturated_output_count": int(np.count_nonzero((expected > -127) & (expected < 127))),
                            "cycles": cycles, "expected_uart_hex": bytes(output).hex()})

    program = deployment.control_program
    initial = (encode_buffer_fill(0, deployment.weight_image_words)
               + bytes((0xA1, deployment.control_words & 0xff, (deployment.control_words >> 8) & 0xff))
               + program)
    inputs = b"".join(input_chunks)
    expected_bytes = b"".join(expected_chunks)
    vectors = REPO / "build" / "test_vectors"
    vectors.mkdir(parents=True, exist_ok=True)
    initial_path = vectors / f"{model_name}_br_initial.mem"
    inputs_path = vectors / f"{model_name}_br_inputs.mem"
    expected_path = vectors / f"{model_name}_br_expected.mem"
    for path, data in ((initial_path, initial), (inputs_path, inputs), (expected_path, expected_bytes)):
        path.write_text("\n".join(f"{byte:02x}" for byte in data) + "\n", encoding="utf-8")
    svh = vectors / "buffer_resident_uart_replay.svh"
    svh.write_text("\n".join([
        "`define UART_REPLAY_INITIAL_MEM \"build/test_vectors/" + initial_path.name + "\"",
        "`define UART_REPLAY_INPUT_MEM \"build/test_vectors/" + inputs_path.name + "\"",
        "`define UART_REPLAY_EXPECTED_MEM \"build/test_vectors/" + expected_path.name + "\"",
        f"`define UART_REPLAY_INITIAL_N {len(initial)}",
        f"`define UART_REPLAY_INPUT_N {len(inputs)}",
        f"`define UART_REPLAY_INPUT_N_PER_SAMPLE {len(input_chunks[0])}",
        f"`define UART_REPLAY_EXPECTED_N {len(expected_bytes)}",
        f"`define UART_REPLAY_OUTPUT_N_PER_SAMPLE {len(expected_chunks[0])}",
        f"`define UART_REPLAY_SAMPLES {n}",
        "`define UART_REPLAY_ARRAY_SIZE 8",
        "`define UART_REPLAY_BUFFER_SIZE 32768",
        "`define UART_REPLAY_WEIGHT_REGION_WORDS 27648",
        "`define UART_REPLAY_PROG_DEPTH 32768",
        "`define UART_REPLAY_UART_BAUD 3125000",
        "`define UART_REPLAY_CORE_CLK_HZ 50000000",
        "`define UART_REPLAY_CLK_PERIOD_NS 20",
        "`define UART_REPLAY_RST_ACTIVE_LOW 0",
        "`define UART_REPLAY_MAX_BATCH_COUNT 4",
    ]) + "\n", encoding="utf-8")

    iv, vvp = _resolve_iverilog_tools()
    if not iv or not vvp:
        return {"passed": False, "diagnostic": "iverilog/vvp binaries unavailable", "samples": sample_info}
    sources = [
        "rtl/tb/tb_buffer_resident_uart_replay.sv", "rtl/tb/xpm_memory_sdpram_stub.sv",
        "rtl/top/top.sv", "rtl/memory/instr_bram.sv", "rtl/PEArray/pe_controller.sv",
        "rtl/PEArray/pe_array.sv", "rtl/PEArray/pe.sv", "rtl/quantizer/quantizer.sv",
        "rtl/quantizer/quantizer_array.sv", "rtl/LeakyReLU/leaky_relu.sv",
        "rtl/LeakyReLU/leaky_relu_array.sv", "rtl/unified_buffer/unified_buffer.sv",
        "rtl/fifo/fifo_rx.sv", "rtl/fifo/fifo_tx.sv", "rtl/UART/uart.sv",
        "rtl/UART/uart_receiver.sv", "rtl/UART/uart_transmitter.sv", "rtl/UART/clk_divider.sv",
    ]
    executable = REPO / "build" / "rtl_sim" / f"tb_{model_name}_buffer_resident.out"
    executable.parent.mkdir(parents=True, exist_ok=True)
    compile_result = subprocess.run([iv, "-g2012", "-DICARUS", "-o", str(executable),
                                     *[str(REPO / source) for source in sources]],
                                    cwd=REPO, capture_output=True, text=True)
    if compile_result.returncode:
        return {"passed": False, "diagnostic": "iverilog compile failed",
                "log_tail": (compile_result.stdout + compile_result.stderr)[-5000:], "samples": sample_info}
    run_result = subprocess.run([vvp, str(executable)], cwd=REPO, capture_output=True, text=True)
    log = (run_result.stdout or "") + (run_result.stderr or "")
    seen = re.findall(r"SAMPLE_RESULT index=(\d+) byte_count=(\d+) pass=(\d+)", log)
    perf_rows = re.findall(r"SAMPLE_PERF_PROGRAM_CYCLES index=(\d+) cycles=(\d+)", log)
    cold_match = re.search(r"COLD_CORE_CYCLES_INCLUDING_WEIGHT_PROGRAM_INPUT_FILL=(\d+)", log)
    steady_rows = re.findall(r"STEADY_CORE_CYCLES_INCLUDING_INPUT_FILL index=(\d+) cycles=(\d+)", log)
    return {
        "passed": run_result.returncode == 0 and "TB_RESULT: PASS" in log and len(seen) == n
                  and all(row[2] == "1" for row in seen),
        "samples_requested": n,
        "samples_passed": sum(row[2] == "1" for row in seen),
        "output_bytes_checked": len(expected_bytes),
        "rtl_perf_program_cycles_per_inference": [
            {"sample_index": int(index), "cycles": int(cycles)} for index, cycles in perf_rows
        ],
        "cold_core_cycles_including_weight_program_input_fill": int(cold_match.group(1)) if cold_match else None,
        "steady_core_cycles_including_repeat_and_input_fill": [
            {"sample_index": int(index), "cycles": int(cycles)} for index, cycles in steady_rows
        ],
        "rtl_parameters": {"ARRAY_SIZE": 8, "COMPUTE_DATA_WIDTH": 8, "BUFFER_SIZE": 32768,
                           "WEIGHT_REGION_WORDS": 27648, "PROG_DEPTH": 32768,
                           "MAX_BATCH_COUNT": 4, "RST_ACTIVE_LOW": 0,
                           "CORE_CLK_HZ": 50000000, "MMCM": "bypassed in simulation",
                           "UART_BAUD_SIM": 3125000},
        "samples": sample_info,
        "log_tail": log[-5000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mnist_mlp", "fashion_mlp"), required=True)
    parser.add_argument("--images", type=int, default=None, help="limit test images; default is all 10,000")
    parser.add_argument("--fashion-data", type=Path, default=REPO / "software" / "data" / "fashion-mnist")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rtl-replay", action="store_true",
                        help="run UART-protocol RTL replay for the first 20 test images")
    parser.add_argument("--rtl-replay-images", type=int, default=20)
    args = parser.parse_args()
    if args.model == "mnist_mlp":
        xtr, ytr, xte, yte = _load_dataset()
        model = _train_float_model(xtr, ytr)
    else:
        xtr, ytr, xte, yte = load_fashion(args.fashion_data)
        model = train_fashion(xtr, ytr)
    result = evaluate(args.model, model, xtr, xte, yte, limit=args.images)
    if args.rtl_replay:
        deployment = deploy_model(model, BoardConfig.boolean_xc7s50(), torch.from_numpy(xtr))
        result["rtl_uart_replay"] = run_rtl_uart_replay(
            args.model, xte, yte, deployment, sample_count=args.rtl_replay_images
        )
    out = args.output or (REPO / "bench" / "results" / f"buffer_resident_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "images": result["evaluated_images"],
                      "float_accuracy": result["float_accuracy"],
                      "int8_accuracy": result["int8_accuracy"],
                      "bit_exact": result["isa_vs_independent_integer_reference"]["all_output_bytes_bit_exact"],
                      "output": str(out)}, indent=2))
    passed = result["isa_vs_independent_integer_reference"]["all_output_bytes_bit_exact"]
    if args.rtl_replay:
        passed = passed and bool(result["rtl_uart_replay"].get("passed"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
