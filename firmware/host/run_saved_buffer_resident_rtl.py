#!/usr/bin/env python3
"""Compile/replay the latest generated buffer-resident UART vectors."""
from __future__ import annotations

import re
import json
import argparse
import subprocess
import sys
from pathlib import Path

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
sys.path.insert(0, str(HOST))
from run_rtl_fused_sim import _resolve_iverilog_tools

SOURCES = [
    "rtl/tb/tb_buffer_resident_uart_replay.sv", "rtl/tb/xpm_memory_sdpram_stub.sv",
    "rtl/top/top.sv", "rtl/memory/instr_bram.sv", "rtl/PEArray/pe_controller.sv",
    "rtl/PEArray/pe_array.sv", "rtl/PEArray/pe.sv", "rtl/quantizer/quantizer.sv",
    "rtl/quantizer/quantizer_array.sv", "rtl/LeakyReLU/leaky_relu.sv",
    "rtl/LeakyReLU/leaky_relu_array.sv", "rtl/unified_buffer/unified_buffer.sv",
    "rtl/fifo/fifo_rx.sv", "rtl/fifo/fifo_tx.sv", "rtl/UART/uart.sv",
    "rtl/UART/uart_receiver.sv", "rtl/UART/uart_transmitter.sv", "rtl/UART/clk_divider.sv",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mnist_mlp", "fashion_mlp"), default=None,
                        help="select saved vectors and update that model's replay artifact")
    args = parser.parse_args()
    model_name = args.model
    if model_name:
        vectors = REPO / "build" / "test_vectors"
        paths = {"initial": vectors / f"{model_name}_br_initial.mem",
                 "inputs": vectors / f"{model_name}_br_inputs.mem",
                 "expected": vectors / f"{model_name}_br_expected.mem"}
        if any(not path.is_file() for path in paths.values()):
            print(f"saved {model_name} replay vectors are missing")
            return 2
        sizes = {key: sum(1 for _ in path.open("r", encoding="utf-8")) for key, path in paths.items()}
        svh = vectors / "buffer_resident_uart_replay.svh"
        svh.write_text("\n".join([
            f'`define UART_REPLAY_INITIAL_MEM "build/test_vectors/{paths["initial"].name}"',
            f'`define UART_REPLAY_INPUT_MEM "build/test_vectors/{paths["inputs"].name}"',
            f'`define UART_REPLAY_EXPECTED_MEM "build/test_vectors/{paths["expected"].name}"',
            f'`define UART_REPLAY_INITIAL_N {sizes["initial"]}',
            f'`define UART_REPLAY_INPUT_N {sizes["inputs"]}',
            "`define UART_REPLAY_INPUT_N_PER_SAMPLE 205",
            f'`define UART_REPLAY_EXPECTED_N {sizes["expected"]}',
            f'`define UART_REPLAY_OUTPUT_N_PER_SAMPLE {sizes["expected"] // 20}',
            "`define UART_REPLAY_SAMPLES 20",
            "`define UART_REPLAY_ARRAY_SIZE 8", "`define UART_REPLAY_BUFFER_SIZE 32768",
            "`define UART_REPLAY_WEIGHT_REGION_WORDS 27648", "`define UART_REPLAY_PROG_DEPTH 32768",
            "`define UART_REPLAY_UART_BAUD 3125000", "`define UART_REPLAY_CORE_CLK_HZ 50000000",
            "`define UART_REPLAY_CLK_PERIOD_NS 20", "`define UART_REPLAY_RST_ACTIVE_LOW 0",
            "`define UART_REPLAY_MAX_BATCH_COUNT 4",
        ]) + "\n", encoding="utf-8")
    iv, vvp = _resolve_iverilog_tools()
    if not iv or not vvp:
        print("iverilog/vvp unavailable")
        return 2
    executable = REPO / "build" / "rtl_sim" / "tb_buffer_resident_saved.out"
    executable.parent.mkdir(parents=True, exist_ok=True)
    comp = subprocess.run([iv, "-g2012", "-DICARUS", "-o", str(executable),
                           *[str(REPO / item) for item in SOURCES]],
                          cwd=REPO, capture_output=True, text=True)
    if comp.returncode:
        print((comp.stdout + comp.stderr)[-5000:])
        return comp.returncode
    run = subprocess.run([vvp, str(executable)], cwd=REPO, capture_output=True, text=True)
    log = (run.stdout or "") + (run.stderr or "")
    print(log[-6000:])
    passed = run.returncode == 0 and "TB_RESULT: PASS" in log
    sample_count = len(re.findall(r"SAMPLE_RESULT index=\d+ byte_count=\d+ pass=1", log))
    if model_name:
        artifact_path = REPO / "bench" / "results" / f"_buffer_resident_{model_name}_rtl20.json"
        if artifact_path.is_file():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            replay = artifact["rtl_uart_replay"]
            perf = re.findall(r"SAMPLE_PERF_PROGRAM_CYCLES index=(\d+) cycles=(\d+)", log)
            cold = re.search(r"COLD_CORE_CYCLES_INCLUDING_WEIGHT_PROGRAM_INPUT_FILL=(\d+)", log)
            steady = re.findall(r"STEADY_CORE_CYCLES_INCLUDING_INPUT_FILL index=(\d+) cycles=(\d+)", log)
            replay["rtl_perf_program_cycles_per_inference"] = [
                {"sample_index": int(i), "cycles": int(c)} for i, c in perf
            ]
            replay["cold_core_cycles_including_weight_program_input_fill"] = int(cold.group(1)) if cold else None
            replay["steady_core_cycles_including_repeat_and_input_fill"] = [
                {"sample_index": int(i), "cycles": int(c)} for i, c in steady
            ]
            replay["log_tail"] = log[-5000:]
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"SUMMARY passed={passed} samples={sample_count}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
