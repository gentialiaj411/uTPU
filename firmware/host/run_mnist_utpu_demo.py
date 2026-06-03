import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board_config import BoardConfig
from isa_encoder import IsaConfig
from isa_simulator import simulate_mem_file
from lowering_fused_mlp_utpu import lower_fused_mlp_program_utpu
from program_loader import ProgramLoader


SEED = 20260601
REPORT_PATH = os.path.join("build", "reports", "mnist_utpu_demo.json")
EXPECTED_SVH_PATH = os.path.join("build", "test_vectors", "mnist_utpu_expected.svh")
TRACE_LOG_PATH = os.path.join("build", "reports", "rtl_mnist_utpu_trace.log")
DATA_ROOT = os.path.join("software", "data")
ALPHA = 0.25


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _data_path(name: str) -> str:
    return os.path.join(_repo_root(), DATA_ROOT, name)


def _resolve_iverilog_tools() -> Tuple[Optional[str], Optional[str]]:
    iv = shutil.which("iverilog")
    vv = shutil.which("vvp")
    if iv and vv:
        return iv, vv
    candidates = [
        (r"C:\iverilog\bin\iverilog.exe", r"C:\iverilog\bin\vvp.exe"),
        (r"C:\Program Files\Icarus Verilog\bin\iverilog.exe", r"C:\Program Files\Icarus Verilog\bin\vvp.exe"),
    ]
    for iv_path, vv_path in candidates:
        if os.path.exists(iv_path) and os.path.exists(vv_path):
            return iv_path, vv_path
    return None, None


def _pack_int4_words(values: List[int]) -> List[int]:
    vals = [int(v) for v in values]
    while len(vals) % 4 != 0:
        vals.append(0)
    words: List[int] = []
    for i in range(0, len(vals), 4):
        word = 0
        for j in range(4):
            word |= (vals[i + j] & 0xF) << (4 * j)
        words.append(word & 0xFFFF)
    return words


def _unpack_int4_words(words: List[int]) -> List[int]:
    out: List[int] = []
    for word in words:
        for shift in range(0, 16, 4):
            nib = (int(word) >> shift) & 0xF
            out.append(nib - 16 if nib & 0x8 else nib)
    return out


def _clip_int4(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), -8, 7).astype(np.int32)


class BiaslessLeakyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64, bias=False)
        self.fc2 = nn.Linear(64, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=ALPHA)
        return self.fc2(x)


def _load_and_downsample() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xtr = np.load(_data_path("mnist_14x14_train.npy")).astype(np.float32)
    ytr = np.load(_data_path("train_labels.npy")).astype(np.int64)
    xte = np.load(_data_path("mnist_14x14_test.npy")).astype(np.float32)
    yte = np.load(_data_path("test_labels.npy")).astype(np.int64)
    xt = torch.from_numpy(xtr[:, None, :, :])
    xe = torch.from_numpy(xte[:, None, :, :])
    xtr8 = F.adaptive_avg_pool2d(xt, (8, 8)).squeeze(1).numpy().reshape(len(xtr), -1)
    xte8 = F.adaptive_avg_pool2d(xe, (8, 8)).squeeze(1).numpy().reshape(len(xte), -1)
    mu = xtr8.mean(axis=0, keepdims=True)
    sig = xtr8.std(axis=0, keepdims=True) + 1e-6
    return (xtr8 - mu) / sig, ytr, (xte8 - mu) / sig, yte


def _train_float_model(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray) -> BiaslessLeakyMLP:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = BiaslessLeakyMLP().eval()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb = torch.from_numpy(xtr)
    yb = torch.from_numpy(ytr)
    batch = 256
    for epoch in range(4):
        perm = torch.randperm(len(xb), generator=torch.Generator().manual_seed(SEED + epoch))
        for i in range(0, len(xb), batch):
            idx = perm[i:i + batch]
            logits = model(xb[idx])
            loss = F.cross_entropy(logits, yb[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _quantize_weights_per_row(weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_rows: List[np.ndarray] = []
    scales: List[float] = []
    for row in weight:
        scale = max(float(np.max(np.abs(row))) / 7.0, 1e-6)
        q_rows.append(np.clip(np.rint(row / scale), -8, 7).astype(np.int8))
        scales.append(scale)
    return np.stack(q_rows, axis=0), np.asarray(scales, dtype=np.float32)


def _quantize_input_vector(x: np.ndarray) -> Tuple[np.ndarray, float]:
    scale = max(float(np.max(np.abs(x))) / 7.0, 1e-6)
    return np.clip(np.rint(x / scale), -8, 7).astype(np.int32), float(scale)


def _proxy_quant_accuracy(model: BiaslessLeakyMLP, xte: np.ndarray, yte: np.ndarray) -> float:
    w1 = model.fc1.weight.detach().cpu().numpy()
    w2 = model.fc2.weight.detach().cpu().numpy()
    q1, s1 = _quantize_weights_per_row(w1)
    q2, s2 = _quantize_weights_per_row(w2)
    correct = 0
    for x, label in zip(xte, yte):
        xq, xscale = _quantize_input_vector(x)
        h = (xq @ q1.T).astype(np.float32) * (xscale * s1[None, :])
        h = np.where(h < 0, h * ALPHA, h)
        hscale = max(float(np.max(np.abs(h))) / 7.0, 1e-6)
        hq = _clip_int4(h / hscale)
        out = (hq @ q2.T).astype(np.float32) * (hscale * s2[None, :])
        pred = int(np.argmax(out))
        correct += int(pred == int(label))
    return float(correct / len(yte))


def _raw_integer_reference(
    x: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> List[int]:
    xq, _ = _quantize_input_vector(x)
    h = xq @ q1.T
    h = _clip_int4(h)
    h = np.where(h < 0, (h >> 2), h)
    out = h @ q2.T
    out = _clip_int4(out)
    return _pack_int4_words(out.tolist())


def _pack_words_to_fetch_bytes(words: List[int]) -> List[int]:
    fetch_bytes: List[int] = []
    for word in words:
        fetch_bytes.extend(list(int(word).to_bytes(2, byteorder="little", signed=False)))
    return fetch_bytes


def _choose_cases(xte: np.ndarray, yte: np.ndarray) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    seen_labels = set()
    for idx, (x, y) in enumerate(zip(xte, yte)):
        if int(y) in seen_labels:
            continue
        chosen.append({"index": int(idx), "label": int(y), "x": x})
        seen_labels.add(int(y))
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        for idx, (x, y) in enumerate(zip(xte, yte)):
            if any(c["index"] == idx for c in chosen):
                continue
            chosen.append({"index": int(idx), "label": int(y), "x": x})
            if len(chosen) == 3:
                break
    return chosen


def _build_vectors(model: BiaslessLeakyMLP, xte: np.ndarray, yte: np.ndarray, board: BoardConfig) -> Dict[str, Any]:
    cfg = IsaConfig(address_width=10, compute_data_width=4)
    w1 = model.fc1.weight.detach().cpu().numpy()
    w2 = model.fc2.weight.detach().cpu().numpy()
    q1, s1 = _quantize_weights_per_row(w1)
    q2, s2 = _quantize_weights_per_row(w2)
    cases = _choose_cases(xte, yte)
    vectors: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(cases, start=1):
        xq, _ = _quantize_input_vector(case["x"])
        program = lower_fused_mlp_program_utpu(
            fc1_weights_int4=q1,
            fc2_weights_int4=q2,
            input_activations_int4=xq,
            residual_input_int4=None,
            array_size=16,
            fc1_apply_relu=True,
            fc2_apply_relu=False,
            apply_quant=True,
            result_addr=ProgramLoader.BUFFER_SECTION_C,
            residual_addr=ProgramLoader.BUFFER_SECTION_D,
            num_pe=1,
            cfg=cfg,
        )
        program_bytes = program["program"]
        program_words = len(program_bytes) // 2
        mem_path = os.path.join("build", "test_vectors", f"mnist_case{case_idx}_program.mem")
        os.makedirs(os.path.dirname(mem_path), exist_ok=True)
        with open(mem_path, "w", encoding="utf-8") as f:
            for i in range(0, len(program_bytes), 2):
                word = int.from_bytes(program_bytes[i:i + 2], "little")
                f.write(f"{word:04x}\n")

        expected_words = _raw_integer_reference(case["x"], q1, q2)
        expected_bytes = _pack_words_to_fetch_bytes(expected_words)
        if len(expected_bytes) != 6:
            raise RuntimeError(f"Expected 6 output bytes, got {len(expected_bytes)}")

        isa_result = simulate_mem_file(
            mem_path,
            array_size=16,
            buffer_size=1024,
            cfg=cfg,
            accumulator_data_width=32,
        )
        isa_bytes = list(isa_result.fetch_bytes)
        if isa_bytes != expected_bytes:
            raise RuntimeError(
                f"Reference/ISA mismatch for case{case_idx}: ref={expected_bytes}, isa={isa_bytes}"
            )

        vectors.append(
            {
                "name": f"case{case_idx}_mnist8x8_label{case['label']}",
                "label": int(case["label"]),
                "index": int(case["index"]),
                "program_mem": mem_path,
                "program_words": int(program_words),
                "expected_fetch_bytes": expected_bytes,
                "expected_outputs": expected_words,
                "fetch_n": len(expected_bytes),
                "program": program_bytes,
                "isa_fetch_bytes": isa_bytes,
            }
        )

    return {
        "array_size": 16,
        "cfg": {"address_width": cfg.address_width, "compute_data_width": cfg.compute_data_width},
        "board": board.as_dict(),
        "cases": vectors,
        "quantized_weights": {
            "fc1_shape": list(q1.shape),
            "fc2_shape": list(q2.shape),
        },
    }


def _write_expected_svh(vectors: Dict[str, Any]) -> None:
    lines = [
        '`ifndef MNIST_UTPU_EXPECTED_SVH',
        '`define MNIST_UTPU_EXPECTED_SVH',
        f'`define TB_PROG_DEPTH {int(vectors["board"]["prog_depth"])}',
        '',
    ]
    for idx, case in enumerate(vectors["cases"], start=1):
        fetch_n = len(case["expected_fetch_bytes"])
        lines.append(f'`define CASE{idx}_NAME "{case["name"]}"')
        mem_path = case["program_mem"].replace("\\", "\\\\").replace("/", "\\\\")
        lines.append(f'`define CASE{idx}_MEM "{mem_path}"')
        lines.append(f'`define CASE{idx}_WORDS {case["program_words"]}')
        lines.append(f'`define CASE{idx}_FETCH_N {fetch_n}')
        for byte_idx, byte in enumerate(case["expected_fetch_bytes"]):
            lines.append(f"`define CASE{idx}_EXP_BYTE_{byte_idx} 8'h{byte:02x}")
        lines.append('')
    lines.append('`endif')
    os.makedirs(os.path.dirname(EXPECTED_SVH_PATH), exist_ok=True)
    with open(EXPECTED_SVH_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _parse_rtl_case_bytes(rtl_log: str) -> Dict[str, List[int]]:
    parsed: Dict[str, List[int]] = {}
    for line in rtl_log.splitlines():
        line = line.strip()
        if "_ACTUAL_BYTES=" not in line:
            continue
        name, payload = line.split("=", 1)
        values = [p for p in payload.split(",") if p]
        parsed[name] = [int(v, 16) for v in values]
    return parsed


def _run_rtl(vectors: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        return False, "iverilog/vvp binaries not found", {"rtl_sim_executed": False, "rtl_sim_passed": False}
    build_dir = os.path.join(_repo_root(), "build", "rtl_sim")
    os.makedirs(build_dir, exist_ok=True)
    out_vvp = os.path.join(build_dir, "tb_mnist_utpu_program.out")
    srcs = [
        "rtl/tb/tb_mnist_utpu_program.sv",
        "rtl/tb/xpm_memory_sdpram_stub.sv",
        "rtl/top/top.sv",
        "rtl/memory/instr_bram.sv",
        "rtl/PEArray/pe_controller.sv",
        "rtl/PEArray/pe_array.sv",
        "rtl/PEArray/pe.sv",
        "rtl/quantizer/quantizer.sv",
        "rtl/quantizer/quantizer_array.sv",
        "rtl/LeakyReLU/leaky_relu.sv",
        "rtl/LeakyReLU/leaky_relu_array.sv",
        "rtl/unified_buffer/unified_buffer.sv",
        "rtl/fifo/fifo_rx.sv",
        "rtl/fifo/fifo_tx.sv",
        "rtl/UART/uart.sv",
        "rtl/UART/uart_receiver.sv",
        "rtl/UART/uart_transmitter.sv",
        "rtl/UART/clk_divider.sv",
    ]
    srcs_abs = [os.path.join(_repo_root(), s) for s in srcs]
    env = os.environ.copy()
    env["TMP"] = build_dir
    env["TEMP"] = build_dir
    env["TMPDIR"] = build_dir
    c = subprocess.run([iv_bin, "-g2012", "-DICARUS", "-o", out_vvp] + srcs_abs, cwd=_repo_root(), env=env, capture_output=True, text=True)
    if c.returncode != 0:
        return False, (c.stdout or "") + "\n" + (c.stderr or ""), {"rtl_sim_executed": True, "rtl_sim_passed": False}
    r = subprocess.run([vv_bin, out_vvp], cwd=_repo_root(), env=env, capture_output=True, text=True)
    ok = (r.returncode == 0) and ("TB_RESULT: PASS" in (r.stdout or ""))
    return ok, (r.stdout or "") + "\n" + (r.stderr or ""), {"rtl_sim_executed": True, "rtl_sim_passed": bool(ok)}


def run_demo(output_json: str = REPORT_PATH) -> Dict[str, Any]:
    root = _repo_root()
    os.chdir(root)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    xtr, ytr, xte, yte = _load_and_downsample()
    model = _train_float_model(xtr, ytr, xte, yte)
    float_acc = float((model(torch.from_numpy(xte)).argmax(1).numpy() == yte).mean())
    quant_acc = _proxy_quant_accuracy(model, xte, yte)

    board = None
    vectors = None
    for candidate in BoardConfig.reference_set():
        vectors = _build_vectors(model, xte, yte, candidate)
        if candidate.fits(max(case["program_words"] for case in vectors["cases"])):
            board = candidate
            break
    if board is None or vectors is None:
        board = BoardConfig.reference_set()[-1]
        vectors = _build_vectors(model, xte, yte, board)

    _write_expected_svh(vectors)
    rtl_ok, rtl_log, rtl_meta = _run_rtl(vectors)
    os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)
    with open(TRACE_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(rtl_log)
    rtl_case_bytes = _parse_rtl_case_bytes(rtl_log)

    cases_report: List[Dict[str, Any]] = []
    for idx, case in enumerate(vectors["cases"], start=1):
        case_key = f"case{idx}"
        rtl_bytes = rtl_case_bytes.get(case["name"] + "_ACTUAL_BYTES")
        cases_report.append(
            {
                "name": case["name"],
                "label": case["label"],
                "index": case["index"],
                "program_words": case["program_words"],
                "expected_fetch_bytes": case["expected_fetch_bytes"],
                "isa_fetch_bytes": case["isa_fetch_bytes"],
                "rtl_fetch_bytes": rtl_bytes,
                "bit_exact_vs_reference": case["expected_fetch_bytes"] == case["isa_fetch_bytes"],
                "isa_rtl_bitmatch": bool(rtl_ok and rtl_bytes == case["expected_fetch_bytes"]),
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_name": "mnist_8x8_64x64x10_leaky_mlp",
        "seed": SEED,
        "dataset": {
            "name": "mnist_14x14_local_downsampled_to_8x8",
            "train_size": int(len(xtr)),
            "test_size": int(len(xte)),
            "downsample_shape": [8, 8],
            "input_dim": 64,
            "output_dim": 10,
        },
        "float_acc": float_acc,
        "quant_acc": quant_acc,
        "board_config": vectors["board"],
        "instruction_bram_words": int(max(case["program_words"] for case in vectors["cases"])),
        "fits_instruction_bram": bool(board.fits(max(case["program_words"] for case in vectors["cases"]))),
        "bit_exact_vs_reference": bool(all(case["expected_fetch_bytes"] == case["isa_fetch_bytes"] for case in vectors["cases"])),
        "isa_rtl_bitmatch": bool(rtl_ok and all(case["rtl_fetch_bytes"] == case["expected_fetch_bytes"] for case in cases_report)),
        "rtl_sim_executed": bool(rtl_meta["rtl_sim_executed"]),
        "rtl_sim_passed": bool(rtl_meta["rtl_sim_passed"]),
        "shapes": {
            "fc1": [64, 64],
            "fc2": [64, 10],
            "array_size": 16,
        },
        "cases": cases_report,
        "quantization": {
            "activation_mode": "per-sample symmetric int4",
            "weight_mode": "per-row symmetric int4",
            "relu_mode": "leaky_relu(alpha=0.25)",
            "biases": False,
        },
        "rtl_trace_log_path": TRACE_LOG_PATH,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    md_path = os.path.join(os.path.dirname(output_json), "mnist_utpu_demo.md")
    md = [
        "# MNIST 8x8 uTPU Demo",
        "",
        f"- float_acc: {report['float_acc']}",
        f"- quant_acc: {report['quant_acc']}",
        f"- board_config: {report['board_config']['name']}",
        f"- instruction_bram_words: {report['instruction_bram_words']}",
        f"- fits_instruction_bram: {report['fits_instruction_bram']}",
        f"- bit_exact_vs_reference: {report['bit_exact_vs_reference']}",
        f"- isa_rtl_bitmatch: {report['isa_rtl_bitmatch']}",
        "",
        "## Cases",
    ]
    for case in cases_report:
        md.append(
            f"- {case['name']}: label={case['label']} words={case['program_words']} "
            f"bit_exact={case['bit_exact_vs_reference']} rtl={case['isa_rtl_bitmatch']}"
        )
    if rtl_log:
        md.extend(["", "## RTL Log", "```text", rtl_log.strip(), "```"])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps({
        "output_json": output_json,
        "quant_acc": report["quant_acc"],
        "float_acc": report["float_acc"],
        "fits_instruction_bram": report["fits_instruction_bram"],
        "bit_exact_vs_reference": report["bit_exact_vs_reference"],
        "isa_rtl_bitmatch": report["isa_rtl_bitmatch"],
        "rtl_sim_executed": rtl_meta["rtl_sim_executed"],
        "rtl_sim_passed": rtl_meta["rtl_sim_passed"],
    }, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the labeled MNIST 8x8 uTPU demo")
    parser.add_argument("--output-json", default=REPORT_PATH)
    args = parser.parse_args()
    report = run_demo(args.output_json)
    return 0 if report["bit_exact_vs_reference"] and report["isa_rtl_bitmatch"] else 1


if __name__ == "__main__":
    sys.exit(main())
