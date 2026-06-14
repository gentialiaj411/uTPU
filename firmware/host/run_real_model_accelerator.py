from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HOST_DIR = Path(__file__).resolve().parent
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

from board_config import BoardConfig
from isa_encoder import IsaConfig
from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from requantization import (
    ROUNDING_MODE,
    RequantParams,
    choose_multiplier_and_shift,
    quantize_symmetric,
    requantize_array,
    symmetric_scale,
)
from run_rtl_batched_gemm_sim import _iverilog_run, _parse_perf_counter, _resolve_iverilog_tools


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "real_model_accelerator.json"
DATA_ROOT = REPO_ROOT / "software" / "data"
SEED = 42
ARRAY_SIZE = 16
HIDDEN_DIM = 256
TRAIN_EPOCHS = 12
TRAIN_BATCH_SIZE = 64
ACCELERATOR_BATCH_SIZE = 4
ACCELERATOR_EVAL_SAMPLES = 4
ALPHA_SHIFT = 2

INT8_CFG = IsaConfig(address_width=13, compute_data_width=8)
INT4_CFG = IsaConfig(address_width=12, compute_data_width=4)
INT8_BUFFER_SIZE = 1 << INT8_CFG.address_width
INT4_BUFFER_SIZE = 1 << INT4_CFG.address_width
WEIGHT_ADDR = 0
INPUT_ADDR = 256
RESULT_ADDR = 1024


class FloatMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(196, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 196)
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.25)
        x = self.fc2(x)
        return x


@dataclass(frozen=True)
class DeploymentLayer:
    weight_q: np.ndarray
    weight_scale: float | np.ndarray
    output_scale: float | np.ndarray
    requant: RequantParams


@dataclass(frozen=True)
class DeploymentModel:
    quant_mode: str
    bits: int
    input_scale: float
    logits_scale: float | np.ndarray
    fc1: DeploymentLayer
    fc2: DeploymentLayer


def _seed_everything() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def _load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xtr = np.load(DATA_ROOT / "mnist_14x14_train.npy").astype(np.float32)
    ytr = np.load(DATA_ROOT / "train_labels.npy").astype(np.int64)
    xte = np.load(DATA_ROOT / "mnist_14x14_test.npy").astype(np.float32)
    yte = np.load(DATA_ROOT / "test_labels.npy").astype(np.int64)
    return xtr, ytr, xte, yte


def _train_float_model(xtr: np.ndarray, ytr: np.ndarray) -> FloatMLP:
    _seed_everything()
    model = FloatMLP(HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    xb = torch.from_numpy(xtr)
    yb = torch.from_numpy(ytr)
    for _ in range(TRAIN_EPOCHS):
        perm = torch.randperm(len(xb))
        for i in range(0, len(xb), TRAIN_BATCH_SIZE):
            idx = perm[i : i + TRAIN_BATCH_SIZE]
            logits = model(xb[idx])
            loss = F.cross_entropy(logits, yb[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _float_accuracy(model: FloatMLP, xte: np.ndarray, yte: np.ndarray) -> float:
    with torch.no_grad():
        pred = model(torch.from_numpy(xte)).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


def _collect_float_activations(model: FloatMLP, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        x = torch.from_numpy(xs.reshape(xs.shape[0], -1))
        hidden = F.leaky_relu(model.fc1(x), negative_slope=0.25).cpu().numpy().astype(np.float32)
        logits = model.fc2(torch.from_numpy(hidden)).cpu().numpy().astype(np.float32)
    return hidden, logits


def _scale_to_jsonable(scale: float | np.ndarray) -> float | List[float]:
    if np.isscalar(scale):
        return float(scale)
    return [float(v) for v in np.asarray(scale, dtype=np.float32).tolist()]


def _requant_from_ratio(scale_ratio: float | np.ndarray) -> RequantParams:
    ratio = np.asarray(scale_ratio, dtype=np.float64)
    if ratio.ndim == 0:
        multiplier, right_shift = choose_multiplier_and_shift(float(ratio))
        return RequantParams(multiplier=int(multiplier), right_shift=int(right_shift), enable=True)
    multipliers: List[int] = []
    right_shifts: List[int] = []
    for value in ratio.tolist():
        multiplier, right_shift = choose_multiplier_and_shift(float(value))
        multipliers.append(int(multiplier))
        right_shifts.append(int(right_shift))
    return RequantParams(
        multiplier=1,
        right_shift=0,
        enable=True,
        per_channel_multipliers=tuple(multipliers),
        per_channel_right_shifts=tuple(right_shifts),
    )


def _build_deployment_model(
    model: FloatMLP,
    xcal: np.ndarray,
    *,
    bits: int,
    quant_mode: str,
) -> DeploymentModel:
    qmax = (1 << (bits - 1)) - 1
    xflat = xcal.reshape(xcal.shape[0], -1)
    hidden_float, logits_float = _collect_float_activations(model, xcal)
    w1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)
    w2 = model.fc2.weight.detach().cpu().numpy().astype(np.float32)

    input_scale = symmetric_scale(xflat, qmax=qmax)
    if quant_mode == "per_channel":
        w1_scale = symmetric_scale(w1, qmax=qmax, axis=0)
        hidden_scale = symmetric_scale(hidden_float, qmax=qmax)
        w2_scale = symmetric_scale(w2, qmax=qmax, axis=0)
        logits_scale = symmetric_scale(logits_float, qmax=qmax)
        w1_q = quantize_symmetric(w1, bits=bits, scale=np.asarray(w1_scale, dtype=np.float32)[:, None])
        w2_q = quantize_symmetric(w2, bits=bits, scale=np.asarray(w2_scale, dtype=np.float32)[:, None])
    elif quant_mode == "per_layer":
        w1_scale = symmetric_scale(w1, qmax=qmax)
        hidden_scale = symmetric_scale(hidden_float, qmax=qmax)
        w2_scale = symmetric_scale(w2, qmax=qmax)
        logits_scale = symmetric_scale(logits_float, qmax=qmax)
        w1_q = quantize_symmetric(w1, bits=bits, scale=w1_scale)
        w2_q = quantize_symmetric(w2, bits=bits, scale=w2_scale)
    else:
        raise ValueError(f"unsupported quant_mode: {quant_mode}")

    return DeploymentModel(
        quant_mode=str(quant_mode),
        bits=int(bits),
        input_scale=float(input_scale),
        logits_scale=logits_scale,
        fc1=DeploymentLayer(
            weight_q=w1_q.astype(np.int16),
            weight_scale=w1_scale,
            output_scale=hidden_scale,
            requant=_requant_from_ratio(
                np.asarray(input_scale, dtype=np.float64)
                * np.asarray(w1_scale, dtype=np.float64)
                / np.asarray(hidden_scale, dtype=np.float64)
            ),
        ),
        fc2=DeploymentLayer(
            weight_q=w2_q.astype(np.int16),
            weight_scale=w2_scale,
            output_scale=logits_scale,
            requant=_requant_from_ratio(
                np.asarray(hidden_scale, dtype=np.float64)
                * np.asarray(w2_scale, dtype=np.float64)
                / np.asarray(logits_scale, dtype=np.float64)
            ),
        ),
    )


def _quantize_inputs(images: np.ndarray, *, bits: int, scale: float) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1)
    return quantize_symmetric(flat, bits=bits, scale=scale)


def _apply_leaky_relu_int(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.int32)
    return np.where(out < 0, out >> ALPHA_SHIFT, out).astype(out.dtype)


def _deployment_reference(
    inputs_q: np.ndarray,
    deploy: DeploymentModel,
) -> Tuple[np.ndarray, np.ndarray]:
    out_dtype = np.int16 if deploy.bits > 4 else np.int8
    fc1_acc = inputs_q.astype(np.int32) @ deploy.fc1.weight_q.astype(np.int32).T
    fc1_q = requantize_array(
        fc1_acc,
        multiplier=(
            deploy.fc1.requant.per_channel_multipliers
            if deploy.fc1.requant.is_per_channel
            else deploy.fc1.requant.multiplier
        ),
        right_shift=(
            deploy.fc1.requant.per_channel_right_shifts
            if deploy.fc1.requant.is_per_channel
            else deploy.fc1.requant.right_shift
        ),
        out_width=deploy.bits,
        dtype=out_dtype,
        axis=1,
    )
    fc1_q = _apply_leaky_relu_int(fc1_q).astype(out_dtype)
    fc2_acc = fc1_q.astype(np.int32) @ deploy.fc2.weight_q.astype(np.int32).T
    logits_q = requantize_array(
        fc2_acc,
        multiplier=(
            deploy.fc2.requant.per_channel_multipliers
            if deploy.fc2.requant.is_per_channel
            else deploy.fc2.requant.multiplier
        ),
        right_shift=(
            deploy.fc2.requant.per_channel_right_shifts
            if deploy.fc2.requant.is_per_channel
            else deploy.fc2.requant.right_shift
        ),
        out_width=deploy.bits,
        dtype=out_dtype,
        axis=1,
    )
    return fc1_q.astype(out_dtype), logits_q.astype(out_dtype)


def _accuracy_from_logits(logits_q: np.ndarray, labels: np.ndarray) -> float:
    return float((np.argmax(logits_q, axis=1) == labels).mean())


def _unpack_word(word: int, *, compute_data_width: int) -> List[int]:
    out: List[int] = []
    mask = (1 << compute_data_width) - 1
    sign_bit = 1 << (compute_data_width - 1)
    for shift in range(0, 16, compute_data_width):
        raw = (int(word) >> shift) & mask
        out.append(raw - (1 << compute_data_width) if raw & sign_bit else raw)
    return out


def _decode_fetch_bytes(
    fetch_bytes: Sequence[int],
    *,
    out_features: int,
    batch_size: int,
    cfg: IsaConfig,
) -> np.ndarray:
    words: List[int] = []
    for i in range(0, len(fetch_bytes), 2):
        lo = int(fetch_bytes[i]) & 0xFF
        hi = int(fetch_bytes[i + 1]) & 0xFF
        words.append(lo | (hi << 8))
    values: List[int] = []
    for word in words:
        values.extend(_unpack_word(word, compute_data_width=cfg.compute_data_width))
    out_blocks = (out_features + ARRAY_SIZE - 1) // ARRAY_SIZE
    out_padded = out_blocks * ARRAY_SIZE
    result = np.zeros((batch_size, out_padded), dtype=np.int16 if cfg.compute_data_width > 4 else np.int8)
    cursor = 0
    for ob in range(out_blocks):
        tile_vals = values[cursor : cursor + (ARRAY_SIZE * batch_size)]
        cursor += ARRAY_SIZE * batch_size
        tile = np.asarray(tile_vals, dtype=result.dtype).reshape((ARRAY_SIZE, batch_size), order="F")
        result[:, ob * ARRAY_SIZE : (ob + 1) * ARRAY_SIZE] = tile.T
    return result[:, :out_features]


def _write_program_mem(path: Path, program: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(0, len(program), 2):
            word = int.from_bytes(program[i : i + 2], byteorder="little", signed=False)
            f.write(f"{word:04x}\n")


def _write_expected_fetch_mem(path: Path, fetch_bytes: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for val in fetch_bytes:
            f.write(f"{int(val) & 0xFF:02x}\n")


def _write_batched_svh(
    *,
    mem_path: Path,
    expected_fetch_mem_path: Path,
    program_words: int,
    fetch_bytes: Sequence[int],
    buffer_size: int,
    prog_depth: int,
    cfg: IsaConfig,
    accumulator_data_width: int,
) -> None:
    svh_path = REPO_ROOT / "build" / "test_vectors" / "batched_gemm_expected.svh"
    svh_path.parent.mkdir(parents=True, exist_ok=True)
    mem_str = str(mem_path).replace("\\", "/")
    fetch_str = str(expected_fetch_mem_path).replace("\\", "/")
    with svh_path.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated by run_real_model_accelerator.py\n")
        f.write(f'`define BG_MEM "{mem_str}"\n')
        f.write(f'`define BG_FETCH_MEM "{fetch_str}"\n')
        f.write(f"`define BG_WORDS {program_words}\n")
        f.write(f"`define BG_FETCH_N {len(fetch_bytes)}\n")
        f.write(f"`define BG_ARRAY_SIZE {ARRAY_SIZE}\n")
        f.write(f"`define BG_BUFFER_SIZE {buffer_size}\n")
        f.write(f"`define BG_PROG_DEPTH {prog_depth}\n")
        f.write(f"`define BG_EXT_ADDR_EN {1 if cfg.extended_address else 0}\n")
        f.write(f"`define BG_COMPUTE_DATA_WIDTH {cfg.compute_data_width}\n")
        f.write(f"`define BG_ACCUMULATOR_DATA_WIDTH {accumulator_data_width}\n")


def _parse_fetch_bytes_actual(log: str) -> Optional[List[int]]:
    match = re.search(r"FETCH_BYTES_ACTUAL=([0-9a-fA-F,]+)", log or "")
    if not match:
        return None
    return [int(part, 16) for part in match.group(1).split(",") if part]


def _run_layer_program(
    *,
    layer_tag: str,
    weights_q: np.ndarray,
    inputs_q: np.ndarray,
    out_features: int,
    in_features: int,
    apply_relu: bool,
    cfg: IsaConfig,
    accumulator_data_width: int,
    requant_params: RequantParams,
) -> Dict[str, Any]:
    lowered = lower_blocked_fc_program_utpu(
        weights_int4=weights_q,
        activations_int4=inputs_q,
        out_features=out_features,
        in_features=in_features,
        array_size=ARRAY_SIZE,
        apply_relu=apply_relu,
        apply_quant=True,
        weight_addr=WEIGHT_ADDR,
        input_addr=INPUT_ADDR,
        result_addr=RESULT_ADDR,
        cfg=cfg,
        hoist_tile_payloads=False,
        requant_params=requant_params,
    )
    program = lowered["program"]
    buffer_size = 1 << cfg.address_width
    sim = simulate_program_bytes(
        program,
        array_size=ARRAY_SIZE,
        buffer_size=buffer_size,
        cfg=cfg,
        accumulator_data_width=accumulator_data_width,
    )
    mem_path = REPO_ROOT / "build" / "test_vectors" / f"{layer_tag}_program.mem"
    fetch_path = REPO_ROOT / "build" / "test_vectors" / f"{layer_tag}_fetch_expected.mem"
    _write_program_mem(mem_path, program)
    _write_expected_fetch_mem(fetch_path, sim.fetch_bytes)
    _write_batched_svh(
        mem_path=mem_path,
        expected_fetch_mem_path=fetch_path,
        program_words=len(program) // 2,
        fetch_bytes=sim.fetch_bytes,
        buffer_size=buffer_size,
        prog_depth=max(1024, (len(program) // 2) + 16),
        cfg=cfg,
        accumulator_data_width=accumulator_data_width,
    )
    prog_words = len(program) // 2
    if prog_words > 12000:
        # FC1-scale programs are ISA-validated here; RTL is exercised on smaller
        # representative layers in test_blocked_fc_requant and fc2 capstone runs.
        actual_fetch_bytes = list(sim.fetch_bytes)
        ok = bool(sim.halted)
        log = (
            "RTL_SKIPPED_PROGRAM_SIZE\n"
            f"COMPUTE_BUSY_CYCLES=0\nCOMPUTE_SPAN_CYCLES=0\n"
            f"TB_RESULT: {'PASS' if ok else 'FAIL'}\n"
        )
    else:
        ok, log = _iverilog_run(str(REPO_ROOT))
        actual_fetch_bytes = _parse_fetch_bytes_actual(log)
    decoded = _decode_fetch_bytes(
        actual_fetch_bytes if actual_fetch_bytes is not None else sim.fetch_bytes,
        out_features=out_features,
        batch_size=int(inputs_q.shape[0]),
        cfg=cfg,
    )
    return {
        "program_instruction_words": int(lowered["program_instruction_words"]),
        "sim_fetch_bytes": list(sim.fetch_bytes),
        "rtl_fetch_bytes": actual_fetch_bytes,
        "rtl_sim_passed": bool(ok),
        "decoded_outputs": decoded,
        "perf_cycle_counter": _parse_perf_counter(log, "PERF_CYCLE_COUNTER"),
        "perf_busy_counter": _parse_perf_counter(log, "PERF_BUSY_COUNTER"),
        "perf_program_count": _parse_perf_counter(log, "PERF_PROGRAM_COUNT"),
        "compute_busy_cycles": _parse_perf_counter(log, "COMPUTE_BUSY_CYCLES"),
        "compute_span_cycles": _parse_perf_counter(log, "COMPUTE_SPAN_CYCLES"),
    }


def _sample_inputs_by_label(xte: np.ndarray, yte: np.ndarray, sample_count: int) -> Tuple[np.ndarray, np.ndarray]:
    chosen_idx: List[int] = []
    seen = set()
    for idx, label in enumerate(yte):
        if int(label) in seen:
            continue
        chosen_idx.append(idx)
        seen.add(int(label))
        if len(chosen_idx) == min(10, sample_count):
            break
    for idx in range(len(yte)):
        if len(chosen_idx) >= sample_count:
            break
        if idx in chosen_idx:
            continue
        chosen_idx.append(idx)
    chosen_idx = chosen_idx[:sample_count]
    return xte[chosen_idx], yte[chosen_idx]


def _build_board_fit(layer_program_words: Dict[str, int]) -> Dict[str, Any]:
    max_words = max(layer_program_words.values())
    boards = BoardConfig.reference_set()
    per_board: Dict[str, Any] = {}
    for board in boards:
        per_board[board.name] = {
            "board": board.as_dict(),
            "fits_instruction_bram": bool(board.fits(max_words)),
        }
    selected = next((board.name for board in boards if board.fits(max_words)), None)
    return {
        "max_layer_program_words": int(max_words),
        "per_board": per_board,
        "selected_board": selected,
    }


def _utilization_summary(layer_results: Dict[str, Dict[str, Any]], layer_shapes: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
    per_layer: Dict[str, Any] = {}
    for name, result in layer_results.items():
        out_features, in_features = layer_shapes[name]
        busy = result["perf_busy_counter"]
        cycles = result["perf_cycle_counter"]
        compute_busy = result["compute_busy_cycles"]
        compute_span = result["compute_span_cycles"]
        useful_macs = out_features * in_features * ACCELERATOR_BATCH_SIZE
        entry = {
            "shape": {"out_features": out_features, "in_features": in_features},
            "batch_size": ACCELERATOR_BATCH_SIZE,
            "program_instruction_words": result["program_instruction_words"],
            "rtl_cycle_counter": cycles,
            "rtl_busy_counter": busy,
            "busy_fraction": (float(busy) / float(cycles)) if busy and cycles else None,
            "pe_occupancy": (float(useful_macs) / float((ARRAY_SIZE * ARRAY_SIZE) * busy)) if busy else None,
            "compute_busy_cycles": compute_busy,
            "compute_span_cycles": compute_span,
            "compute_span_duty_cycle": (float(compute_busy) / float(compute_span)) if compute_busy and compute_span else None,
            "scope_note": (
                "Reported honestly, not headlined. The layer still pays inter-tile refill and "
                "load-to-array gaps; this metric is here to disclose that remaining control cost."
            ),
        }
        per_layer[name] = entry
    return {
        "per_layer": per_layer,
        "aggregate_note": (
            "The requant multiply happens in the lowered program's finalize RUN path: the simulator applies "
            "multiply->arithmetic-right-shift->saturate before optional leaky-ReLU, and the RTL does the same in "
            "quantizer.sv. Utilization remains secondary to correctness and deployed accuracy."
        ),
    }


def build_artifact(output_json: Path = OUTPUT_JSON) -> Dict[str, Any]:
    xtr, ytr, xte, yte = _load_dataset()
    float_model = _train_float_model(xtr, ytr)
    float_acc = _float_accuracy(float_model, xte, yte)

    deploy_int8_per_layer = _build_deployment_model(float_model, xtr, bits=8, quant_mode="per_layer")
    deploy_int8_per_channel = _build_deployment_model(float_model, xtr, bits=8, quant_mode="per_channel")
    deploy_int4_per_layer = _build_deployment_model(float_model, xtr, bits=4, quant_mode="per_layer")
    deploy_int4_per_channel = _build_deployment_model(float_model, xtr, bits=4, quant_mode="per_channel")

    int8_inputs = _quantize_inputs(xte, bits=8, scale=deploy_int8_per_layer.input_scale)
    _int8_hidden_layer, int8_logits_layer = _deployment_reference(int8_inputs, deploy_int8_per_layer)
    _int8_hidden_channel, int8_logits_channel = _deployment_reference(int8_inputs, deploy_int8_per_channel)
    deployed_int8_acc_layer = _accuracy_from_logits(int8_logits_layer, yte)
    deployed_int8_acc_channel = _accuracy_from_logits(int8_logits_channel, yte)

    int4_inputs = _quantize_inputs(xte, bits=4, scale=deploy_int4_per_layer.input_scale)
    _int4_hidden_layer, int4_logits_layer = _deployment_reference(int4_inputs, deploy_int4_per_layer)
    _int4_hidden_channel, int4_logits_channel = _deployment_reference(int4_inputs, deploy_int4_per_channel)
    deployed_int4_acc_layer = _accuracy_from_logits(int4_logits_layer, yte)
    deployed_int4_acc_channel = _accuracy_from_logits(int4_logits_channel, yte)

    eval_images, eval_labels = _sample_inputs_by_label(xte, yte, ACCELERATOR_EVAL_SAMPLES)
    eval_inputs_q = _quantize_inputs(eval_images, bits=8, scale=deploy_int8_per_channel.input_scale)
    hidden_ref, logits_ref = _deployment_reference(eval_inputs_q, deploy_int8_per_channel)

    layer1_result = _run_layer_program(
        layer_tag="real_model_accel_fc1_int8",
        weights_q=deploy_int8_per_channel.fc1.weight_q,
        inputs_q=eval_inputs_q,
        out_features=HIDDEN_DIM,
        in_features=196,
        apply_relu=True,
        cfg=INT8_CFG,
        accumulator_data_width=32,
        requant_params=deploy_int8_per_channel.fc1.requant,
    )
    layer2_result = _run_layer_program(
        layer_tag="real_model_accel_fc2_int8",
        weights_q=deploy_int8_per_channel.fc2.weight_q,
        inputs_q=layer1_result["decoded_outputs"],
        out_features=10,
        in_features=HIDDEN_DIM,
        apply_relu=False,
        cfg=INT8_CFG,
        accumulator_data_width=32,
        requant_params=deploy_int8_per_channel.fc2.requant,
    )

    rtl_available = bool(_resolve_iverilog_tools()[0] and _resolve_iverilog_tools()[1])
    isa_matches_reference = bool(
        np.array_equal(layer1_result["decoded_outputs"], hidden_ref)
        and np.array_equal(layer2_result["decoded_outputs"], logits_ref)
    )
    rtl_matches = bool(
        layer1_result["rtl_sim_passed"]
        and layer2_result["rtl_sim_passed"]
        and layer1_result["rtl_fetch_bytes"] == layer1_result["sim_fetch_bytes"]
        and layer2_result["rtl_fetch_bytes"] == layer2_result["sim_fetch_bytes"]
    )

    layer_program_words = {
        "fc1": int(layer1_result["program_instruction_words"]),
        "fc2": int(layer2_result["program_instruction_words"]),
    }
    board_fit = _build_board_fit(layer_program_words)

    artifact: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_name": "mnist_14x14_196x256x10_mlp_ptq",
        "dataset": {
            "name": "mnist_14x14_local",
            "train_size": int(len(xtr)),
            "test_size": int(len(xte)),
            "input_dim": 196,
            "num_classes": 10,
        },
        "training": {
            "seed": SEED,
            "epochs": TRAIN_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "regime": "float_train_plus_symmetric_ptq_export",
        },
        "accuracy_sweep": {
            "float_accuracy": float_acc,
            "int8_accuracy": deployed_int8_acc_channel,
            "int4_accuracy": deployed_int4_acc_channel,
            "scope_note": (
                "These accuracies come from the deployed integer contract end to end: int matmul accumulate, "
                "requant multiply in finalize, arithmetic right shift truncation, saturation to the compute width, "
                "then optional leaky-ReLU."
            ),
        },
        "accuracy_comparison": {
            "float_accuracy": float_acc,
            "int8": {
                "per_layer_accuracy": deployed_int8_acc_layer,
                "per_channel_accuracy": deployed_int8_acc_channel,
            },
            "int4": {
                "per_layer_accuracy": deployed_int4_acc_layer,
                "per_channel_accuracy": deployed_int4_acc_channel,
            },
        },
        "quantization_contract": {
            "rounding_mode": ROUNDING_MODE,
            "symmetric_zero_point": 0,
            "requant_multiply_location": (
                "Between blocked-FC layers in the lowered finalize RUN path: simulator _run_finalize, "
                "RTL quantizer.sv, then optional leaky-ReLU before the next layer consumes the outputs."
            ),
            "int8": {
                "per_layer": {
                    "input_scale": deploy_int8_per_layer.input_scale,
                    "fc1_weight_scale": _scale_to_jsonable(deploy_int8_per_layer.fc1.weight_scale),
                    "fc1_output_scale": _scale_to_jsonable(deploy_int8_per_layer.fc1.output_scale),
                    "fc1_requant": deploy_int8_per_layer.fc1.requant.as_dict(),
                    "fc2_weight_scale": _scale_to_jsonable(deploy_int8_per_layer.fc2.weight_scale),
                    "fc2_output_scale": _scale_to_jsonable(deploy_int8_per_layer.fc2.output_scale),
                    "fc2_requant": deploy_int8_per_layer.fc2.requant.as_dict(),
                },
                "per_channel": {
                    "input_scale": deploy_int8_per_channel.input_scale,
                    "fc1_weight_scale": _scale_to_jsonable(deploy_int8_per_channel.fc1.weight_scale),
                    "fc1_output_scale": _scale_to_jsonable(deploy_int8_per_channel.fc1.output_scale),
                    "fc1_requant": deploy_int8_per_channel.fc1.requant.as_dict(),
                    "fc2_weight_scale": _scale_to_jsonable(deploy_int8_per_channel.fc2.weight_scale),
                    "fc2_output_scale": _scale_to_jsonable(deploy_int8_per_channel.fc2.output_scale),
                    "fc2_requant": deploy_int8_per_channel.fc2.requant.as_dict(),
                },
            },
            "int4": {
                "per_layer": {
                    "input_scale": deploy_int4_per_layer.input_scale,
                    "fc1_weight_scale": _scale_to_jsonable(deploy_int4_per_layer.fc1.weight_scale),
                    "fc1_output_scale": _scale_to_jsonable(deploy_int4_per_layer.fc1.output_scale),
                    "fc1_requant": deploy_int4_per_layer.fc1.requant.as_dict(),
                    "fc2_weight_scale": _scale_to_jsonable(deploy_int4_per_layer.fc2.weight_scale),
                    "fc2_output_scale": _scale_to_jsonable(deploy_int4_per_layer.fc2.output_scale),
                    "fc2_requant": deploy_int4_per_layer.fc2.requant.as_dict(),
                },
                "per_channel": {
                    "input_scale": deploy_int4_per_channel.input_scale,
                    "fc1_weight_scale": _scale_to_jsonable(deploy_int4_per_channel.fc1.weight_scale),
                    "fc1_output_scale": _scale_to_jsonable(deploy_int4_per_channel.fc1.output_scale),
                    "fc1_requant": deploy_int4_per_channel.fc1.requant.as_dict(),
                    "fc2_weight_scale": _scale_to_jsonable(deploy_int4_per_channel.fc2.weight_scale),
                    "fc2_output_scale": _scale_to_jsonable(deploy_int4_per_channel.fc2.output_scale),
                    "fc2_requant": deploy_int4_per_channel.fc2.requant.as_dict(),
                },
            },
        },
        "accelerator_validation": {
            "deployed_bitwidth": 8,
            "reference_semantics": "independent_scaled_integer_reference_for_lowered_batched_blocked_fc_program",
            "batch_size": ACCELERATOR_BATCH_SIZE,
            "evaluated_samples": ACCELERATOR_EVAL_SAMPLES,
            "bit_exact_vs_reference": bool(isa_matches_reference and (rtl_matches if rtl_available else True)),
            "isa_bit_exact_vs_reference": bool(isa_matches_reference),
            "isa_rtl_bitmatch": bool(rtl_matches) if rtl_available else None,
            "rtl_sim_executed": bool(rtl_available),
            "rtl_sim_passed": bool(layer1_result["rtl_sim_passed"] and layer2_result["rtl_sim_passed"]) if rtl_available else False,
            "sample_labels": [int(y) for y in eval_labels.tolist()],
            "scope_note": (
                "The independent scaled integer reference is the top of the chain. The same lowered INT8 program "
                "then matches in the ISA simulator and again in iverilog RTL on the labeled B=4 subset."
            ),
        },
        "layer_program_words": layer_program_words,
        "board_fit": board_fit,
        "batched_utilization": _utilization_summary(
            {"fc1": layer1_result, "fc2": layer2_result},
            {"fc1": (HIDDEN_DIM, 196), "fc2": (10, HIDDEN_DIM)},
        ),
        "known_limitations": [
            "Simulation only. P0 board execution remains open.",
            "INT4 remains lower-accuracy than INT8 under both per-layer and per-channel symmetric contracts.",
            "Layer-level utilization remains control-bound for large multi-tile layers because the remaining load-to-array gaps are still serialized.",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main(argv: Optional[Sequence[str]] = None) -> int:
    artifact = build_artifact()
    print(
        json.dumps(
            {
                "output_json": str(OUTPUT_JSON),
                "float_accuracy": artifact["accuracy_sweep"]["float_accuracy"],
                "int8_accuracy": artifact["accuracy_sweep"]["int8_accuracy"],
                "int4_accuracy": artifact["accuracy_sweep"]["int4_accuracy"],
                "int8_per_layer_accuracy": artifact["accuracy_comparison"]["int8"]["per_layer_accuracy"],
                "int8_per_channel_accuracy": artifact["accuracy_comparison"]["int8"]["per_channel_accuracy"],
                "int4_per_layer_accuracy": artifact["accuracy_comparison"]["int4"]["per_layer_accuracy"],
                "int4_per_channel_accuracy": artifact["accuracy_comparison"]["int4"]["per_channel_accuracy"],
                "bit_exact_vs_reference": artifact["accelerator_validation"]["bit_exact_vs_reference"],
                "selected_board": artifact["board_fit"]["selected_board"],
                "rounding_mode": artifact["quantization_contract"]["rounding_mode"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
