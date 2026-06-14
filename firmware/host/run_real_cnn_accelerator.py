from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
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

from diff_oracle import BackendUnavailable, compare, register_backend, run_all_backends
from graph_ir import GraphIR, OpKind, OpNode
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
from utpu_conv2d_lowering import (
    DEFAULT_CONV_CFG,
    conv2d_im2col_int_oracle,
    lower_conv2d_im2col_utpu,
    simulate_lowered_conv2d_utpu,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "utpu_small_cnn_validation.json"
DATA_ROOT = REPO_ROOT / "software" / "data" / "cifar-10-batches-py"
SEED = 42
INPUT_IMAGE_SIZE = 32
TRAIN_EPOCHS = 12
TRAIN_BATCH_SIZE = 256
CALIBRATION_SAMPLES = 4096
ORACLE_CUDA_BATCH = 256
UTPU_BATCH = 64
UTPU_MAX_WORKERS = min(8, max(1, (os.cpu_count() or 1)))
ISA_VALIDATION_SUBSET = 128
ARRAY_SIZE = 16
WEIGHT_ADDR = 0
INPUT_ADDR = 256
RESULT_ADDR = 1024
RTL_PROGRAM_WORD_LIMIT = 12000
ALPHA_SHIFT = 2
INT8_CFG = IsaConfig(address_width=13, compute_data_width=8)
_UTPU_WORKER_GRAPH: GraphIR | None = None


class SmallAllConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = nn.Linear(64 * 8 * 8, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x), negative_slope=0.25)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.25)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.25)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.25)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


@dataclass(frozen=True)
class QuantizedLayer:
    name: str
    weight_q: np.ndarray
    weight_scale: float
    output_scale: float
    requant: RequantParams
    stride: Optional[Tuple[int, int]] = None
    padding: Optional[Tuple[int, int]] = None
    apply_relu: bool = False


@dataclass(frozen=True)
class QuantizedModel:
    input_scale: float
    logits_scale: float
    conv1: QuantizedLayer
    conv2: QuantizedLayer
    conv3: QuantizedLayer
    conv4: QuantizedLayer
    fc: QuantizedLayer


def _seed_everything() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _load_cifar10_batches() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"missing CIFAR-10 dataset at {DATA_ROOT}; populate software/data/cifar-10-batches-py first"
        )

    def _load_split(files: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        xs: List[np.ndarray] = []
        ys: List[int] = []
        for file_name in files:
            with (DATA_ROOT / file_name).open("rb") as f:
                payload = pickle.load(f, encoding="bytes")
            xs.append(np.asarray(payload[b"data"], dtype=np.uint8))
            ys.extend(int(v) for v in payload[b"labels"])
        x = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = np.asarray(ys, dtype=np.int64)
        return x, y

    xtr, ytr = _load_split([f"data_batch_{idx}" for idx in range(1, 6)])
    xte, yte = _load_split(["test_batch"])
    return xtr, ytr, xte, yte


def _resize_inputs(images_nchw: np.ndarray) -> np.ndarray:
    if INPUT_IMAGE_SIZE == 32:
        return np.asarray(images_nchw, dtype=np.float32, copy=False)
    xt = torch.from_numpy(np.asarray(images_nchw, dtype=np.float32))
    with torch.no_grad():
        resized = F.interpolate(
            xt,
            size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    return resized.numpy().astype(np.float32, copy=False)


def _train_float_model(xtr: np.ndarray, ytr: np.ndarray) -> SmallAllConvNet:
    _seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallAllConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    xb = torch.from_numpy(xtr)
    yb = torch.from_numpy(ytr)
    for _ in range(TRAIN_EPOCHS):
        perm = torch.randperm(len(xb))
        model.train()
        for start in range(0, len(xb), TRAIN_BATCH_SIZE):
            idx = perm[start : start + TRAIN_BATCH_SIZE]
            batch_x = xb[idx].to(device=device, dtype=torch.float32)
            batch_y = yb[idx].to(device=device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _float_accuracy(model: SmallAllConvNet, xte: np.ndarray, yte: np.ndarray) -> float:
    device = next(model.parameters()).device
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(xte), 512):
            logits = model(torch.from_numpy(xte[start : start + 512]).to(device=device, dtype=torch.float32))
            preds.append(logits.argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    return float((pred == yte).mean())


def _collect_float_activations(
    model: SmallAllConvNet,
    xcal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(xcal).to(device=device, dtype=torch.float32)
        a1 = F.leaky_relu(model.conv1(x), negative_slope=0.25)
        a2 = F.leaky_relu(model.conv2(a1), negative_slope=0.25)
        a3 = F.leaky_relu(model.conv3(a2), negative_slope=0.25)
        a4 = F.leaky_relu(model.conv4(a3), negative_slope=0.25)
        logits = model.fc(a4.reshape(a4.shape[0], -1))
    return (
        a1.cpu().numpy().astype(np.float32, copy=False),
        a2.cpu().numpy().astype(np.float32, copy=False),
        a3.cpu().numpy().astype(np.float32, copy=False),
        a4.cpu().numpy().astype(np.float32, copy=False),
        logits.cpu().numpy().astype(np.float32, copy=False),
    )


def _build_quantized_model(model: SmallAllConvNet, xcal: np.ndarray) -> QuantizedModel:
    qmax = 127
    a1, a2, a3, a4, logits = _collect_float_activations(model, xcal)
    w1 = model.conv1.weight.detach().cpu().numpy().astype(np.float32)
    w2 = model.conv2.weight.detach().cpu().numpy().astype(np.float32)
    w3 = model.conv3.weight.detach().cpu().numpy().astype(np.float32)
    w4 = model.conv4.weight.detach().cpu().numpy().astype(np.float32)
    w5 = model.fc.weight.detach().cpu().numpy().astype(np.float32)

    input_scale = symmetric_scale(xcal, qmax=qmax)
    w1_scale = symmetric_scale(w1, qmax=qmax)
    a1_scale = symmetric_scale(a1, qmax=qmax)
    w2_scale = symmetric_scale(w2, qmax=qmax)
    a2_scale = symmetric_scale(a2, qmax=qmax)
    w3_scale = symmetric_scale(w3, qmax=qmax)
    a3_scale = symmetric_scale(a3, qmax=qmax)
    w4_scale = symmetric_scale(w4, qmax=qmax)
    a4_scale = symmetric_scale(a4, qmax=qmax)
    w5_scale = symmetric_scale(w5, qmax=qmax)
    logits_scale = symmetric_scale(logits, qmax=qmax)

    return QuantizedModel(
        input_scale=float(input_scale),
        logits_scale=float(logits_scale),
        conv1=QuantizedLayer(
            name="conv1",
            weight_q=quantize_symmetric(w1, bits=8, scale=w1_scale).astype(np.int8),
            weight_scale=float(w1_scale),
            output_scale=float(a1_scale),
            requant=RequantParams(*choose_multiplier_and_shift((input_scale * w1_scale) / a1_scale), enable=True),
            stride=(1, 1),
            padding=(1, 1),
            apply_relu=True,
        ),
        conv2=QuantizedLayer(
            name="conv2",
            weight_q=quantize_symmetric(w2, bits=8, scale=w2_scale).astype(np.int8),
            weight_scale=float(w2_scale),
            output_scale=float(a2_scale),
            requant=RequantParams(*choose_multiplier_and_shift((a1_scale * w2_scale) / a2_scale), enable=True),
            stride=(2, 2),
            padding=(1, 1),
            apply_relu=True,
        ),
        conv3=QuantizedLayer(
            name="conv3",
            weight_q=quantize_symmetric(w3, bits=8, scale=w3_scale).astype(np.int8),
            weight_scale=float(w3_scale),
            output_scale=float(a3_scale),
            requant=RequantParams(*choose_multiplier_and_shift((a2_scale * w3_scale) / a3_scale), enable=True),
            stride=(2, 2),
            padding=(1, 1),
            apply_relu=True,
        ),
        conv4=QuantizedLayer(
            name="conv4",
            weight_q=quantize_symmetric(w4, bits=8, scale=w4_scale).astype(np.int8),
            weight_scale=float(w4_scale),
            output_scale=float(a4_scale),
            requant=RequantParams(*choose_multiplier_and_shift((a3_scale * w4_scale) / a4_scale), enable=True),
            stride=(1, 1),
            padding=(1, 1),
            apply_relu=True,
        ),
        fc=QuantizedLayer(
            name="fc",
            weight_q=quantize_symmetric(w5, bits=8, scale=w5_scale).astype(np.int8),
            weight_scale=float(w5_scale),
            output_scale=float(logits_scale),
            requant=RequantParams(*choose_multiplier_and_shift((a4_scale * w5_scale) / logits_scale), enable=True),
            apply_relu=False,
        ),
    )


def _quantize_inputs(x: np.ndarray, *, scale: float) -> np.ndarray:
    return quantize_symmetric(np.asarray(x, dtype=np.float32), bits=8, scale=scale).astype(np.int8)


def _requantize_accum_torch(
    accum: torch.Tensor,
    params: RequantParams,
    *,
    apply_relu: bool,
) -> torch.Tensor:
    q = (accum.to(dtype=torch.int64) * int(params.multiplier)) >> int(params.right_shift)
    q = torch.clamp(q, -128, 127)
    if apply_relu:
        q = torch.where(q < 0, q >> ALPHA_SHIFT, q)
    return q.to(dtype=torch.int8)


def _requantize_accum_numpy(
    accum: np.ndarray,
    params: RequantParams,
    *,
    apply_relu: bool,
) -> np.ndarray:
    q = requantize_array(
        np.asarray(accum, dtype=np.int32),
        multiplier=int(params.multiplier),
        right_shift=int(params.right_shift),
        out_width=8,
        dtype=np.int8,
    )
    if apply_relu:
        q = np.where(q < 0, q.astype(np.int32) >> ALPHA_SHIFT, q.astype(np.int32)).astype(np.int8)
    return q


def _decode_fetch_bytes(
    fetch_bytes: Sequence[int],
    *,
    out_features: int,
    batch_size: int,
    cfg: IsaConfig,
) -> np.ndarray:
    words: List[int] = []
    for idx in range(0, len(fetch_bytes), 2):
        lo = int(fetch_bytes[idx]) & 0xFF
        hi = int(fetch_bytes[idx + 1]) & 0xFF if idx + 1 < len(fetch_bytes) else 0
        words.append(lo | (hi << 8))
    values: List[int] = []
    mask = (1 << cfg.compute_data_width) - 1
    sign_bit = 1 << (cfg.compute_data_width - 1)
    for word in words:
        for shift in range(0, 16, cfg.compute_data_width):
            raw = (word >> shift) & mask
            values.append(raw - (1 << cfg.compute_data_width) if raw & sign_bit else raw)
    out_blocks = (int(out_features) + ARRAY_SIZE - 1) // ARRAY_SIZE
    out_padded = out_blocks * ARRAY_SIZE
    result = np.zeros((int(batch_size), out_padded), dtype=np.int8)
    cursor = 0
    for ob in range(out_blocks):
        tile_vals = values[cursor : cursor + (ARRAY_SIZE * int(batch_size))]
        cursor += ARRAY_SIZE * int(batch_size)
        tile = np.asarray(tile_vals, dtype=np.int8).reshape((ARRAY_SIZE, int(batch_size)), order="F")
        result[:, ob * ARRAY_SIZE : (ob + 1) * ARRAY_SIZE] = tile.T
    return result[:, : int(out_features)]


def _simulate_utpu_fc_batch(
    inputs_q: np.ndarray,
    layer: QuantizedLayer,
    *,
    cfg: IsaConfig = INT8_CFG,
    collect_program: bool = False,
) -> Dict[str, Any]:
    lowered = lower_blocked_fc_program_utpu(
        weights_int4=layer.weight_q,
        activations_int4=inputs_q,
        out_features=int(layer.weight_q.shape[0]),
        in_features=int(layer.weight_q.shape[1]),
        array_size=ARRAY_SIZE,
        apply_relu=bool(layer.apply_relu),
        apply_quant=True,
        weight_addr=WEIGHT_ADDR,
        input_addr=INPUT_ADDR,
        result_addr=RESULT_ADDR,
        cfg=cfg,
        requant_params=layer.requant,
    )
    sim = simulate_program_bytes(
        lowered["program"],
        array_size=ARRAY_SIZE,
        buffer_size=int(1 << cfg.address_width),
        cfg=cfg,
        accumulator_data_width=32,
    )
    decoded = _decode_fetch_bytes(
        sim.fetch_bytes,
        out_features=int(layer.weight_q.shape[0]),
        batch_size=int(inputs_q.shape[0]),
        cfg=cfg,
    )
    out: Dict[str, Any] = {"output": decoded.astype(np.int8, copy=False)}
    if collect_program:
        out["program"] = lowered["program"]
        out["program_instruction_words"] = int(lowered["program_instruction_words"])
        out["expected_fetch_bytes"] = list(sim.fetch_bytes)
    return out


def _build_shared_graph(qmodel: QuantizedModel) -> GraphIR:
    graph = GraphIR(name="cifar10_32x32_fullres_small_allconv_int8")
    graph.inputs = ["x"]
    graph.outputs = ["logits_q"]
    graph.add_value("x", shape=(None, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), dtype="int8")
    graph.add_op(
        OpNode(
            name="conv1",
            op=OpKind.CONV2D,
            inputs=["x"],
            outputs=["conv1_q"],
            attrs={
                "weight_int": qmodel.conv1.weight_q,
                "bias": None,
                "stride": qmodel.conv1.stride,
                "padding": qmodel.conv1.padding,
                "dilation": (1, 1),
                "groups": 1,
                "requant": qmodel.conv1.requant.as_dict(),
                "apply_relu": True,
            },
        )
    )
    graph.add_op(
        OpNode(
            name="conv2",
            op=OpKind.CONV2D,
            inputs=["conv1_q"],
            outputs=["conv2_q"],
            attrs={
                "weight_int": qmodel.conv2.weight_q,
                "bias": None,
                "stride": qmodel.conv2.stride,
                "padding": qmodel.conv2.padding,
                "dilation": (1, 1),
                "groups": 1,
                "requant": qmodel.conv2.requant.as_dict(),
                "apply_relu": True,
            },
        )
    )
    graph.add_op(
        OpNode(
            name="conv3",
            op=OpKind.CONV2D,
            inputs=["conv2_q"],
            outputs=["conv3_q"],
            attrs={
                "weight_int": qmodel.conv3.weight_q,
                "bias": None,
                "stride": qmodel.conv3.stride,
                "padding": qmodel.conv3.padding,
                "dilation": (1, 1),
                "groups": 1,
                "requant": qmodel.conv3.requant.as_dict(),
                "apply_relu": True,
            },
        )
    )
    graph.add_op(
        OpNode(
            name="conv4",
            op=OpKind.CONV2D,
            inputs=["conv3_q"],
            outputs=["conv4_q"],
            attrs={
                "weight_int": qmodel.conv4.weight_q,
                "bias": None,
                "stride": qmodel.conv4.stride,
                "padding": qmodel.conv4.padding,
                "dilation": (1, 1),
                "groups": 1,
                "requant": qmodel.conv4.requant.as_dict(),
                "apply_relu": True,
            },
        )
    )
    graph.add_op(
        OpNode(
            name="flatten",
            op=OpKind.VIEW,
            inputs=["conv4_q"],
            outputs=["flat_q"],
            attrs={"args": (-1, 64 * 8 * 8)},
        )
    )
    graph.add_op(
        OpNode(
            name="fc",
            op=OpKind.LINEAR,
            inputs=["flat_q"],
            outputs=["logits_q"],
            attrs={
                "weight_int": qmodel.fc.weight_q,
                "bias": None,
                "requant": qmodel.fc.requant.as_dict(),
                "apply_relu": False,
                "in_features": 64 * 8 * 8,
                "out_features": 10,
            },
        )
    )
    return graph


def _params_from_dict(raw: Dict[str, Any]) -> RequantParams:
    return RequantParams(
        multiplier=int(raw["multiplier"]),
        right_shift=int(raw["right_shift"]),
        enable=bool(raw.get("enable", True)),
    )


def _run_integer_oracle_graph(graph: GraphIR, inputs: Sequence[Any]) -> np.ndarray:
    values: Dict[str, np.ndarray] = {"x": np.asarray(inputs[0], dtype=np.int8)}
    for op in graph.ops:
        if op.op == OpKind.CONV2D:
            values[op.outputs[0]] = conv2d_im2col_int_oracle(
                values[op.inputs[0]],
                op.attrs["weight_int"],
                stride=op.attrs["stride"],
                padding=op.attrs["padding"],
                apply_relu=bool(op.attrs["apply_relu"]),
                requant_params=_params_from_dict(op.attrs["requant"]),
                cfg=DEFAULT_CONV_CFG,
            ).astype(np.int8, copy=False)
            continue
        if op.op == OpKind.VIEW:
            values[op.outputs[0]] = values[op.inputs[0]].reshape(values[op.inputs[0]].shape[0], -1).astype(np.int8)
            continue
        if op.op == OpKind.LINEAR:
            x = values[op.inputs[0]].astype(np.int32)
            w = np.asarray(op.attrs["weight_int"], dtype=np.int8).astype(np.int32)
            acc = x @ w.T
            values[op.outputs[0]] = _requantize_accum_numpy(
                acc,
                _params_from_dict(op.attrs["requant"]),
                apply_relu=bool(op.attrs.get("apply_relu", False)),
            )
            continue
        raise ValueError(f"unsupported op in integer oracle runner: {op.op}")
    return np.asarray(values[graph.outputs[0]], dtype=np.int8)


def _run_integer_oracle_chunked(graph: GraphIR, inputs_q: np.ndarray) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, len(inputs_q), ORACLE_CUDA_BATCH):
        outputs.append(_run_integer_oracle_graph(graph, [inputs_q[start : start + ORACLE_CUDA_BATCH]]))
    return np.concatenate(outputs, axis=0)


def _run_cuda_integer_graph(graph: GraphIR, inputs: Sequence[Any]) -> np.ndarray:
    if not torch.cuda.is_available():
        raise BackendUnavailable("torch.cuda.is_available() returned false")
    device = torch.device("cuda")
    values: Dict[str, torch.Tensor] = {
        "x": torch.from_numpy(np.asarray(inputs[0], dtype=np.int8)).to(device=device, dtype=torch.float32)
    }
    with torch.no_grad():
        for op in graph.ops:
            if op.op == OpKind.CONV2D:
                x = values[op.inputs[0]]
                w = torch.from_numpy(np.asarray(op.attrs["weight_int"], dtype=np.int8)).to(device=device, dtype=torch.float32)
                acc = F.conv2d(
                    x,
                    w,
                    None,
                    stride=op.attrs["stride"],
                    padding=op.attrs["padding"],
                    groups=1,
                )
                q = _requantize_accum_torch(
                    acc,
                    _params_from_dict(op.attrs["requant"]),
                    apply_relu=bool(op.attrs["apply_relu"]),
                )
                values[op.outputs[0]] = q.to(dtype=torch.float32)
                continue
            if op.op == OpKind.VIEW:
                values[op.outputs[0]] = values[op.inputs[0]].reshape(values[op.inputs[0]].shape[0], -1)
                continue
            if op.op == OpKind.LINEAR:
                x = values[op.inputs[0]]
                w = torch.from_numpy(np.asarray(op.attrs["weight_int"], dtype=np.int8)).to(device=device, dtype=torch.float32)
                acc = F.linear(x, w, None)
                q = _requantize_accum_torch(
                    acc,
                    _params_from_dict(op.attrs["requant"]),
                    apply_relu=bool(op.attrs.get("apply_relu", False)),
                )
                values[op.outputs[0]] = q.to(dtype=torch.float32)
                continue
            raise RuntimeError(f"unsupported op in cuda integer graph runner: {op.op}")
    torch.cuda.synchronize()
    return values[graph.outputs[0]].to(dtype=torch.int8).cpu().numpy()


def _run_cuda_integer_chunked(graph: GraphIR, inputs_q: np.ndarray) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, len(inputs_q), ORACLE_CUDA_BATCH):
        outputs.append(_run_cuda_integer_graph(graph, [inputs_q[start : start + ORACLE_CUDA_BATCH]]))
    return np.concatenate(outputs, axis=0)


def _run_utpu_integer_graph_batch(graph: GraphIR, batch: np.ndarray) -> np.ndarray:
    values: Dict[str, np.ndarray] = {"x": np.asarray(batch, dtype=np.int8)}
    for op in graph.ops:
        if op.op == OpKind.CONV2D:
            sim = simulate_lowered_conv2d_utpu(
                values[op.inputs[0]],
                op.attrs["weight_int"],
                bias=None,
                stride=op.attrs["stride"],
                padding=op.attrs["padding"],
                dilation=(1, 1),
                groups=1,
                apply_relu=bool(op.attrs["apply_relu"]),
                cfg=DEFAULT_CONV_CFG,
                requant_params=_params_from_dict(op.attrs["requant"]),
            )
            values[op.outputs[0]] = np.asarray(sim["output"], dtype=np.int8)
            continue
        if op.op == OpKind.VIEW:
            values[op.outputs[0]] = values[op.inputs[0]].reshape(values[op.inputs[0]].shape[0], -1).astype(np.int8)
            continue
        if op.op == OpKind.LINEAR:
            result = _simulate_utpu_fc_batch(values[op.inputs[0]], QuantizedLayer(
                name="fc",
                weight_q=np.asarray(op.attrs["weight_int"], dtype=np.int8),
                weight_scale=1.0,
                output_scale=1.0,
                requant=_params_from_dict(op.attrs["requant"]),
                apply_relu=bool(op.attrs.get("apply_relu", False)),
            ))
            values[op.outputs[0]] = np.asarray(result["output"], dtype=np.int8)
            continue
        raise RuntimeError(f"unsupported op in utpu integer runner: {op.op}")
    return np.asarray(values[graph.outputs[0]], dtype=np.int8)


def _init_utpu_worker(graph: GraphIR) -> None:
    global _UTPU_WORKER_GRAPH
    _UTPU_WORKER_GRAPH = graph


def _run_utpu_worker(batch: np.ndarray) -> np.ndarray:
    if _UTPU_WORKER_GRAPH is None:
        raise RuntimeError("uTPU worker graph was not initialized")
    return _run_utpu_integer_graph_batch(_UTPU_WORKER_GRAPH, batch)


def _run_utpu_integer_graph(graph: GraphIR, inputs_q: np.ndarray) -> np.ndarray:
    batches = [np.asarray(inputs_q[start : start + UTPU_BATCH], dtype=np.int8) for start in range(0, len(inputs_q), UTPU_BATCH)]
    worker_count = min(UTPU_MAX_WORKERS, len(batches))
    if worker_count <= 1:
        outputs = [_run_utpu_integer_graph_batch(graph, batch) for batch in batches]
        return np.concatenate(outputs, axis=0)
    with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_utpu_worker, initargs=(graph,)) as executor:
        outputs = list(executor.map(_run_utpu_worker, batches))
    return np.concatenate(outputs, axis=0)


def _accuracy_from_logits(logits_q: np.ndarray, labels: np.ndarray) -> float:
    return float((np.argmax(logits_q, axis=1) == labels).mean())


def _write_program_mem(path: Path, program: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in range(0, len(program), 2):
            word = int.from_bytes(program[idx : idx + 2], byteorder="little", signed=False)
            f.write(f"{word:04x}\n")


def _write_expected_fetch_mem(path: Path, fetch_bytes: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in fetch_bytes:
            f.write(f"{int(value) & 0xFF:02x}\n")


def _write_batched_svh(
    *,
    mem_path: Path,
    expected_fetch_mem_path: Path,
    program_words: int,
    fetch_bytes: Sequence[int],
    cfg: IsaConfig,
) -> None:
    svh_path = REPO_ROOT / "build" / "test_vectors" / "batched_gemm_expected.svh"
    svh_path.parent.mkdir(parents=True, exist_ok=True)
    mem_str = str(mem_path).replace("\\", "/")
    fetch_str = str(expected_fetch_mem_path).replace("\\", "/")
    with svh_path.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated by run_real_cnn_accelerator.py\n")
        f.write(f'`define BG_MEM "{mem_str}"\n')
        f.write(f'`define BG_FETCH_MEM "{fetch_str}"\n')
        f.write(f"`define BG_WORDS {program_words}\n")
        f.write(f"`define BG_FETCH_N {len(fetch_bytes)}\n")
        f.write(f"`define BG_ARRAY_SIZE {ARRAY_SIZE}\n")
        f.write(f"`define BG_BUFFER_SIZE {1 << cfg.address_width}\n")
        f.write(f"`define BG_PROG_DEPTH {max(1024, program_words + 16)}\n")
        f.write(f"`define BG_EXT_ADDR_EN {1 if cfg.extended_address else 0}\n")
        f.write(f"`define BG_COMPUTE_DATA_WIDTH {cfg.compute_data_width}\n")
        f.write("`define BG_ACCUMULATOR_DATA_WIDTH 32\n")


def _run_single_program_rtl(program: bytes, expected_fetch_bytes: Sequence[int], stem: str) -> Dict[str, Any]:
    program_words = len(program) // 2
    if program_words > RTL_PROGRAM_WORD_LIMIT:
        return {
            "rtl_sim_executed": False,
            "rtl_sim_passed": False,
            "skip_reason": f"program exceeds RTL budget ({program_words} words > {RTL_PROGRAM_WORD_LIMIT})",
            "program_instruction_words": int(program_words),
        }
    mem_path = REPO_ROOT / "build" / "test_vectors" / f"{stem}.mem"
    fetch_path = REPO_ROOT / "build" / "test_vectors" / f"{stem}_fetch.mem"
    _write_program_mem(mem_path, program)
    _write_expected_fetch_mem(fetch_path, expected_fetch_bytes)
    _write_batched_svh(
        mem_path=mem_path,
        expected_fetch_mem_path=fetch_path,
        program_words=program_words,
        fetch_bytes=expected_fetch_bytes,
        cfg=INT8_CFG,
    )
    ok, log = _iverilog_run(str(REPO_ROOT))
    return {
        "rtl_sim_executed": "not found" not in (log or ""),
        "rtl_sim_passed": bool(ok),
        "program_instruction_words": int(program_words),
        "perf_cycle_counter": _parse_perf_counter(log, "PERF_CYCLE_COUNTER"),
        "perf_busy_counter": _parse_perf_counter(log, "PERF_BUSY_COUNTER"),
        "compute_busy_cycles": _parse_perf_counter(log, "COMPUTE_BUSY_CYCLES"),
        "compute_span_cycles": _parse_perf_counter(log, "COMPUTE_SPAN_CYCLES"),
    }


def _representative_rtl_layers(qmodel: QuantizedModel, sample_input_q: np.ndarray) -> Dict[str, Any]:
    conv1 = lower_conv2d_im2col_utpu(
        sample_input_q,
        qmodel.conv1.weight_q,
        stride=qmodel.conv1.stride,
        padding=qmodel.conv1.padding,
        apply_relu=True,
        cfg=INT8_CFG,
        requant_params=qmodel.conv1.requant,
    )
    sample_conv1 = simulate_lowered_conv2d_utpu(
        sample_input_q,
        qmodel.conv1.weight_q,
        stride=qmodel.conv1.stride,
        padding=qmodel.conv1.padding,
        apply_relu=True,
        cfg=INT8_CFG,
        requant_params=qmodel.conv1.requant,
    )["output"].astype(np.int8)
    conv4_input = simulate_lowered_conv2d_utpu(
        simulate_lowered_conv2d_utpu(
            sample_conv1,
            qmodel.conv2.weight_q,
            stride=qmodel.conv2.stride,
            padding=qmodel.conv2.padding,
            apply_relu=True,
            cfg=INT8_CFG,
            requant_params=qmodel.conv2.requant,
        )["output"],
        qmodel.conv3.weight_q,
        stride=qmodel.conv3.stride,
        padding=qmodel.conv3.padding,
        apply_relu=True,
        cfg=INT8_CFG,
        requant_params=qmodel.conv3.requant,
    )["output"].astype(np.int8)
    conv4 = lower_conv2d_im2col_utpu(
        conv4_input,
        qmodel.conv4.weight_q,
        stride=qmodel.conv4.stride,
        padding=qmodel.conv4.padding,
        apply_relu=True,
        cfg=INT8_CFG,
        requant_params=qmodel.conv4.requant,
    )
    sample_conv4 = simulate_lowered_conv2d_utpu(
        conv4_input,
        qmodel.conv4.weight_q,
        stride=qmodel.conv4.stride,
        padding=qmodel.conv4.padding,
        apply_relu=True,
        cfg=INT8_CFG,
        requant_params=qmodel.conv4.requant,
    )["output"].astype(np.int8)
    fc = _simulate_utpu_fc_batch(sample_conv4.reshape(1, -1).astype(np.int8), qmodel.fc, collect_program=True)
    layers: Dict[str, Any] = {}
    layers["conv1"] = _run_single_program_rtl(
        conv1.programs[0].program,
        conv1.programs[0].expected_fetch_bytes,
        stem="fullres_cnn_conv1_tile0",
    )
    layers["conv4"] = _run_single_program_rtl(
        conv4.programs[0].program,
        conv4.programs[0].expected_fetch_bytes,
        stem="fullres_cnn_conv4_tile0",
    )
    layers["fc"] = _run_single_program_rtl(
        fc["program"],
        fc["expected_fetch_bytes"],
        stem="fullres_cnn_fc",
    )
    return layers


def build_artifact(output_json: Path = OUTPUT_JSON) -> Dict[str, Any]:
    xtr_raw, ytr, xte_raw, yte = _load_cifar10_batches()
    xtr = _resize_inputs(xtr_raw)
    xte = _resize_inputs(xte_raw)
    float_model = _train_float_model(xtr, ytr)
    float_acc = _float_accuracy(float_model, xte, yte)

    xcal = xtr[:CALIBRATION_SAMPLES]
    qmodel = _build_quantized_model(float_model, xcal)
    inputs_q = _quantize_inputs(xte, scale=qmodel.input_scale)
    subset_size = min(ISA_VALIDATION_SUBSET, len(inputs_q))
    inputs_q_subset = np.asarray(inputs_q[:subset_size], dtype=np.int8)
    yte_subset = np.asarray(yte[:subset_size], dtype=np.int64)

    graph = _build_shared_graph(qmodel)

    register_backend("integer_oracle", lambda g, inputs, **_: _run_integer_oracle_chunked(g, np.asarray(inputs[0], dtype=np.int8)))
    register_backend("cuda", lambda g, inputs, **_: _run_cuda_integer_chunked(g, np.asarray(inputs[0], dtype=np.int8)))
    register_backend("utpu", lambda g, inputs, **_: _run_utpu_integer_graph(g, np.asarray(inputs[0], dtype=np.int8)))

    oracle_logits_full = _run_integer_oracle_chunked(graph, inputs_q)
    outputs_subset = run_all_backends(graph, [inputs_q_subset], backends=("integer_oracle", "cuda", "utpu"))
    parity_subset = compare(outputs_subset, rtol=0.0, atol=0.0)
    oracle_logits_subset = np.asarray(outputs_subset["integer_oracle"].output, dtype=np.int8)
    utpu_logits_subset = np.asarray(outputs_subset["utpu"].output, dtype=np.int8)
    cuda_logits_subset = np.asarray(outputs_subset["cuda"].output, dtype=np.int8)

    sample_input_q = inputs_q[:1]
    rtl_layers = _representative_rtl_layers(qmodel, sample_input_q)
    rtl_pass_count = int(sum(1 for result in rtl_layers.values() if result.get("rtl_sim_passed")))
    rtl_exec_count = int(sum(1 for result in rtl_layers.values() if result.get("rtl_sim_executed")))

    artifact: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_name": "cifar10_32x32_fullres_small_allconv_int8",
        "dataset": {
            "name": "cifar10_native_32x32_local",
            "source_dir": str(DATA_ROOT),
            "train_size": int(len(xtr)),
            "test_size": int(len(xte)),
            "input_shape": [3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE],
            "num_classes": 10,
        },
        "architecture": {
            "kind": "all_convolutional_plus_final_fc",
            "layers": [
                "conv3x3_s1_3to16 + leaky_relu(alpha=0.25)",
                "conv3x3_s2_16to32 + leaky_relu(alpha=0.25)",
                "conv3x3_s2_32to64 + leaky_relu(alpha=0.25)",
                "conv3x3_s1_64to64 + leaky_relu(alpha=0.25)",
                "flatten",
                "fc_4096to10",
            ],
            "why": "stride-2 convs replace pooling so the graph stays on conv + blocked-FC datapaths only",
            "input_preprocess": "CIFAR-10 kept at native 32x32 resolution before quantization",
        },
        "training": {
            "seed": SEED,
            "epochs": TRAIN_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "optimizer": "Adam(lr=0.002)",
        },
        "quantization_contract": {
            "bits": 8,
            "rounding_mode": ROUNDING_MODE,
            "symmetric_zero_point": 0,
            "requant_multiply_location": (
                "Each lowered conv/FC layer finalizes with int-accumulate -> int64 multiply -> arithmetic right shift -> "
                "saturate to INT8 in the simulator finalize path; the RTL mirrors that in quantizer.sv before optional "
                "leaky-ReLU and before the next layer consumes the result."
            ),
            "layers": {
                "conv1": {
                    "weight_scale": qmodel.conv1.weight_scale,
                    "output_scale": qmodel.conv1.output_scale,
                    "requant": qmodel.conv1.requant.as_dict(),
                },
                "conv2": {
                    "weight_scale": qmodel.conv2.weight_scale,
                    "output_scale": qmodel.conv2.output_scale,
                    "requant": qmodel.conv2.requant.as_dict(),
                },
                "conv3": {
                    "weight_scale": qmodel.conv3.weight_scale,
                    "output_scale": qmodel.conv3.output_scale,
                    "requant": qmodel.conv3.requant.as_dict(),
                },
                "conv4": {
                    "weight_scale": qmodel.conv4.weight_scale,
                    "output_scale": qmodel.conv4.output_scale,
                    "requant": qmodel.conv4.requant.as_dict(),
                },
                "fc": {
                    "weight_scale": qmodel.fc.weight_scale,
                    "output_scale": qmodel.fc.output_scale,
                    "requant": qmodel.fc.requant.as_dict(),
                },
            },
        },
        "accuracy": {
            "float_accuracy": float_acc,
            "deployed_int8_accuracy_integer_oracle_full_test": _accuracy_from_logits(oracle_logits_full, yte),
            "validated_subset_size": int(subset_size),
            "integer_oracle_accuracy_validated_subset": _accuracy_from_logits(oracle_logits_subset, yte_subset),
            "cuda_integer_accuracy_validated_subset": _accuracy_from_logits(cuda_logits_subset, yte_subset),
            "utpu_isa_accuracy_validated_subset": _accuracy_from_logits(utpu_logits_subset, yte_subset),
            "scope_note": (
                "Float accuracy is measured on the full 10k CIFAR-10 test split. Deployed INT8 accuracy is scored on the full "
                "10k with the integer oracle implementing the same integer contract. The oracle is validated against CUDA and "
                "the cycle-accurate uTPU ISA simulator on the recorded subset only."
            ),
        },
        "correctness": {
            "oracle_utpu_bit_exact_validated_subset": bool(np.array_equal(oracle_logits_subset, utpu_logits_subset)),
            "oracle_cuda_bit_exact_validated_subset": bool(np.array_equal(oracle_logits_subset, cuda_logits_subset)),
            "cuda_utpu_bit_exact_validated_subset": bool(np.array_equal(cuda_logits_subset, utpu_logits_subset)),
            "validated_subset_samples": int(subset_size),
            "full_test_samples_accuracy_only": int(len(yte)),
        },
        "backend_parity_validated_subset": parity_subset.to_dict(),
        "rtl_bitmatch": {
            "iverilog_available": bool(_resolve_iverilog_tools()[0] and _resolve_iverilog_tools()[1]),
            "layers": rtl_layers,
            "rtl_pass_count": rtl_pass_count,
            "rtl_executed_count": rtl_exec_count,
        },
        "mapping": {
            "strategy": "im2col",
            "where_mapping_happens": "firmware/host/utpu_conv2d_lowering.py::lower_conv2d_im2col_utpu",
            "why": (
                "im2col reuses the existing blocked-FC GEMM and finalize-requant path directly. The cost is activation "
                "materialization proportional to kernel_area * output_positions, which is acceptable for this bounded "
                "small-CNN simulation phase."
            ),
        },
        "honest_scope": (
            "Lowered a full-resolution CIFAR-10 CNN to the accelerator's GEMM datapath. Full-10k deployed INT8 accuracy is "
            "scored on the integer oracle implementing the deployed contract, and that oracle is bit-exact to CUDA and the "
            "uTPU ISA simulator on the validated subset recorded in this artifact. RTL coverage remains representative-layer "
            "only. This is simulation only; no FPGA/silicon execution claim."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> int:
    artifact = build_artifact()
    print(
        json.dumps(
            {
                "output_json": str(OUTPUT_JSON),
                "float_accuracy": artifact["accuracy"]["float_accuracy"],
                "deployed_int8_accuracy_integer_oracle_full_test": artifact["accuracy"]["deployed_int8_accuracy_integer_oracle_full_test"],
                "validated_subset_size": artifact["accuracy"]["validated_subset_size"],
                "backend_parity_bit_exact_subset": artifact["backend_parity_validated_subset"]["all_bit_exact"],
                "backends_compared_subset": artifact["backend_parity_validated_subset"]["backends_compared"],
                "backends_skipped_subset": artifact["backend_parity_validated_subset"]["backends_skipped"],
                "oracle_utpu_bit_exact_subset": artifact["correctness"]["oracle_utpu_bit_exact_validated_subset"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
