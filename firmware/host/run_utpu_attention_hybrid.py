from __future__ import annotations

import json
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

from diff_oracle import compare, register_backend, run_all_backends
from graph_ir import GraphIR, OpKind, OpNode
from graph_lowering import plan_blocked_fc_graph
from graph_reference_interpreter import GraphReferenceInterpreter
from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from requantization import RequantParams, choose_multiplier_and_shift, quantize_symmetric, requantize_array, symmetric_scale
from run_rtl_batched_gemm_sim import _iverilog_run, _resolve_iverilog_tools
from utpu_batched_matmul_lowering import DEFAULT_BMM_CFG, _decode_fetch_bytes, lower_batched_matmul_utpu


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = REPO_ROOT / "bench" / "results" / "utpu_attention_hybrid.json"
SEED = 7
BATCH = 2
TOKENS = 4
MODEL_DIM = 8
HEADS = 2
HEAD_DIM = MODEL_DIM // HEADS
ARRAY_SIZE = 16
QMAX = 127
RTL_WORD_LIMIT = 12000


class AttentionOnlyBlock(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.norm = nn.RMSNorm(d, elementwise_affine=False)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.h = h
        self.hd = d // h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        n = self.norm(x)
        q = self.q(n).view(b, t, self.h, self.hd).permute(0, 2, 1, 3)
        k = self.k(n).view(b, t, self.h, self.hd).permute(0, 2, 1, 3)
        v = self.v(n).view(b, t, self.h, self.hd).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2))
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(b, t, MODEL_DIM)
        return x + self.o(ctx)


@dataclass(frozen=True)
class QuantizedLinear:
    weight_q: np.ndarray
    weight_scale: float
    output_scale: float
    requant: RequantParams


@dataclass(frozen=True)
class QuantizedBMM:
    output_scale: float
    requant: RequantParams


@dataclass(frozen=True)
class QuantizedAttention:
    norm_output_scale: float
    input_residual_scale: float
    q_proj: QuantizedLinear
    k_proj: QuantizedLinear
    v_proj: QuantizedLinear
    scores: QuantizedBMM
    probs_scale: float
    ctx: QuantizedBMM
    out_proj: QuantizedLinear


def _seed_everything() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def _cuda_backend_execution_device() -> torch.device:
    return torch.device("cpu")


def _params_from_dict(raw: Dict[str, Any]) -> RequantParams:
    return RequantParams.from_dict(raw)


def _matmul_requant_params(lhs_scale: float, rhs_scale: float, output_scale: float) -> RequantParams:
    multiplier, right_shift = choose_multiplier_and_shift((lhs_scale * rhs_scale) / output_scale)
    return RequantParams(multiplier=int(multiplier), right_shift=int(right_shift), enable=True)


def _linear_quantized_from_float(weight: np.ndarray, input_scale: float, output_scale: float) -> QuantizedLinear:
    weight_scale = float(symmetric_scale(weight, qmax=QMAX))
    weight_q = quantize_symmetric(weight, bits=8, scale=weight_scale).astype(np.int8)
    return QuantizedLinear(
        weight_q=weight_q,
        weight_scale=weight_scale,
        output_scale=float(output_scale),
        requant=_matmul_requant_params(input_scale, weight_scale, output_scale),
    )


def _requant_int32(accum: np.ndarray, params: RequantParams, *, output_scale_axis: int = -1) -> np.ndarray:
    return requantize_array(
        np.asarray(accum, dtype=np.int32),
        multiplier=params.per_channel_multipliers if params.is_per_channel else params.multiplier,
        right_shift=params.per_channel_right_shifts if params.is_per_channel else params.right_shift,
        out_width=8,
        dtype=np.int8,
        axis=output_scale_axis,
    )


def _host_layer_norm_quantized(x: np.ndarray, *, scale: float) -> np.ndarray:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    normed = F.rms_norm(xt, (xt.shape[-1],), eps=1e-5).numpy().astype(np.float32, copy=False)
    return quantize_symmetric(normed, bits=8, scale=scale).astype(np.int8)


def _build_quantized_attention(model: AttentionOnlyBlock, x: np.ndarray) -> QuantizedAttention:
    xt = torch.from_numpy(x)
    with torch.no_grad():
        norm = F.rms_norm(xt, (MODEL_DIM,), eps=1e-5).numpy().astype(np.float32, copy=False)
        q = model.q(torch.from_numpy(norm)).numpy().astype(np.float32, copy=False)
        k = model.k(torch.from_numpy(norm)).numpy().astype(np.float32, copy=False)
        v = model.v(torch.from_numpy(norm)).numpy().astype(np.float32, copy=False)
        q4 = q.reshape(BATCH, TOKENS, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k4 = k.reshape(BATCH, TOKENS, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v4 = v.reshape(BATCH, TOKENS, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        scores = np.matmul(q4, np.swapaxes(k4, -1, -2))
        probs = F.softmax(torch.from_numpy(scores), dim=-1).numpy().astype(np.float32, copy=False)
        ctx = np.matmul(probs, v4).transpose(0, 2, 1, 3).reshape(BATCH, TOKENS, MODEL_DIM)
        out = model.o(torch.from_numpy(ctx)).numpy().astype(np.float32, copy=False)

    norm_scale = float(symmetric_scale(norm, qmax=QMAX))
    q_scale = float(symmetric_scale(q, qmax=QMAX))
    k_scale = float(symmetric_scale(k, qmax=QMAX))
    v_scale = float(symmetric_scale(v, qmax=QMAX))
    scores_scale = float(symmetric_scale(scores, qmax=QMAX))
    probs_scale = float(symmetric_scale(probs, qmax=QMAX))
    ctx_scale = float(symmetric_scale(ctx, qmax=QMAX))
    out_scale = float(symmetric_scale(out, qmax=QMAX))
    return QuantizedAttention(
        norm_output_scale=norm_scale,
        input_residual_scale=float(symmetric_scale(x, qmax=QMAX)),
        q_proj=_linear_quantized_from_float(model.q.weight.detach().cpu().numpy().astype(np.float32), norm_scale, q_scale),
        k_proj=_linear_quantized_from_float(model.k.weight.detach().cpu().numpy().astype(np.float32), norm_scale, k_scale),
        v_proj=_linear_quantized_from_float(model.v.weight.detach().cpu().numpy().astype(np.float32), norm_scale, v_scale),
        scores=QuantizedBMM(output_scale=scores_scale, requant=_matmul_requant_params(q_scale, k_scale, scores_scale)),
        probs_scale=probs_scale,
        ctx=QuantizedBMM(output_scale=ctx_scale, requant=_matmul_requant_params(probs_scale, v_scale, ctx_scale)),
        out_proj=_linear_quantized_from_float(model.o.weight.detach().cpu().numpy().astype(np.float32), ctx_scale, out_scale),
    )


def _build_hybrid_graph(qcfg: QuantizedAttention) -> GraphIR:
    graph = GraphIR(name="utpu_attention_hybrid")
    graph.inputs = ["x"]
    graph.outputs = ["hybrid_out"]
    graph.add_value("x", shape=(BATCH, TOKENS, MODEL_DIM), dtype="int8")
    graph.add_op(OpNode("norm", OpKind.LAYER_NORM, ["x"], ["norm_q"], attrs={
        "eps": 1e-5,
        "norm_kind": "rms_norm",
        "output_scale": qcfg.norm_output_scale,
        "execution_domain": "host",
    }))
    graph.add_op(OpNode("q_proj", OpKind.LINEAR, ["norm_q"], ["q_proj_q"], attrs={
        "weight_int": qcfg.q_proj.weight_q,
        "requant": qcfg.q_proj.requant.as_dict(),
        "output_scale": qcfg.q_proj.output_scale,
        "execution_domain": "accelerator",
    }))
    graph.add_op(OpNode("k_proj", OpKind.LINEAR, ["norm_q"], ["k_proj_q"], attrs={
        "weight_int": qcfg.k_proj.weight_q,
        "requant": qcfg.k_proj.requant.as_dict(),
        "output_scale": qcfg.k_proj.output_scale,
        "execution_domain": "accelerator",
    }))
    graph.add_op(OpNode("v_proj", OpKind.LINEAR, ["norm_q"], ["v_proj_q"], attrs={
        "weight_int": qcfg.v_proj.weight_q,
        "requant": qcfg.v_proj.requant.as_dict(),
        "output_scale": qcfg.v_proj.output_scale,
        "execution_domain": "accelerator",
    }))
    graph.add_op(OpNode("q_view", OpKind.VIEW, ["q_proj_q"], ["q_view"], attrs={"args": (BATCH, TOKENS, HEADS, HEAD_DIM), "execution_domain": "host"}))
    graph.add_op(OpNode("k_view", OpKind.VIEW, ["k_proj_q"], ["k_view"], attrs={"args": (BATCH, TOKENS, HEADS, HEAD_DIM), "execution_domain": "host"}))
    graph.add_op(OpNode("v_view", OpKind.VIEW, ["v_proj_q"], ["v_view"], attrs={"args": (BATCH, TOKENS, HEADS, HEAD_DIM), "execution_domain": "host"}))
    graph.add_op(OpNode("q_perm", OpKind.PERMUTE, ["q_view"], ["q_perm"], attrs={"args": (0, 2, 1, 3), "execution_domain": "host"}))
    graph.add_op(OpNode("k_perm", OpKind.PERMUTE, ["k_view"], ["k_perm"], attrs={"args": (0, 2, 1, 3), "execution_domain": "host"}))
    graph.add_op(OpNode("v_perm", OpKind.PERMUTE, ["v_view"], ["v_perm"], attrs={"args": (0, 2, 1, 3), "execution_domain": "host"}))
    graph.add_op(OpNode("k_t", OpKind.PERMUTE, ["k_perm"], ["k_t"], attrs={"args": (0, 1, 3, 2), "execution_domain": "host"}))
    graph.add_op(OpNode("qk", OpKind.BATCHED_MATMUL, ["q_perm", "k_t"], ["qk_q"], attrs={
        "requant": qcfg.scores.requant.as_dict(),
        "output_scale": qcfg.scores.output_scale,
        "execution_domain": "accelerator",
    }))
    graph.add_op(OpNode("softmax_host", OpKind.SCALED_SOFTMAX, ["qk_q"], ["probs_q"], attrs={
        "scale": 1.0 / np.sqrt(float(HEAD_DIM)),
        "input_scale": qcfg.scores.output_scale,
        "output_scale": qcfg.probs_scale,
        "execution_domain": "host",
    }))
    graph.add_op(OpNode("av", OpKind.BATCHED_MATMUL, ["probs_q", "v_perm"], ["ctx_q"], attrs={
        "requant": qcfg.ctx.requant.as_dict(),
        "output_scale": qcfg.ctx.output_scale,
        "execution_domain": "accelerator",
    }))
    graph.add_op(OpNode("ctx_perm", OpKind.PERMUTE, ["ctx_q"], ["ctx_perm"], attrs={"args": (0, 2, 1, 3), "execution_domain": "host"}))
    graph.add_op(OpNode("ctx_flat", OpKind.VIEW, ["ctx_perm"], ["ctx_flat"], attrs={"args": (BATCH, TOKENS, MODEL_DIM), "execution_domain": "host"}))
    graph.add_op(OpNode("out_proj", OpKind.LINEAR, ["ctx_flat"], ["out_proj_q"], attrs={
        "weight_int": qcfg.out_proj.weight_q,
        "requant": qcfg.out_proj.requant.as_dict(),
        "output_scale": qcfg.out_proj.output_scale,
        "execution_domain": "accelerator",
    }))
    graph.add_op(OpNode("hybrid_out_host", OpKind.ADD, ["x", "out_proj_q"], ["hybrid_out"], attrs={
        "lhs_scale": qcfg.input_residual_scale,
        "rhs_scale": qcfg.out_proj.output_scale,
        "execution_domain": "host",
    }))
    return graph


def _record_accelerator_output(name: str, value: np.ndarray, trace: Dict[str, np.ndarray]) -> None:
    if name in {"q_proj", "k_proj", "v_proj", "qk", "av", "out_proj"}:
        trace[name] = np.asarray(value, dtype=np.int8, copy=False)


def _execute_linear_utpu(x: np.ndarray, weight_int: np.ndarray, requant: RequantParams) -> np.ndarray:
    x_int8 = np.asarray(x, dtype=np.int8)
    w_int8 = np.asarray(weight_int, dtype=np.int8)
    in_features = int(w_int8.shape[1])
    out_features = int(w_int8.shape[0])
    if x_int8.shape[-1] != in_features:
        raise ValueError(f"linear input feature mismatch: expected {in_features}, got {x_int8.shape[-1]}")
    prefix = tuple(int(v) for v in x_int8.shape[:-1])
    x_flat = x_int8.reshape(-1, in_features)
    lowered = lower_blocked_fc_program_utpu(
        weights_int4=w_int8,
        activations_int4=x_flat,
        out_features=out_features,
        in_features=in_features,
        array_size=ARRAY_SIZE,
        apply_relu=False,
        apply_quant=True,
        weight_addr=0,
        input_addr=256,
        result_addr=1024,
        cfg=DEFAULT_BMM_CFG,
        requant_params=requant,
    )
    sim = simulate_program_bytes(
        lowered["program"],
        array_size=ARRAY_SIZE,
        buffer_size=(1 << DEFAULT_BMM_CFG.address_width),
        cfg=DEFAULT_BMM_CFG,
        accumulator_data_width=32,
    )
    decoded = _decode_fetch_bytes(
        sim.fetch_bytes,
        out_features=out_features,
        batch_size=x_flat.shape[0],
        array_size=ARRAY_SIZE,
        cfg=DEFAULT_BMM_CFG,
    )
    return decoded.reshape(prefix + (out_features,)).astype(np.int8, copy=False)


def _execute_hybrid_integer(graph: GraphIR, x_q: np.ndarray, *, backend: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    values: Dict[str, np.ndarray] = {"x": np.asarray(x_q, dtype=np.int8)}
    accel_outputs: Dict[str, np.ndarray] = {}
    for op in graph.ops:
        if op.op == OpKind.LAYER_NORM:
            values[op.outputs[0]] = _host_layer_norm_quantized(values[op.inputs[0]].astype(np.float32), scale=float(op.attrs["output_scale"]))
            continue
        if op.op == OpKind.LINEAR:
            x = np.asarray(values[op.inputs[0]], dtype=np.int32)
            w = np.asarray(op.attrs["weight_int"], dtype=np.int8).astype(np.int32)
            if backend == "utpu":
                out = _execute_linear_utpu(
                    np.asarray(values[op.inputs[0]], dtype=np.int8),
                    np.asarray(op.attrs["weight_int"], dtype=np.int8),
                    _params_from_dict(op.attrs["requant"]),
                )
            else:
                acc = np.matmul(x, w.T)
                out = _requant_int32(acc, _params_from_dict(op.attrs["requant"]))
            values[op.outputs[0]] = out
            _record_accelerator_output(op.name, out, accel_outputs)
            continue
        if op.op == OpKind.VIEW:
            values[op.outputs[0]] = np.asarray(values[op.inputs[0]]).reshape(op.attrs["args"]).astype(np.int8, copy=False)
            continue
        if op.op == OpKind.PERMUTE:
            values[op.outputs[0]] = np.transpose(values[op.inputs[0]], axes=op.attrs["args"]).astype(np.int8, copy=False)
            continue
        if op.op == OpKind.BATCHED_MATMUL:
            lhs = np.asarray(values[op.inputs[0]], dtype=np.int8)
            rhs = np.asarray(values[op.inputs[1]], dtype=np.int8)
            if backend == "utpu":
                out = lower_batched_matmul_utpu(lhs, rhs, cfg=DEFAULT_BMM_CFG, requant_params=_params_from_dict(op.attrs["requant"])).output
            else:
                acc = np.matmul(lhs.astype(np.int32), rhs.astype(np.int32))
                out = _requant_int32(acc, _params_from_dict(op.attrs["requant"]))
            values[op.outputs[0]] = out.astype(np.int8, copy=False)
            _record_accelerator_output(op.name, values[op.outputs[0]], accel_outputs)
            continue
        if op.op == OpKind.SCALED_SOFTMAX:
            inp = np.asarray(values[op.inputs[0]], dtype=np.int8).astype(np.float32) * float(op.attrs["input_scale"])
            scaled = inp * float(op.attrs["scale"])
            probs = F.softmax(torch.from_numpy(scaled), dim=-1).numpy().astype(np.float32, copy=False)
            values[op.outputs[0]] = quantize_symmetric(probs, bits=8, scale=float(op.attrs["output_scale"])).astype(np.int8)
            continue
        if op.op == OpKind.ADD:
            lhs = np.asarray(values[op.inputs[0]], dtype=np.int8).astype(np.float32) * float(op.attrs["lhs_scale"])
            rhs = np.asarray(values[op.inputs[1]], dtype=np.int8).astype(np.float32) * float(op.attrs["rhs_scale"])
            values[op.outputs[0]] = (lhs + rhs).astype(np.float32, copy=False)
            continue
        raise ValueError(f"unsupported op in hybrid executor: {op.op}")
    packed = np.concatenate([accel_outputs[name].reshape(-1) for name in ("q_proj", "k_proj", "v_proj", "qk", "av", "out_proj")], axis=0)
    return packed.astype(np.int8, copy=False), accel_outputs, values


def _execute_hybrid_cuda(graph: GraphIR, x_q: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    device = _cuda_backend_execution_device()
    values: Dict[str, torch.Tensor] = {"x": torch.from_numpy(np.asarray(x_q, dtype=np.int8)).to(device=device, dtype=torch.int8)}
    accel_outputs: Dict[str, np.ndarray] = {}
    for op in graph.ops:
        if op.op == OpKind.LAYER_NORM:
            x = values[op.inputs[0]].to(dtype=torch.float32)
            normed = F.rms_norm(x, (x.shape[-1],), eps=float(op.attrs["eps"]))
            q = torch.round(normed / float(op.attrs["output_scale"])).clamp(-128, 127).to(dtype=torch.int8)
            values[op.outputs[0]] = q
            continue
        if op.op == OpKind.LINEAR:
            x = values[op.inputs[0]].to(dtype=torch.int32)
            w = torch.from_numpy(np.asarray(op.attrs["weight_int"], dtype=np.int8)).to(device=device, dtype=torch.int32)
            acc = torch.matmul(x, w.t())
            params = _params_from_dict(op.attrs["requant"])
            q = ((acc.to(dtype=torch.int64) * int(params.multiplier)) >> int(params.right_shift)).clamp(-128, 127).to(dtype=torch.int8)
            values[op.outputs[0]] = q
            _record_accelerator_output(op.name, q.detach().cpu().numpy(), accel_outputs)
            continue
        if op.op == OpKind.VIEW:
            values[op.outputs[0]] = values[op.inputs[0]].reshape(op.attrs["args"])
            continue
        if op.op == OpKind.PERMUTE:
            values[op.outputs[0]] = values[op.inputs[0]].permute(op.attrs["args"])
            continue
        if op.op == OpKind.BATCHED_MATMUL:
            lhs = values[op.inputs[0]].to(dtype=torch.int32)
            rhs = values[op.inputs[1]].to(dtype=torch.int32)
            acc = torch.matmul(lhs, rhs)
            params = _params_from_dict(op.attrs["requant"])
            q = ((acc.to(dtype=torch.int64) * int(params.multiplier)) >> int(params.right_shift)).clamp(-128, 127).to(dtype=torch.int8)
            values[op.outputs[0]] = q
            _record_accelerator_output(op.name, q.detach().cpu().numpy(), accel_outputs)
            continue
        if op.op == OpKind.SCALED_SOFTMAX:
            inp = values[op.inputs[0]].to(dtype=torch.float32) * float(op.attrs["input_scale"])
            probs = torch.softmax(inp * float(op.attrs["scale"]), dim=-1)
            q = torch.round(probs / float(op.attrs["output_scale"])).clamp(-128, 127).to(dtype=torch.int8)
            values[op.outputs[0]] = q
            continue
        if op.op == OpKind.ADD:
            lhs = values[op.inputs[0]].to(dtype=torch.float32) * float(op.attrs["lhs_scale"])
            rhs = values[op.inputs[1]].to(dtype=torch.float32) * float(op.attrs["rhs_scale"])
            values[op.outputs[0]] = lhs + rhs
            continue
        raise ValueError(f"unsupported op in hybrid cuda executor: {op.op}")
    packed = np.concatenate([accel_outputs[name].reshape(-1) for name in ("q_proj", "k_proj", "v_proj", "qk", "av", "out_proj")], axis=0)
    final_values = {k: v.detach().cpu().numpy() for k, v in values.items()}
    return packed.astype(np.int8, copy=False), accel_outputs, final_values


def _rtl_check_dynamic_shape(lhs: np.ndarray, rhs: np.ndarray, requant: RequantParams) -> Dict[str, Any]:
    lowered = lower_batched_matmul_utpu(lhs, rhs, cfg=DEFAULT_BMM_CFG, requant_params=requant)
    program = lowered.programs[0]
    if program.program_instruction_words > RTL_WORD_LIMIT:
        return {
            "status": "skipped_oversized",
            "program_instruction_words": int(program.program_instruction_words),
            "reason": f"program exceeds RTL budget ({program.program_instruction_words} > {RTL_WORD_LIMIT})",
        }
    mem_path = REPO_ROOT / "build" / "test_vectors" / "attention_bmm_program.mem"
    fetch_path = REPO_ROOT / "build" / "test_vectors" / "attention_bmm_fetch.mem"
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    with mem_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(program.program), 2):
            word = int.from_bytes(program.program[i : i + 2], byteorder="little", signed=False)
            f.write(f"{word:04x}\n")
    with fetch_path.open("w", encoding="utf-8") as f:
        for value in program.expected_fetch_bytes:
            f.write(f"{int(value) & 0xFF:02x}\n")
    svh_path = REPO_ROOT / "build" / "test_vectors" / "batched_gemm_expected.svh"
    mem_str = str(mem_path).replace("\\", "/")
    fetch_str = str(fetch_path).replace("\\", "/")
    with svh_path.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated by run_utpu_attention_hybrid.py\n")
        f.write(f'`define BG_MEM "{mem_str}"\n')
        f.write(f'`define BG_FETCH_MEM "{fetch_str}"\n')
        f.write(f"`define BG_WORDS {program.program_instruction_words}\n")
        f.write(f"`define BG_FETCH_N {len(program.expected_fetch_bytes)}\n")
        f.write(f"`define BG_ARRAY_SIZE {ARRAY_SIZE}\n")
        f.write(f"`define BG_BUFFER_SIZE {1 << DEFAULT_BMM_CFG.address_width}\n")
        f.write(f"`define BG_PROG_DEPTH {max(1024, program.program_instruction_words + 16)}\n")
        f.write("`define BG_EXT_ADDR_EN 1\n")
        f.write("`define BG_COMPUTE_DATA_WIDTH 8\n")
        f.write("`define BG_ACCUMULATOR_DATA_WIDTH 32\n")
    ok, log = _iverilog_run(str(REPO_ROOT))
    return {
        "status": "ok" if ok else "failed",
        "program_instruction_words": int(program.program_instruction_words),
        "rtl_sim_passed": bool(ok),
        "iverilog_available": bool(_resolve_iverilog_tools()[0] and _resolve_iverilog_tools()[1]),
        "log_excerpt": "\n".join((log or "").splitlines()[-12:]),
    }


def build_artifact(output_json: Path = OUTPUT_JSON) -> Dict[str, Any]:
    _seed_everything()
    model = AttentionOnlyBlock(MODEL_DIM, HEADS).eval()
    x = torch.randn(BATCH, TOKENS, MODEL_DIM).numpy().astype(np.float32)
    qcfg = _build_quantized_attention(model, x)
    graph = _build_hybrid_graph(qcfg)
    x_q = quantize_symmetric(x, bits=8, scale=qcfg.input_residual_scale).astype(np.int8)

    register_backend("integer_oracle", lambda g, inputs, **_: _execute_hybrid_integer(g, np.asarray(inputs[0], dtype=np.int8), backend="integer_oracle")[0])
    register_backend("cuda", lambda g, inputs, **_: _execute_hybrid_cuda(g, np.asarray(inputs[0], dtype=np.int8))[0])
    register_backend("utpu", lambda g, inputs, **_: _execute_hybrid_integer(g, np.asarray(inputs[0], dtype=np.int8), backend="utpu")[0])
    outputs = run_all_backends(graph, [x_q], backends=("integer_oracle", "cuda", "utpu"))
    parity = compare(outputs, rtol=0.0, atol=0.0)

    oracle_pack, oracle_ops, oracle_values = _execute_hybrid_integer(graph, x_q, backend="integer_oracle")
    utpu_pack, utpu_ops, _ = _execute_hybrid_integer(graph, x_q, backend="utpu")
    per_op = {}
    for name in ("q_proj", "k_proj", "v_proj", "qk", "av", "out_proj"):
        per_op[name] = {
            "shape": list(oracle_ops[name].shape),
            "oracle_utpu_bit_exact": bool(np.array_equal(oracle_ops[name], utpu_ops[name])),
        }

    lowered_ops = [op.name for op in graph.ops if str(op.attrs.get("execution_domain", "")) == "accelerator"]
    host_ops = [op.name for op in graph.ops if str(op.attrs.get("execution_domain", "")) == "host"]

    q = oracle_ops["q_proj"].reshape(BATCH, TOKENS, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = oracle_ops["k_proj"].reshape(BATCH, TOKENS, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = oracle_ops["v_proj"].reshape(BATCH, TOKENS, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    probs = np.asarray(oracle_values["probs_q"], dtype=np.int8)
    rtl_checks = {
        "qk_shape": _rtl_check_dynamic_shape(q, np.transpose(k, (0, 1, 3, 2)), qcfg.scores.requant),
        "av_shape": _rtl_check_dynamic_shape(probs, v, qcfg.ctx.requant),
    }

    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_name": "attention_only_block_hybrid_utpu_phase1",
        "shape": {"batch": BATCH, "tokens": TOKENS, "model_dim": MODEL_DIM, "heads": HEADS, "head_dim": HEAD_DIM},
        "host_accelerator_boundary": {
            "accelerator_ops": ["q_proj", "k_proj", "v_proj", "qk", "av", "out_proj"],
            "host_ops": ["norm", "view/permute reshapes", "scaled_softmax", "residual_add"],
            "scope_note": "This phase lowers only the attention GEMMs to the accelerator. Softmax, norm, and residual add stay on the host.",
        },
        "correctness_gate": {
            "packed_accelerator_outputs_bit_exact_vs_integer_reference": bool(np.array_equal(oracle_pack, utpu_pack)),
            "per_op": per_op,
        },
        "backend_parity": parity.to_dict(),
        "cuda_backend_execution_device": str(_cuda_backend_execution_device()),
        "cuda_backend_scope_note": (
            "This parity backend reuses the shared Torch execution semantics for the quantized hybrid graph. "
            "It runs on CPU on this host because Torch CUDA integer matmul kernels do not support the required INT32 addmm path here."
        ),
        "rtl_bitmatch": rtl_checks,
        "compile_summary": {
            "lowered_backend_ops": lowered_ops,
            "fallback_ops": host_ops,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main(argv: Optional[Sequence[str]] = None) -> int:
    artifact = build_artifact()
    print(json.dumps({
        "output_json": str(OUTPUT_JSON),
        "accelerator_ops": artifact["host_accelerator_boundary"]["accelerator_ops"],
        "host_ops": artifact["host_accelerator_boundary"]["host_ops"],
        "all_bit_exact": artifact["backend_parity"]["all_bit_exact"],
        "backends_compared": artifact["backend_parity"]["backends_compared"],
        "backends_skipped": artifact["backend_parity"]["backends_skipped"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
