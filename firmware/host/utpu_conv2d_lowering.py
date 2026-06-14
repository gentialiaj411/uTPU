from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from isa_encoder import IsaConfig
from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from requantization import RequantParams


DEFAULT_CONV_CFG = IsaConfig(address_width=13, compute_data_width=8)
DEFAULT_WEIGHT_ADDR = 0
DEFAULT_INPUT_ADDR = 256
DEFAULT_RESULT_ADDR = 1024
MAX_BATCH_SIZE = 64


def _pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            v = int(value[0])
            return v, v
        return int(value[0]), int(value[1])
    v = int(value)
    return v, v


def _as_compute_int(values: Any, width: int) -> np.ndarray:
    arr = np.rint(np.asarray(values, dtype=np.float32)).astype(np.int16)
    hi = (1 << (width - 1)) - 1
    lo = -(1 << (width - 1))
    return np.clip(arr, lo, hi).astype(np.int8 if width <= 8 else np.int16)


def _pack_fetch_bytes(vals: np.ndarray, *, cfg: IsaConfig) -> List[int]:
    flat = np.asarray(vals, dtype=np.int16).reshape(-1).tolist()
    out: List[int] = []
    width = int(cfg.compute_data_width)
    items = int(cfg.items_per_word)
    mask = (1 << width) - 1
    for i in range(0, len(flat), items):
        chunk = flat[i : i + items]
        while len(chunk) < items:
            chunk.append(0)
        word = 0
        for j, value in enumerate(chunk):
            word |= (int(value) & mask) << (j * width)
        out.append(word & 0xFF)
        out.append((word >> 8) & 0xFF)
    return out


def _expected_fetch_bytes_from_output(
    output_batch_features: np.ndarray,
    *,
    out_features: int,
    array_size: int,
    cfg: IsaConfig,
) -> List[int]:
    batch_size = int(output_batch_features.shape[0])
    out_blocks = (int(out_features) + int(array_size) - 1) // int(array_size)
    out_padded = out_blocks * int(array_size)
    padded = np.zeros((batch_size, out_padded), dtype=np.int16)
    padded[:, : int(out_features)] = np.asarray(output_batch_features, dtype=np.int16)
    flat: List[int] = []
    for ob in range(out_blocks):
        block = padded[:, ob * int(array_size) : (ob + 1) * int(array_size)].T
        flat.extend(block.flatten(order="F").tolist())
    return _pack_fetch_bytes(np.asarray(flat, dtype=np.int16), cfg=cfg)


def _decode_fetch_bytes(
    fetch_bytes: Sequence[int],
    *,
    out_features: int,
    batch_size: int,
    array_size: int,
    cfg: IsaConfig,
) -> np.ndarray:
    words: List[int] = []
    for i in range(0, len(fetch_bytes), 2):
        lo = int(fetch_bytes[i]) & 0xFF
        hi = int(fetch_bytes[i + 1]) & 0xFF if i + 1 < len(fetch_bytes) else 0
        words.append(lo | (hi << 8))

    mask = (1 << cfg.compute_data_width) - 1
    sign_bit = 1 << (cfg.compute_data_width - 1)
    values: List[int] = []
    for word in words:
        for shift in range(0, 16, cfg.compute_data_width):
            raw = (word >> shift) & mask
            values.append(raw - (1 << cfg.compute_data_width) if raw & sign_bit else raw)

    out_blocks = (int(out_features) + int(array_size) - 1) // int(array_size)
    out_padded = out_blocks * int(array_size)
    result = np.zeros((int(batch_size), out_padded), dtype=np.int16 if cfg.compute_data_width > 4 else np.int8)
    cursor = 0
    for ob in range(out_blocks):
        tile_count = int(array_size) * int(batch_size)
        tile_vals = values[cursor : cursor + tile_count]
        cursor += tile_count
        tile = np.asarray(tile_vals, dtype=result.dtype).reshape((int(array_size), int(batch_size)), order="F")
        result[:, ob * int(array_size) : (ob + 1) * int(array_size)] = tile.T
    return result[:, : int(out_features)]


def _im2col_positions(
    x_nchw: np.ndarray,
    *,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    groups: int,
) -> Tuple[List[np.ndarray], int, int]:
    x = np.asarray(x_nchw)
    n, c_in, h_in, w_in = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    pad_h, pad_w = padding
    if groups < 1:
        raise ValueError(f"groups must be >= 1, got {groups}")
    if c_in % groups != 0:
        raise ValueError(f"input channels {c_in} not divisible by groups={groups}")

    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )
    h_out = ((h_in + (2 * pad_h) - kh) // sh) + 1
    w_out = ((w_in + (2 * pad_w) - kw) // sw) + 1
    if h_out < 1 or w_out < 1:
        raise ValueError(
            f"invalid conv output size from input={x.shape}, kernel={kernel_size}, stride={stride}, padding={padding}"
        )

    c_in_per_group = c_in // groups
    k_per_group = c_in_per_group * kh * kw
    per_group: List[np.ndarray] = []
    for g in range(groups):
        cols = np.zeros((n * h_out * w_out, k_per_group), dtype=x.dtype)
        row = 0
        c0 = g * c_in_per_group
        c1 = (g + 1) * c_in_per_group
        for batch_idx in range(n):
            for oy in range(h_out):
                y0 = oy * sh
                for ox in range(w_out):
                    x0 = ox * sw
                    patch = x_pad[batch_idx, c0:c1, y0 : y0 + kh, x0 : x0 + kw]
                    cols[row] = patch.reshape(-1)
                    row += 1
        per_group.append(cols)
    return per_group, h_out, w_out


def conv2d_im2col_int_oracle(
    x_nchw: Any,
    weight_oihw: Any,
    *,
    stride: Any = 1,
    padding: Any = 0,
    groups: int = 1,
    apply_relu: bool = False,
    requant_params: Optional[RequantParams] = None,
    cfg: IsaConfig = DEFAULT_CONV_CFG,
) -> np.ndarray:
    x = _as_compute_int(x_nchw, cfg.compute_data_width)
    w = _as_compute_int(weight_oihw, cfg.compute_data_width)
    stride_hw = _pair(stride)
    padding_hw = _pair(padding)
    cols_by_group, h_out, w_out = _im2col_positions(
        x,
        kernel_size=(int(w.shape[2]), int(w.shape[3])),
        stride=stride_hw,
        padding=padding_hw,
        groups=int(groups),
    )
    c_out, _, _, _ = w.shape
    c_out_per_group = c_out // int(groups)
    flat_out = np.zeros((int(x.shape[0]) * h_out * w_out, c_out), dtype=np.int32)
    for g, cols in enumerate(cols_by_group):
        w_group = w[g * c_out_per_group : (g + 1) * c_out_per_group].reshape(c_out_per_group, -1).astype(np.int32)
        acc = cols.astype(np.int32) @ w_group.T
        if requant_params is not None and requant_params.enable:
            q = np.clip(
                (acc.astype(np.int64) * int(requant_params.multiplier)) >> int(requant_params.right_shift),
                -(1 << (cfg.compute_data_width - 1)),
                (1 << (cfg.compute_data_width - 1)) - 1,
            ).astype(np.int32)
        else:
            q = np.clip(
                acc,
                -(1 << (cfg.compute_data_width - 1)),
                (1 << (cfg.compute_data_width - 1)) - 1,
            ).astype(np.int32)
        if apply_relu:
            q = np.where(q < 0, q >> 2, q).astype(np.int32)
        flat_out[:, g * c_out_per_group : (g + 1) * c_out_per_group] = q
    nchw = flat_out.reshape(int(x.shape[0]), h_out, w_out, c_out).transpose(0, 3, 1, 2)
    return nchw.astype(np.float32, copy=False)


@dataclass(frozen=True)
class LoweredConvProgram:
    group_index: int
    batch_start: int
    batch_size: int
    program: bytes
    program_instruction_words: int
    expected_fetch_bytes: List[int]
    out_features: int


@dataclass(frozen=True)
class LoweredConv2DResult:
    mapping: str
    input_shape: Tuple[int, int, int, int]
    weight_shape: Tuple[int, int, int, int]
    output_shape: Tuple[int, int, int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    groups: int
    cfg: Dict[str, int]
    requant_params: Dict[str, Any]
    programs: Tuple[LoweredConvProgram, ...]


def lower_conv2d_im2col_utpu(
    x_nchw: Any,
    weight_oihw: Any,
    *,
    bias: Optional[Any] = None,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
    apply_relu: bool = False,
    array_size: int = 16,
    cfg: IsaConfig = DEFAULT_CONV_CFG,
    requant_params: Optional[RequantParams] = None,
    weight_addr: int = DEFAULT_WEIGHT_ADDR,
    input_addr: int = DEFAULT_INPUT_ADDR,
    result_addr: int = DEFAULT_RESULT_ADDR,
) -> LoweredConv2DResult:
    if bias is not None:
        raise ValueError("uTPU conv2d lowering currently supports bias=None only")
    if _pair(dilation) != (1, 1):
        raise ValueError("uTPU conv2d lowering currently supports dilation=(1, 1) only")
    x = _as_compute_int(x_nchw, cfg.compute_data_width)
    w = _as_compute_int(weight_oihw, cfg.compute_data_width)
    stride_hw = _pair(stride)
    padding_hw = _pair(padding)
    cols_by_group, h_out, w_out = _im2col_positions(
        x,
        kernel_size=(int(w.shape[2]), int(w.shape[3])),
        stride=stride_hw,
        padding=padding_hw,
        groups=int(groups),
    )
    c_out = int(w.shape[0])
    c_out_per_group = c_out // int(groups)
    programs: List[LoweredConvProgram] = []
    for group_idx, cols in enumerate(cols_by_group):
        w_group = w[group_idx * c_out_per_group : (group_idx + 1) * c_out_per_group].reshape(c_out_per_group, -1)
        total_positions = int(cols.shape[0])
        for batch_start in range(0, total_positions, MAX_BATCH_SIZE):
            batch_cols = cols[batch_start : batch_start + MAX_BATCH_SIZE]
            lowered = lower_blocked_fc_program_utpu(
                weights_int4=w_group,
                activations_int4=batch_cols,
                out_features=int(w_group.shape[0]),
                in_features=int(w_group.shape[1]),
                array_size=int(array_size),
                apply_relu=bool(apply_relu),
                apply_quant=True,
                weight_addr=int(weight_addr),
                input_addr=int(input_addr),
                result_addr=int(result_addr),
                cfg=cfg,
                requant_params=requant_params,
            )
            expected = conv2d_im2col_int_oracle(
                x_nchw=x.transpose(0, 1, 2, 3),  # explicit copy-free alias, keeps signature stable
                weight_oihw=w[group_idx * c_out_per_group : (group_idx + 1) * c_out_per_group],
                stride=stride_hw,
                padding=padding_hw,
                groups=1,
                apply_relu=bool(apply_relu),
                requant_params=requant_params,
                cfg=cfg,
            ).transpose(0, 2, 3, 1).reshape(total_positions, c_out_per_group)[
                batch_start : batch_start + int(batch_cols.shape[0])
            ]
            expected_fetch = _expected_fetch_bytes_from_output(
                np.asarray(expected, dtype=np.float32),
                out_features=int(w_group.shape[0]),
                array_size=int(array_size),
                cfg=cfg,
            )
            programs.append(
                LoweredConvProgram(
                    group_index=int(group_idx),
                    batch_start=int(batch_start),
                    batch_size=int(batch_cols.shape[0]),
                    program=lowered["program"],
                    program_instruction_words=int(lowered["program_instruction_words"]),
                    expected_fetch_bytes=expected_fetch,
                    out_features=int(w_group.shape[0]),
                )
            )
    return LoweredConv2DResult(
        mapping="conv2d -> im2col -> batched blocked-FC GEMM -> ISA program",
        input_shape=tuple(int(v) for v in x.shape),
        weight_shape=tuple(int(v) for v in w.shape),
        output_shape=(int(x.shape[0]), c_out, h_out, w_out),
        stride=stride_hw,
        padding=padding_hw,
        groups=int(groups),
        cfg={
            "address_width": int(cfg.address_width),
            "compute_data_width": int(cfg.compute_data_width),
            "buffer_size_words": int(1 << cfg.address_width),
        },
        requant_params=(
            requant_params.as_dict() if requant_params is not None else RequantParams(1, 0, enable=True).as_dict()
        ),
        programs=tuple(programs),
    )


def simulate_lowered_conv2d_utpu(
    x_nchw: Any,
    weight_oihw: Any,
    *,
    bias: Optional[Any] = None,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
    apply_relu: bool = False,
    array_size: int = 16,
    cfg: IsaConfig = DEFAULT_CONV_CFG,
    requant_params: Optional[RequantParams] = None,
    accumulator_data_width: int = 32,
) -> Dict[str, Any]:
    lowered = lower_conv2d_im2col_utpu(
        x_nchw,
        weight_oihw,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        apply_relu=apply_relu,
        array_size=array_size,
        cfg=cfg,
        requant_params=requant_params,
    )
    output_n, output_c, output_h, output_w = lowered.output_shape
    c_out_per_group = output_c // int(groups)
    flat_out = np.zeros((output_n * output_h * output_w, output_c), dtype=np.float32)
    per_program: List[Dict[str, Any]] = []
    for program in lowered.programs:
        sim = simulate_program_bytes(
            program.program,
            array_size=int(array_size),
            buffer_size=int(1 << cfg.address_width),
            cfg=cfg,
            accumulator_data_width=int(accumulator_data_width),
        )
        decoded = _decode_fetch_bytes(
            sim.fetch_bytes,
            out_features=program.out_features,
            batch_size=program.batch_size,
            array_size=int(array_size),
            cfg=cfg,
        ).astype(np.float32, copy=False)
        c0 = program.group_index * c_out_per_group
        c1 = c0 + c_out_per_group
        flat_out[program.batch_start : program.batch_start + program.batch_size, c0:c1] = decoded
        per_program.append(
            {
                "group_index": int(program.group_index),
                "batch_start": int(program.batch_start),
                "batch_size": int(program.batch_size),
                "program_instruction_words": int(program.program_instruction_words),
                "isa_expected_bitmatch": bool(list(sim.fetch_bytes) == list(program.expected_fetch_bytes)),
                "cycle_count_sequential": int(sim.cycle_count_sequential),
            }
        )
    out = flat_out.reshape(output_n, output_h, output_w, output_c).transpose(0, 3, 1, 2)
    return {
        "mapping": lowered.mapping,
        "output": out.astype(np.float32, copy=False),
        "lowered": lowered,
        "per_program": per_program,
        "all_programs_bit_exact_vs_oracle": bool(all(item["isa_expected_bitmatch"] for item in per_program)),
    }
