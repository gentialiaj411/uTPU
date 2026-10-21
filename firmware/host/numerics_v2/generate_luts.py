"""Generate every numerics-v2 LUT. Float is confined to this offline script."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _record(width, signed, shape, values, domain):
    return {"width": width, "signed": signed, "depth": int(np.prod(shape)),
            "shape": list(shape), "domain": domain, "values": np.asarray(values).tolist()}


def _mem(path, values, width):
    mask = (1 << width) - 1
    digits = (width + 3) // 4
    flat = np.asarray(values).reshape(-1)
    path.write_text("".join(f"{int(item) & mask:0{digits}x}\n" for item in flat), encoding="ascii")


def generate(output: Path, silu_input_scale: float, silu_output_scale: float):
    output.mkdir(parents=True, exist_ok=True)
    position = np.arange(512)[:, None]
    pair = np.arange(4)[None, :]
    angle = position / (10000.0 ** ((2.0 * pair) / 8.0))
    cos = np.clip(np.rint(np.cos(angle) * 32767.0), -32768, 32767).astype(np.int16)
    sin = np.clip(np.rint(np.sin(angle) * 32767.0), -32768, 32767).astype(np.int16)
    code = np.arange(-128, 128, dtype=np.float64)
    real = code * silu_input_scale
    silu = np.clip(np.rint((real / (1.0 + np.exp(-real))) / silu_output_scale), -128, 127).astype(np.int8)
    # Runtime SiLU keeps the dynamic input magnitude: sigmoid is looked up in
    # Q15, then multiplied by the original INT32 activation.  Values outside
    # [-8, 8] use the asymptotes 0 and 1 in hardware.
    sigmoid_x = np.arange(-512, 513, dtype=np.float64) / 64.0
    sigmoid = np.clip(np.rint((1.0 / (1.0 + np.exp(-sigmoid_x))) * 32767.0), 0, 32767).astype(np.uint16)
    # Stable-softmax delta has scale 1/64 over [-16, 0].
    exp = np.clip(np.rint(np.exp(-np.arange(1025) / 64.0) * 32767.0), 0, 32767).astype(np.uint16)
    data = {
        "rope_cos_q15": _record(16, True, cos.shape, cos, "position 0..511 x RoPE pair 0..3"),
        "rope_sin_q15": _record(16, True, sin.shape, sin, "position 0..511 x RoPE pair 0..3"),
        "silu_i8": _record(8, True, silu.shape, silu, {"input_code": [-128, 127], "input_scale": silu_input_scale, "output_scale": silu_output_scale}),
        "sigmoid_q15": _record(16, False, sigmoid.shape, sigmoid, {"input": [-8.0, 8.0], "step": 0.015625, "outside": "0 below, 32767 above"}),
        "exp_q15": _record(16, False, exp.shape, exp, "exp(-index/64), index 0..1024"),
    }
    (output / "luts.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for name, record in data.items():
        _mem(output / f"{name}.mem", record["values"], record["width"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("luts"))
    parser.add_argument("--silu-input-scale", type=float, required=True)
    parser.add_argument("--silu-output-scale", type=float, required=True)
    args = parser.parse_args()
    generate(args.output, args.silu_input_scale, args.silu_output_scale)
