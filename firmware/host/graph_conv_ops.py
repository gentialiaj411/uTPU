"""NumPy reference helpers for 2D convolution and pooling (ResNet-style graphs)."""

from typing import Any, Optional, Sequence, Tuple

import numpy as np


def _pair(value: Any, default: int = 1) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            v = int(value[0])
            return v, v
        return int(value[0]), int(value[1])
    v = int(value)
    return v, v


def _pad_spatial(
    x: np.ndarray,
    padding: Sequence[int],
) -> np.ndarray:
    if len(padding) == 1:
        pad_h = pad_w = int(padding[0])
    else:
        pad_h, pad_w = int(padding[0]), int(padding[1])
    if pad_h == 0 and pad_w == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0.0,
    )


def conv2d_nchw_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: Any = 1,
    padding: Any = 0,
    groups: int = 1,
) -> np.ndarray:
    """NCHW float32 convolution reference (im2col + matmul)."""
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    groups = int(groups)
    if groups < 1:
        raise ValueError(f"groups must be >= 1, got {groups}")

    n, c_in, h_in, w_in = x.shape
    c_out, c_w, kh, kw = w.shape
    if c_w * groups != c_in:
        raise ValueError(f"weight in_channels {c_w * groups} != input channels {c_in}")
    if c_out % groups != 0:
        raise ValueError(f"out_channels {c_out} not divisible by groups={groups}")

    stride_h, stride_w = _pair(stride)
    if isinstance(padding, (tuple, list)):
        if len(padding) == 1:
            pad_h = pad_w = int(padding[0])
        else:
            pad_h, pad_w = int(padding[0]), int(padding[1])
    else:
        pad_h = pad_w = int(padding)

    x_pad = _pad_spatial(x, (pad_h, pad_w))
    _, _, h_pad, w_pad = x_pad.shape
    h_out = (h_pad - kh) // stride_h + 1
    w_out = (w_pad - kw) // stride_w + 1
    if h_out < 1 or w_out < 1:
        raise ValueError(f"invalid conv output size from input {x.shape}, weight {w.shape}")

    out = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)
    c_out_per_group = c_out // groups
    c_in_per_group = c_in // groups

    for g in range(groups):
        x_g = x_pad[:, g * c_in_per_group : (g + 1) * c_in_per_group]
        w_g = w[g * c_out_per_group : (g + 1) * c_out_per_group]
        for o in range(c_out_per_group):
            w_o = w_g[o]
            for iy in range(h_out):
                y0 = iy * stride_h
                for ix in range(w_out):
                    x0 = ix * stride_w
                    patch = x_g[:, :, y0 : y0 + kh, x0 : x0 + kw]
                    out[:, g * c_out_per_group + o, iy, ix] = np.sum(
                        patch * w_o.reshape(1, c_in_per_group, kh, kw), axis=(1, 2, 3)
                    )

    if bias is not None:
        b = np.asarray(bias, dtype=np.float32).reshape(1, c_out, 1, 1)
        out = out + b
    return out.astype(np.float32, copy=False)


def max_pool2d_nchw_numpy(
    x: np.ndarray,
    kernel_size: Any,
    stride: Optional[Any] = None,
    padding: Any = 0,
    dilation: Any = 1,
    ceil_mode: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    kh, kw = _pair(kernel_size)
    if stride is None:
        sh, sw = kh, kw
    else:
        sh, sw = _pair(stride)
    dh, dw = _pair(dilation)
    pad_h, pad_w = _pair(padding)

    n, c, h_in, w_in = x.shape
    h_out = (h_in + 2 * pad_h - dh * (kh - 1) - 1) // sh + 1
    w_out = (w_in + 2 * pad_w - dw * (kw - 1) - 1) // sw + 1
    if ceil_mode:
        if (h_in + 2 * pad_h - dh * (kh - 1) - 1) % sh != 0:
            h_out += 1
        if (w_in + 2 * pad_w - dw * (kw - 1) - 1) % sw != 0:
            w_out += 1

    out = np.full((n, c, h_out, w_out), -np.inf, dtype=np.float32)
    for iy in range(h_out):
        for ix in range(w_out):
            y_origin = iy * sh - pad_h
            x_origin = ix * sw - pad_w
            window = np.full((n, c, kh, kw), -np.inf, dtype=np.float32)
            for ky in range(kh):
                y = y_origin + ky * dh
                if y < 0 or y >= h_in:
                    continue
                for kx in range(kw):
                    x_pos = x_origin + kx * dw
                    if x_pos < 0 or x_pos >= w_in:
                        continue
                    window[:, :, ky, kx] = x[:, :, y, x_pos]
            out[:, :, iy, ix] = np.max(window, axis=(2, 3))
    return out.astype(np.float32, copy=False)


def adaptive_avg_pool2d_nchw_numpy(x: np.ndarray, output_size: Any) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    oh, ow = _pair(output_size)
    n, c, h_in, w_in = x.shape
    out = np.zeros((n, c, oh, ow), dtype=np.float32)
    for oy in range(oh):
        y0 = int(np.floor(oy * h_in / oh))
        y1 = int(np.ceil((oy + 1) * h_in / oh))
        for ox in range(ow):
            x0 = int(np.floor(ox * w_in / ow))
            x1 = int(np.ceil((ox + 1) * w_in / ow))
            region = x[:, :, y0:y1, x0:x1]
            out[:, :, oy, ox] = np.mean(region, axis=(2, 3))
    return out.astype(np.float32, copy=False)


def fold_conv_bn_weights(
    conv_weight: np.ndarray,
    conv_bias: Optional[np.ndarray],
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    bn_running_mean: np.ndarray,
    bn_running_var: np.ndarray,
    bn_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm into Conv2d for inference."""
    w = np.asarray(conv_weight, dtype=np.float32)
    gamma = np.asarray(bn_weight, dtype=np.float32)
    beta = np.asarray(bn_bias, dtype=np.float32)
    mean = np.asarray(bn_running_mean, dtype=np.float32)
    var = np.asarray(bn_running_var, dtype=np.float32)
    std = np.sqrt(var + float(bn_eps))
    scale = gamma / std
    w_fold = w * scale.reshape(-1, 1, 1, 1)
    if conv_bias is None:
        b_conv = np.zeros((w.shape[0],), dtype=np.float32)
    else:
        b_conv = np.asarray(conv_bias, dtype=np.float32)
    b_fold = beta - mean * scale + b_conv * scale
    return w_fold.astype(np.float32, copy=False), b_fold.astype(np.float32, copy=False)
