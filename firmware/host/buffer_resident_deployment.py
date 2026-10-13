"""Compile bias-free INT8 MLPs into an A5 weight image and control-only ISA.

The deployment path deliberately reuses the FX importer, graph pass manager,
uTPU blocked-FC planner and existing instruction lowerer. Model parameters and
inputs live in the unified buffer; the instruction image contains only control
and requantization words. This first deployment image contract supports
sequential Linear/Linear+LeakyReLU graphs. Other graph operations fail with a
named unsupported-op diagnostic instead of being silently omitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from board_config import BoardConfig
from fx_importer import FXImportError, import_fx_graph_module
from graph_ir import GraphIR, OpKind
from graph_lowering import plan_blocked_fc_graph
from graph_passes import BackendLegalityError, GraphPassManager
from isa_encoder import IsaConfig
from isa_simulator import words_from_bytes
from lowering_blocked_fc_utpu import _pack_array_to_words, lower_blocked_fc_program_utpu
from requantization import RequantParams, choose_multiplier_and_shift, quantize_symmetric, symmetric_scale, requantize_array

MAGIC_BUF_FILL = 0xA5
SUPPORTED_DEPLOYMENT_OPS = {OpKind.LINEAR, OpKind.LINEAR_RELU, OpKind.VIEW}


@dataclass(frozen=True)
class DeploymentLayer:
    name: str
    input_features: int
    output_features: int
    weight_q: np.ndarray
    weight_scales: np.ndarray
    input_scale: float
    output_scale: float
    requant: RequantParams
    activation: Optional[str]
    activation_slope: Optional[float]
    weight_base: int
    weight_words: int
    input_base: int
    output_base: int
    output_words: int
    control_words: int


@dataclass(frozen=True)
class DeploymentImage:
    model_name: str
    board_name: str
    config: IsaConfig
    input_scale: float
    weight_image_words: Tuple[int, ...]
    weight_image_bytes: bytes
    control_program: bytes
    layers: Tuple[DeploymentLayer, ...]
    address_map: Mapping[str, Any]
    fit_report: Mapping[str, Any]
    alpha_shift: int = 2

    @property
    def control_words(self) -> int:
        return len(self.control_program) // 2

    @property
    def output_features(self) -> int:
        return self.layers[-1].output_features

    def input_fill_command(self, input_q: np.ndarray) -> bytes:
        """Return one A5 command that writes a quantized input to scratch."""
        input_q = np.asarray(input_q, dtype=np.int8).reshape(-1)
        first = self.layers[0]
        padded = ((first.input_features + 7) // 8) * 8
        if input_q.size != first.input_features:
            raise ValueError(
                f"input has {input_q.size} values; model expects {first.input_features}"
            )
        padded_input = np.zeros((padded,), dtype=np.int8)
        padded_input[: input_q.size] = input_q
        words = _pack_array_to_words(padded_input, cfg=self.config)
        return encode_buffer_fill(first.input_base, words)


def encode_buffer_fill(address: int, words: Sequence[int]) -> bytes:
    """Encode MAGIC_BUF_FILL + 16-bit destination/count + little-endian words."""
    address = int(address)
    count = len(words)
    if not 0 <= address <= 0xFFFF or not 1 <= count <= 0xFFFF:
        raise ValueError(f"invalid buffer fill address/count: {address}/{count}")
    if address + count > 0x10000:
        raise ValueError("buffer fill address range exceeds the 16-bit protocol")
    payload = bytearray((MAGIC_BUF_FILL, address & 0xFF, (address >> 8) & 0xFF,
                         count & 0xFF, (count >> 8) & 0xFF))
    for word in words:
        payload.extend((int(word) & 0xFF, (int(word) >> 8) & 0xFF))
    return bytes(payload)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _calibration_tuple(calibration_inputs: Any) -> Tuple[Any, ...]:
    if isinstance(calibration_inputs, tuple):
        return calibration_inputs
    if isinstance(calibration_inputs, list):
        return tuple(calibration_inputs)
    return (calibration_inputs,)


def _requant_params(ratios: np.ndarray) -> RequantParams:
    ratios = np.asarray(ratios, dtype=np.float64).reshape(-1)
    multipliers: List[int] = []
    shifts: List[int] = []
    for ratio in ratios:
        multiplier, shift = choose_multiplier_and_shift(float(ratio))
        multipliers.append(int(multiplier))
        shifts.append(int(shift))
    return RequantParams(
        multiplier=1,
        right_shift=0,
        enable=True,
        per_channel_multipliers=tuple(multipliers),
        per_channel_right_shifts=tuple(shifts),
    )


def _collect_calibration(model: Any, calibration_inputs: Tuple[Any, ...], layer_names: Sequence[str]):
    import torch

    module_map = dict(model.named_modules())
    captured: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"inputs": [], "outputs": []} for name in layer_names
    }
    handles = []

    def make_hook(name: str):
        def hook(_module: Any, inputs: Tuple[Any, ...], output: Any) -> None:
            captured[name]["inputs"].append(_as_numpy(inputs[0]).astype(np.float32, copy=False))
            captured[name]["outputs"].append(_as_numpy(output).astype(np.float32, copy=False))
        return hook

    for name in layer_names:
        module = module_map.get(name)
        if module is None or not isinstance(module, torch.nn.Linear):
            raise ValueError(f"FX linear target '{name}' did not resolve to nn.Linear")
        if module.bias is not None:
            raise ValueError(f"uTPU INT8 deployment does not support bias in linear '{name}'")
        handles.append(module.register_forward_hook(make_hook(name)))
    model_was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            output = model(*calibration_inputs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(model_was_training)
    if not isinstance(output, torch.Tensor):
        raise ValueError("deployment currently requires one tensor model output")
    if any(not captured[name]["inputs"] for name in layer_names):
        raise ValueError("each deployed Linear must execute during calibration")
    return captured, _as_numpy(output).astype(np.float32, copy=False)


def _pack_weight_tiles(weight_q: np.ndarray, array_size: int, cfg: IsaConfig) -> List[int]:
    out_features, in_features = weight_q.shape
    out_padded = ((out_features + array_size - 1) // array_size) * array_size
    in_padded = ((in_features + array_size - 1) // array_size) * array_size
    padded = np.zeros((out_padded, in_padded), dtype=np.int8)
    padded[:out_features, :in_features] = weight_q
    words: List[int] = []
    for out0 in range(0, out_padded, array_size):
        for in0 in range(0, in_padded, array_size):
            words.extend(_pack_array_to_words(
                padded[out0 : out0 + array_size, in0 : in0 + array_size], cfg=cfg
            ))
    return words


def _fit_error(name: str, needed: int, available: int) -> ValueError:
    return ValueError(f"model '{name}' does not fit Boolean unified buffer: needs {needed} words, has {available}")


def deploy_model(
    torch_module: Any,
    board: BoardConfig,
    calibration_inputs: Any,
    *,
    alpha_shift: int = 2,
) -> DeploymentImage:
    """Build an INT8, buffer-resident control image for a sequential MLP.

    This runs FX tracing/shape propagation, graph import and optimization, then
    checks the uTPU planner's Linear nodes before emitting per-layer uTPU ISA.
    ``calibration_inputs`` is required so activation scales are measured from
    real tensors; weights and activations use symmetric INT8 quantization.
    """
    try:
        import torch
        from torch.fx.passes.shape_prop import ShapeProp
    except Exception as exc:
        raise RuntimeError("PyTorch is required for buffer-resident deployment") from exc

    inputs = _calibration_tuple(calibration_inputs)
    if not inputs:
        raise ValueError("calibration_inputs must contain the model input")
    model_name = torch_module.__class__.__name__
    fx_module = torch.fx.symbolic_trace(torch_module)
    ShapeProp(fx_module).propagate(*inputs)
    graph = import_fx_graph_module(fx_module, name=model_name)
    try:
        pass_result = GraphPassManager(target_backend="cuda").run(graph)
    except BackendLegalityError as exc:
        raise ValueError(f"graph pass legality failed for deployment: {exc}") from exc
    graph = pass_result.graph

    unexpected = [op for op in graph.ops if op.op not in SUPPORTED_DEPLOYMENT_OPS]
    if unexpected:
        names = ", ".join(f"{op.name}:{op.op}" for op in unexpected)
        raise ValueError(
            "unsupported uTPU deployment op(s) in this persistent-image flow: " + names
        )
    linear_ops = [op for op in graph.ops if op.op in {OpKind.LINEAR, OpKind.LINEAR_RELU}]
    if not linear_ops:
        raise ValueError("deployment graph has no Linear or Linear+ReLU operations")
    for op in linear_ops:
        if op.attrs.get("bias") is not None:
            raise ValueError(f"uTPU INT8 deployment does not support bias in '{op.name}'")
        # Graph fusion preserves the LeakyReLU slope when one was explicit;
        # an ordinary ReLU has no slope attribute and therefore means zero.
        slope = float(op.attrs.get("negative_slope", 0.0)) if op.op == OpKind.LINEAR_RELU else None
        if slope == 0.0:
            raise ValueError(
                f"uTPU deployment does not support plain ReLU (slope 0) in '{op.name}'; "
                "the RTL finalize path currently implements LeakyReLU slope 0.25 only"
            )
        if slope is not None and slope != 0.25:
            raise ValueError(
                f"uTPU deployment currently implements LeakyReLU slope 0.25 (alpha_shift=2); "
                f"'{op.name}' requests {slope}"
            )

    plan = plan_blocked_fc_graph(
        graph,
        array_size=board.array_size,
        apply_quant=True,
        target_backend="utpu",
    )
    if len(plan.lowered_ops) != len(linear_ops) or plan.unsupported_ops:
        detail = [f"{item.graph_op}:{item.op}" for item in plan.unsupported_ops]
        raise ValueError(f"uTPU lowering did not cover all Linear ops: {detail}")
    non_shape_fallbacks = [item for item in plan.fallback_ops if item.op != OpKind.VIEW]
    if non_shape_fallbacks:
        detail = [f"{item.graph_op}:{item.op}" for item in non_shape_fallbacks]
        raise ValueError(f"uTPU lowering has unsupported operations: {detail}")

    layer_names = [str(op.attrs["target"]) for op in linear_ops]
    captured, calibration_output = _collect_calibration(torch_module, inputs, layer_names)
    first_input = _as_numpy(inputs[0]).astype(np.float32, copy=False)
    if first_input.ndim < 2:
        raise ValueError("MLP calibration input must have a batch dimension")
    input_scale = float(symmetric_scale(first_input.reshape(first_input.shape[0], -1), qmax=127))

    cfg = IsaConfig(
        address_width=max(9, int(board.buffer_size - 1).bit_length()),
        compute_data_width=int(board.data_width_bits),
    )
    if cfg.compute_data_width != 8:
        raise ValueError("buffer-resident deployment currently requires an INT8 board config")

    # Quantize each linear layer with per-output-channel weight scales and
    # calibration-derived activation scales under the existing requant contract.
    quantized: List[Dict[str, Any]] = []
    for index, op in enumerate(linear_ops):
        target = str(op.attrs["target"])
        weight = np.asarray(op.attrs["weight"], dtype=np.float32)
        max_abs = np.max(np.abs(weight), axis=1)
        weight_scales = np.maximum(max_abs / 127.0, 1e-8).astype(np.float32)
        weight_q = np.clip(np.rint(weight / weight_scales[:, None]), -127, 127).astype(np.int8)
        layer_inputs = np.concatenate(captured[target]["inputs"], axis=0)
        layer_input_scale = float(symmetric_scale(layer_inputs, qmax=127))
        if index + 1 < len(linear_ops):
            next_name = str(linear_ops[index + 1].attrs["target"])
            next_inputs = np.concatenate(captured[next_name]["inputs"], axis=0)
            layer_output_scale = float(symmetric_scale(next_inputs, qmax=127))
        else:
            layer_output_scale = float(symmetric_scale(calibration_output, qmax=127))
        requant = _requant_params(layer_input_scale * weight_scales / layer_output_scale)
        quantized.append({
            "op": op,
            "name": target,
            "weight_q": weight_q,
            "weight_scales": weight_scales,
            "input_scale": layer_input_scale,
            "output_scale": layer_output_scale,
            "requant": requant,
        })

    weight_capacity = int(board.weight_region_words or board.buffer_size)
    weight_image: List[int] = []
    layer_weight_bases: List[Tuple[int, int]] = []
    for item in quantized:
        packed = _pack_weight_tiles(item["weight_q"], board.array_size, cfg)
        base = len(weight_image)
        layer_weight_bases.append((base, len(packed)))
        weight_image.extend(packed)
    if len(weight_image) > weight_capacity:
        raise _fit_error(model_name, len(weight_image), weight_capacity)

    input_base = weight_capacity
    first_in = int(quantized[0]["weight_q"].shape[1])
    first_in_padded = ((first_in + board.array_size - 1) // board.array_size) * board.array_size
    scratch_cursor = input_base + first_in_padded // cfg.items_per_word
    current_input_base = input_base
    combined_words: List[int] = []
    layers: List[DeploymentLayer] = []
    for index, item in enumerate(quantized):
        weight_q = item["weight_q"]
        out_features, in_features = weight_q.shape
        out_padded = ((out_features + board.array_size - 1) // board.array_size) * board.array_size
        result_base = scratch_cursor
        result_words = out_padded // cfg.items_per_word
        scratch_cursor += result_words
        weight_base, weight_words = layer_weight_bases[index]
        op = item["op"]
        activation = "leaky_relu" if op.op == OpKind.LINEAR_RELU else None
        slope = float(op.attrs.get("negative_slope", 0.0)) if activation else None
        lowered = lower_blocked_fc_program_utpu(
            weights_int4=weight_q,
            activations_int4=np.zeros((in_features,), dtype=np.int8),
            out_features=out_features,
            in_features=in_features,
            array_size=board.array_size,
            apply_relu=bool(activation),
            apply_quant=True,
            weight_addr=weight_base,
            input_addr=current_input_base,
            result_addr=result_base,
            prog_depth=board.prog_depth,
            cfg=cfg,
            requant_params=item["requant"],
            embed_weights=False,
            embed_inputs=False,
            emit_fetch=index == len(quantized) - 1,
        )
        if not lowered["fits_instruction_bram"]:
            raise ValueError(
                f"model '{model_name}' layer '{item['name']}' needs more than "
                f"PROG_DEPTH={board.prog_depth} words"
            )
        layer_words = words_from_bytes(lowered["program"])
        if not layer_words or layer_words[-1] != 0x4:
            raise ValueError(f"lowerer returned malformed HALT-terminated program for '{item['name']}'")
        combined_words.extend(layer_words[:-1])
        layers.append(DeploymentLayer(
            name=item["name"],
            input_features=int(in_features),
            output_features=int(out_features),
            weight_q=weight_q.copy(),
            weight_scales=item["weight_scales"].copy(),
            input_scale=float(item["input_scale"]),
            output_scale=float(item["output_scale"]),
            requant=item["requant"],
            activation=activation,
            activation_slope=slope,
            weight_base=weight_base,
            weight_words=weight_words,
            input_base=current_input_base,
            output_base=result_base,
            output_words=result_words,
            control_words=int(lowered["program_instruction_words"] - 1),
        ))
        current_input_base = result_base
    combined_words.append(0x4)
    control_program = b"".join(int(word).to_bytes(2, "little") for word in combined_words)
    control_count = len(combined_words)
    scratch_used = scratch_cursor - input_base
    if control_count > int(board.prog_depth):
        raise ValueError(
            f"model '{model_name}' control program needs {control_count} words, "
            f"board PROG_DEPTH is {board.prog_depth}"
        )
    total_used = input_base + scratch_used
    if total_used > int(board.buffer_size):
        raise _fit_error(model_name, total_used, int(board.buffer_size))
    weight_bytes = b"".join(int(word).to_bytes(2, "little") for word in weight_image)
    fit_report = {
        "fits": True,
        "buffer_size_words": int(board.buffer_size),
        "weight_region_capacity_words": weight_capacity,
        "weight_region_used_words": len(weight_image),
        "activation_scratch_base": input_base,
        "activation_scratch_words": scratch_used,
        "total_buffer_words_used": total_used,
        "prog_depth_words": int(board.prog_depth),
        "control_program_words": control_count,
        "weight_region_fits": len(weight_image) <= weight_capacity,
        "activation_scratch_fits": total_used <= int(board.buffer_size),
        "control_program_fits": control_count <= int(board.prog_depth),
        "bram_budget": _boolean_bram_estimate(board),
    }
    return DeploymentImage(
        model_name=model_name,
        board_name=board.name,
        config=cfg,
        input_scale=input_scale,
        weight_image_words=tuple(weight_image),
        weight_image_bytes=weight_bytes,
        control_program=control_program,
        layers=tuple(layers),
        address_map={
            "weight_layers": {
                layer.name: {"start": layer.weight_base, "end": layer.weight_base + layer.weight_words}
                for layer in layers
            },
            "input": {"start": input_base, "words": first_in_padded // cfg.items_per_word},
            "activation_outputs": {
                layer.name: {"start": layer.output_base, "words": layer.output_words}
                for layer in layers
            },
        },
        fit_report=fit_report,
        alpha_shift=int(alpha_shift),
    )


def _boolean_bram_estimate(board: BoardConfig) -> Dict[str, Any]:
    """Conservative linear estimate from committed A7-100T sweeps, not synthesis."""
    pd_words = int(board.prog_depth)
    buf_words = int(board.buffer_size)
    # At pd=32768/buf=4096, the committed sweep measured 33 tiles. Its
    # instruction memory accounts for ceil(32768*16/36k)=15 and the buffer
    # accounts for ceil(4096*16/36k)=2, leaving ~16 tiles of fixed logic.
    # Extrapolate each BRAM-backed memory by bit capacity for this configuration.
    fixed_logic_tiles = 33 - 15 - 2
    estimate = (
        fixed_logic_tiles
        + (pd_words * 16 + 36863) // 36864
        + (buf_words * 16 + 36863) // 36864
    )
    return {
        "estimated_bram36_used": int(round(estimate)),
        "bram36_available": 75,
        "estimated_margin_tiles": int(75 - round(estimate)),
        "estimated_lut_used": 16600,
        "lut_available": 32600,
        "estimated_lut_utilization_fraction": 16600.0 / 32600.0,
        "source_artifacts": [
            "bench/results/prog_depth_sweep.json",
            "bench/results/design_space_sweep.json",
        ],
        "status": "estimate_only_unverified_on_boolean",
        "lut_source_artifact": "bench/results/design_space_sweep.json (N=8, INT8, batch4, pd65536/buf4096, Artix-7)",
        "method": "BRAM: measured 32k-depth/4k-buffer fixed logic plus memory-bit extrapolation; LUT: measured 16.6k Artix batch4 point carried as a rough Boolean estimate; no XC7S50 synthesis",
    }


def quantize_input(image: Any, scale: float) -> np.ndarray:
    values = _as_numpy(image).astype(np.float32, copy=False).reshape(-1)
    return np.asarray(quantize_symmetric(values, bits=8, scale=float(scale)), dtype=np.int8)


def integer_reference(image_q: np.ndarray, deployment: DeploymentImage) -> np.ndarray:
    """Independent NumPy integer inference under the emitted per-layer contract."""
    values = np.asarray(image_q, dtype=np.int8).reshape(1, -1)
    for layer in deployment.layers:
        accum = values.astype(np.int32) @ layer.weight_q.astype(np.int32).T
        requant = layer.requant
        result = requantize_array(
            accum,
            multiplier=requant.per_channel_multipliers,
            right_shift=requant.per_channel_right_shifts,
            out_width=8,
            dtype=np.int8,
            axis=1,
        )
        if layer.activation == "leaky_relu":
            slope = 0.0 if layer.activation_slope is None else float(layer.activation_slope)
            if slope == 0.0:
                result = np.maximum(result, 0).astype(np.int8)
            elif slope == 0.25:
                result = np.where(result < 0, result >> deployment.alpha_shift, result).astype(np.int8)
            else:
                raise ValueError(f"integer reference does not support activation slope {slope}")
        values = np.asarray(result, dtype=np.int8)
    return values.reshape(-1)[: deployment.output_features]


def run_isa_inference(image_q: np.ndarray, deployment: DeploymentImage, simulator: Any) -> Tuple[np.ndarray, int]:
    """Run a control image with preloaded weights/input, preserving the buffer."""
    input_base = int(deployment.layers[0].input_base)
    padded = ((deployment.layers[0].input_features + 7) // 8) * 8
    values = np.zeros((padded,), dtype=np.int8)
    values[: len(image_q)] = np.asarray(image_q, dtype=np.int8).reshape(-1)
    input_words = _pack_array_to_words(values, cfg=deployment.config)
    for offset, word in enumerate(input_words):
        simulator.buffer[input_base + offset] = int(word)
    result = simulator.run_words(words_from_bytes(deployment.control_program), preserve_buffer=True)
    if not result.halted:
        raise RuntimeError("ISA simulator did not halt on the deployment control program")
    signed = [value - 256 if value & 0x80 else value for value in result.fetch_bytes[: deployment.output_features]]
    return np.asarray(signed, dtype=np.int8), int(result.cycle_count_sequential)
