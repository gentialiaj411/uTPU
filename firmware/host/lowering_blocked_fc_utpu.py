import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from isa_encoder import ISAEncoder, DEFAULT_CFG, IsaConfig, pack_values_to_word
from requantization import RequantParams
from compiler_abstractions import (
    BlockedFCProblem,
    build_blocked_fc_schedule,
    utpu_target_desc,
)


def _store_int4_array_to_buffer(
    encoder: ISAEncoder,
    base_addr: int,
    data,
    *,
    order: str = "C",
) -> None:
    flat = list(np.asarray(data).flatten(order=order))
    addr = base_addr
    items_per_word = encoder.cfg.items_per_word
    for i in range(0, len(flat), items_per_word):
        chunk = flat[i:i + items_per_word]
        while len(chunk) < items_per_word:
            chunk.append(0)
        encoder.store(addr, chunk)
        addr += 1


def _pack_array_to_words(
    data,
    *,
    cfg: IsaConfig,
    order: str = "C",
) -> List[int]:
    flat = list(np.asarray(data, dtype=np.int8).flatten(order=order))
    words: List[int] = []
    for i in range(0, len(flat), cfg.items_per_word):
        chunk = flat[i:i + cfg.items_per_word]
        while len(chunk) < cfg.items_per_word:
            chunk.append(0)
        words.append(pack_values_to_word(chunk, cfg))
    return words


def _normalize_batched_activations(
    activations_int4,
    in_features: int,
) -> Tuple[np.ndarray, int]:
    x = np.asarray(activations_int4, dtype=np.int8)
    if x.ndim == 1:
        if x.shape[0] != in_features:
            raise ValueError(f"activation length mismatch: expected {in_features}, got {x.shape[0]}")
        return x.reshape(1, in_features), 1
    if x.ndim == 2:
        if x.shape[1] != in_features:
            raise ValueError(
                f"batched activation shape mismatch: expected second dim {in_features}, got {x.shape}"
            )
        return x, int(x.shape[0])
    raise ValueError(f"activations must be 1D or 2D, got ndim={x.ndim}")


DEFAULT_PROG_DEPTH = 1024
MAX_BATCH_SIZE = 64


def lower_blocked_fc_program_utpu(
    weights_int4,
    activations_int4,
    out_features: int,
    in_features: int,
    array_size: int,
    apply_relu: bool,
    apply_quant: bool,
    weight_addr: int,
    input_addr: int,
    result_addr: int,
    prog_depth: int = DEFAULT_PROG_DEPTH,
    cfg: IsaConfig = DEFAULT_CFG,
    hoist_tile_payloads: bool = False,
    requant_params: Optional[RequantParams] = None,
) -> Dict[str, Any]:
    schedule = build_blocked_fc_schedule(
        problem=BlockedFCProblem(
            out_features=out_features,
            in_features=in_features,
            array_size=array_size,
        ),
        target=utpu_target_desc(array_size=array_size),
    )

    w = np.asarray(weights_int4, dtype=np.int8)
    x_batch, batch_size = _normalize_batched_activations(activations_int4, in_features)
    if w.shape != (out_features, in_features):
        raise ValueError(f"weights shape mismatch: expected {(out_features, in_features)}, got {w.shape}")
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch_size={batch_size} exceeds max supported batch size {MAX_BATCH_SIZE}")
    if batch_size > 1 and not cfg.extended_address:
        raise ValueError("batched blocked-FC lowering requires extended-address ISA encoding")
    if requant_params is not None and requant_params.is_per_channel:
        if not cfg.extended_address:
            raise ValueError("per-channel requant lowering requires extended-address ISA encoding")
        if requant_params.vector_length != out_features:
            raise ValueError(
                f"per-channel requant vector length mismatch: expected {out_features}, got {requant_params.vector_length}"
            )

    out_blocks = schedule.out_blocks
    in_blocks = schedule.in_blocks
    out_padded = schedule.out_padded
    in_padded = schedule.in_padded

    w_pad = np.zeros((out_padded, in_padded), dtype=np.int8)
    w_pad[:out_features, :in_features] = w
    x_pad = np.zeros((batch_size, in_padded), dtype=np.int8)
    x_pad[:, :in_features] = x_batch

    encoder = ISAEncoder(cfg=cfg)
    if requant_params is not None and not requant_params.is_per_channel:
        encoder.requant_params(
            int(requant_params.multiplier),
            int(requant_params.right_shift),
            enable=bool(requant_params.enable),
        )
    block_ops = 0
    output_words_per_chunk = (array_size * array_size) // cfg.items_per_word
    output_chunks_per_out_block = max(1, (batch_size + array_size - 1) // array_size)
    output_words_per_out_block = output_words_per_chunk * output_chunks_per_out_block if batch_size > 1 else (array_size // cfg.items_per_word)
    use_hoisted_tiles = bool(
        hoist_tile_payloads and batch_size > 1 and cfg.extended_address
    )
    if use_hoisted_tiles:
        weight_words_per_tile = (array_size * array_size) // cfg.items_per_word
        input_words_per_tile = (array_size * batch_size) // cfg.items_per_word
        weight_tiles_total_words = out_blocks * in_blocks * weight_words_per_tile
        input_tiles_total_words = in_blocks * input_words_per_tile
        hoisted_input_addr = weight_addr + weight_tiles_total_words
        hoisted_result_addr = hoisted_input_addr + input_tiles_total_words
        total_required_words = hoisted_result_addr + (out_blocks * output_words_per_out_block)
        buffer_capacity_words = 1 << cfg.address_width
        if total_required_words > buffer_capacity_words:
            raise ValueError(
                "hoisted blocked-FC payloads exceed buffer capacity: "
                f"need {total_required_words} words, have {buffer_capacity_words}"
            )
        for ob in range(out_blocks):
            o0 = ob * array_size
            o1 = o0 + array_size
            for ib in range(in_blocks):
                i0 = ib * array_size
                i1 = i0 + array_size
                weight_block = w_pad[o0:o1, i0:i1]
                tile_addr = weight_addr + ((ob * in_blocks + ib) * weight_words_per_tile)
                encoder.burst_store(
                    tile_addr,
                    _pack_array_to_words(weight_block, cfg=cfg),
                )
        for ib in range(in_blocks):
            i0 = ib * array_size
            i1 = i0 + array_size
            input_block = x_pad[:, i0:i1]
            input_matrix = np.zeros((array_size, batch_size), dtype=np.int8)
            input_matrix[:, :batch_size] = input_block.T
            tile_addr = hoisted_input_addr + (ib * input_words_per_tile)
            encoder.burst_store(
                tile_addr,
                _pack_array_to_words(input_matrix, cfg=cfg, order="F"),
            )
    else:
        hoisted_input_addr = input_addr
        hoisted_result_addr = result_addr

    for ob in range(out_blocks):
        out_base_addr = hoisted_result_addr + ob * output_words_per_out_block
        o0 = ob * array_size
        o1 = o0 + array_size
        for ib in range(in_blocks):
            i0 = ib * array_size
            i1 = i0 + array_size
            weight_block = w_pad[o0:o1, i0:i1]
            input_block = x_pad[:, i0:i1]

            if use_hoisted_tiles:
                weight_tile_addr = weight_addr + (
                    (ob * in_blocks + ib) * ((array_size * array_size) // cfg.items_per_word)
                )
                input_tile_addr = hoisted_input_addr + (
                    ib * ((array_size * batch_size) // cfg.items_per_word)
                )
                encoder.loadWeights(weight_tile_addr)
            else:
                _store_int4_array_to_buffer(encoder, weight_addr, weight_block)
                encoder.loadWeights(weight_addr)
            if batch_size == 1:
                _store_int4_array_to_buffer(encoder, input_addr, input_block[0])
                encoder.loadInputs(input_addr, batch_count=1)
            else:
                if not use_hoisted_tiles:
                    input_matrix = np.zeros((array_size, batch_size), dtype=np.int8)
                    input_matrix[:, :batch_size] = input_block.T
                    _store_int4_array_to_buffer(encoder, input_addr, input_matrix, order="F")
                encoder.loadInputs(
                    input_tile_addr if use_hoisted_tiles else input_addr,
                    batch_count=batch_size,
                )
            encoder.run(
                out_base_addr,
                compute=True,
                quantize=False,
                relu=False,
                acc_clear=(ib == 0),
                batch_count=batch_size,
            )
            block_ops += 1

        if requant_params is not None and requant_params.is_per_channel:
            block_count = min(array_size, max(0, out_features - o0))
            block_params = requant_params.block(o0, block_count, pad_to=array_size)
            assert block_params.per_channel_multipliers is not None
            assert block_params.per_channel_right_shifts is not None
            encoder.requant_params(
                block_params.per_channel_multipliers,
                block_params.per_channel_right_shifts,
                enable=bool(block_params.enable),
            )

        encoder.run(
            out_base_addr,
            compute=False,
            quantize=apply_quant,
            relu=apply_relu,
            acc_clear=False,
            batch_count=batch_size,
        )
        fetch_words = (batch_size * array_size) // cfg.items_per_word
        for widx in range(fetch_words):
            addr = out_base_addr + widx
            encoder.fetch(addr, top_half=False)
            encoder.fetch(addr, top_half=True)

    encoder.halt()
    program = encoder.getProgram()
    words = len(program) // 2

    executable = True
    blockers = []
    if not apply_quant:
        executable = False
        blockers.append(
            "Current blocked-FC runtime finalizes through quantized int4 buffer output; "
            "raw int32 host-visible output path is not yet exposed."
        )

    return {
        "program": program,
        "program_instruction_words": int(words),
        "fits_instruction_bram": bool(words <= int(prog_depth)),
        "instruction_bram_words": int(prog_depth),
        "array_size": int(array_size),
        "out_blocks": int(out_blocks),
        "in_blocks": int(in_blocks),
        "block_ops": int(block_ops),
        "batch_size": int(batch_size),
        "hoist_tile_payloads": bool(use_hoisted_tiles),
        "requant_params": requant_params.as_dict() if requant_params is not None else None,
        "executable_on_current_fpga_path": bool(executable),
        "int32_accumulation_supported": True,
        "quantize_after_accumulation_supported": True,
        "blockers": blockers,
    }
