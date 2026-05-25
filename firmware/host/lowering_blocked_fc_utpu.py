import numpy as np
from typing import Dict, Any

from isa_encoder import ISAEncoder
from compiler_abstractions import (
    BlockedFCProblem,
    build_blocked_fc_schedule,
    utpu_target_desc,
)


def _store_int4_array_to_buffer(encoder: ISAEncoder, base_addr: int, data) -> None:
    flat = list(np.asarray(data).flatten())
    addr = base_addr
    for i in range(0, len(flat), 4):
        chunk = flat[i:i + 4]
        while len(chunk) < 4:
            chunk.append(0)
        encoder.store(addr, chunk)
        addr += 1


DEFAULT_PROG_DEPTH = 1024


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
    x = np.asarray(activations_int4, dtype=np.int8).flatten()
    if w.shape != (out_features, in_features):
        raise ValueError(f"weights shape mismatch: expected {(out_features, in_features)}, got {w.shape}")
    if x.shape[0] != in_features:
        raise ValueError(f"activation length mismatch: expected {in_features}, got {x.shape[0]}")

    out_blocks = schedule.out_blocks
    in_blocks = schedule.in_blocks
    out_padded = schedule.out_padded
    in_padded = schedule.in_padded

    w_pad = np.zeros((out_padded, in_padded), dtype=np.int8)
    w_pad[:out_features, :in_features] = w
    x_pad = np.zeros(in_padded, dtype=np.int8)
    x_pad[:in_features] = x

    encoder = ISAEncoder()
    block_ops = 0
    for ob in range(out_blocks):
        out_base_addr = result_addr + ob * (array_size // 4)
        o0 = ob * array_size
        o1 = o0 + array_size
        for ib in range(in_blocks):
            i0 = ib * array_size
            i1 = i0 + array_size
            weight_block = w_pad[o0:o1, i0:i1]
            input_block = x_pad[i0:i1]

            _store_int4_array_to_buffer(encoder, weight_addr, weight_block)
            encoder.loadWeights(weight_addr)
            _store_int4_array_to_buffer(encoder, input_addr, input_block)
            encoder.loadInputs(input_addr)
            encoder.run(
                out_base_addr,
                compute=True,
                quantize=False,
                relu=False,
                acc_clear=(ib == 0),
            )
            block_ops += 1

        encoder.run(
            out_base_addr,
            compute=False,
            quantize=apply_quant,
            relu=apply_relu,
            acc_clear=False,
        )
        for widx in range(array_size // 4):
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
        "executable_on_current_fpga_path": bool(executable),
        "int32_accumulation_supported": True,
        "quantize_after_accumulation_supported": True,
        "blockers": blockers,
    }
