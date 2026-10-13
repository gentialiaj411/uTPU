from __future__ import annotations

import numpy as np

from isa_encoder import IsaConfig, ISAEncoder, OPCODE_BSTORE, OPCODE_STORE
from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import _pack_array_to_words, lower_blocked_fc_program_utpu


def _word_stream(program: bytes) -> list[int]:
    return [program[i] | (program[i + 1] << 8) for i in range(0, len(program), 2)]


def test_prefilled_weight_and_input_lowering_emits_control_only_and_executes():
    cfg = IsaConfig(address_width=9, compute_data_width=8)
    weights = np.zeros((4, 8), dtype=np.int8)
    weights[0, 0] = 1
    weights[1, 1] = 1
    weights[2, 2] = 1
    weights[3, 3] = 1
    inputs = np.asarray([1, 2, 3, 4, 0, 0, 0, 0], dtype=np.int8)
    weight_addr, input_addr, result_addr = 0, 32, 64

    control = lower_blocked_fc_program_utpu(
        weights_int4=weights,
        activations_int4=inputs,
        out_features=4,
        in_features=8,
        array_size=8,
        apply_relu=False,
        apply_quant=True,
        weight_addr=weight_addr,
        input_addr=input_addr,
        result_addr=result_addr,
        prog_depth=512,
        cfg=cfg,
        embed_weights=False,
        embed_inputs=False,
        emit_fetch=True,
    )
    words = _word_stream(control["program"])
    opcodes = [word & 0x7 for word in words]
    assert OPCODE_BSTORE not in opcodes
    assert OPCODE_STORE not in opcodes
    assert control["program_instruction_words"] <= 512
    assert control["weight_words_per_tile"] == 32

    preload = ISAEncoder(cfg=cfg)
    preload.burst_store(weight_addr, _pack_array_to_words(weights, cfg=cfg))
    preload.burst_store(input_addr, _pack_array_to_words(inputs, cfg=cfg))
    result = simulate_program_bytes(
        preload.getProgram() + control["program"],
        array_size=8,
        buffer_size=512,
        cfg=cfg,
        accumulator_data_width=32,
    )
    assert result.halted
    assert result.fetch_bytes[:8] == [1, 2, 3, 4, 0, 0, 0, 0]
