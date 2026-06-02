import numpy as np
import numpy as np

from isa_encoder import IsaConfig
from program_loader import ProgramLoader
from isa_encoder import ISAEncoder
from isa_simulator import simulate_mem_file, simulate_program_bytes
from generate_fused_rtl_test_vectors import generate_vectors


def test_isa_simulator_store_fetch_bytes():
    encoder = ISAEncoder()
    encoder.store(0, [1, -2, 3, -4])
    encoder.fetch(0, top_half=False)
    encoder.fetch(0, top_half=True)
    encoder.halt()

    result = simulate_program_bytes(encoder.getProgram())

    assert result.halted is True
    assert result.fetch_bytes == [0xE1, 0xC3]
    assert result.executed_ops["store"] == 1
    assert result.executed_ops["fetch"] == 2


def test_isa_simulator_matches_fused_vectors():
    vectors = generate_vectors()
    assert len(vectors["cases"]) >= 3
    for case in vectors["cases"]:
        result = simulate_mem_file(case["program_mem"], array_size=vectors["array_size"])
        assert result.halted is True
        assert len(result.fetch_bytes) == len(case["expected_fetch_bytes"])
        assert result.fetch_bytes == case["expected_fetch_bytes"]


def test_isa_simulator_matches_residual_fused_reference():
    rng = np.random.default_rng(0x5EED)
    cfg = IsaConfig(address_width=14, compute_data_width=4)
    loader = ProgramLoader(uart=None, verbose=False, cfg=cfg)
    array_size = 16
    fc1_w = rng.integers(-2, 3, size=(array_size, array_size), dtype=np.int8)
    fc2_w = rng.integers(-2, 3, size=(array_size, array_size), dtype=np.int8)
    x = rng.integers(-2, 3, size=(array_size,), dtype=np.int8)
    residual = rng.integers(-2, 3, size=(array_size,), dtype=np.int8)

    fused = loader.build_full_inference_program_compressed_fused(
        fc1_weights_int4=fc1_w,
        fc2_weights_int4=fc2_w,
        input_activations_int4=x,
        residual_input_int4=residual,
        array_size=array_size,
        fc1_apply_relu=True,
        fc2_apply_relu=False,
        apply_quant=True,
        result_addr=ProgramLoader.BUFFER_SECTION_C,
        residual_addr=ProgramLoader.BUFFER_SECTION_D,
    )
    result = simulate_program_bytes(
        fused["program"],
        array_size=array_size,
        buffer_size=4096,
        cfg=cfg,
        accumulator_data_width=32,
    )
    assert result.halted is True
    assert result.executed_ops["residual_add"] == 1

    fc1_acc = fc1_w.astype(np.int32) @ x.astype(np.int32)
    fc1_acc = fc1_acc + residual.astype(np.int32)
    fc1_q = np.clip(fc1_acc, -8, 7).astype(np.int32)
    fc1_q = np.where(fc1_q < 0, fc1_q >> 2, fc1_q).astype(np.int32)
    fc2_acc = fc2_w.astype(np.int32) @ fc1_q.astype(np.int32)
    fc2_q = np.clip(fc2_acc, -8, 7).astype(np.int32)

    expected_bytes = []
    for i in range(0, array_size, 4):
        chunk = fc2_q[i : i + 4].tolist()
        while len(chunk) < 4:
            chunk.append(0)
        expected_bytes.append((chunk[0] & 0xF) | ((chunk[1] & 0xF) << 4))
        expected_bytes.append((chunk[2] & 0xF) | ((chunk[3] & 0xF) << 4))

    assert result.fetch_bytes == expected_bytes
