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
    for case in vectors["cases"]:
        result = simulate_mem_file(case["program_mem"], array_size=vectors["array_size"])
        assert result.halted is True
        assert result.fetch_bytes == case["expected_fetch_bytes"]
