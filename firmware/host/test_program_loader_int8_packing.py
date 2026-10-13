"""Regression coverage for generic activation/weight word packing."""

from isa_encoder import IsaConfig
from program_loader import ProgramLoader


def test_int8_values_pack_two_signed_bytes_per_16_bit_buffer_word():
    loader = ProgramLoader(
        uart=None,
        verbose=False,
        cfg=IsaConfig(address_width=12, compute_data_width=8),
    )

    assert loader._pack_int4_words([0x12, -3, 0x7F, -128]) == [0xFD12, 0x807F]


def test_int4_word_packing_remains_byte_identical():
    loader = ProgramLoader(
        uart=None,
        verbose=False,
        cfg=IsaConfig(address_width=9, compute_data_width=4),
    )

    assert loader._pack_int4_words([1, -2, 7, -8, 3]) == [0x87E1, 0x0003]
