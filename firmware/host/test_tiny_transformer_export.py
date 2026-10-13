import numpy as np

from tiny_transformer_export import choose_activation_exponent, quantize_weight_per_output_channel, linear_record


def test_train_calibrated_activation_exponent_is_smallest_safe_power_of_two():
    values = np.array([-0.99, 1.0, 1.01], dtype=np.float32)
    # 2^-7 gives 1.01 / 2^-7 > 127; 2^-6 is the first safe exponent.
    assert choose_activation_exponent(values, percentile=100.0) == -6


def test_per_output_channel_linear_record_uses_existing_multiplier_chooser():
    weight = np.array([[0.1, -0.2], [0.8, -0.4]], dtype=np.float32)
    codes, scales = quantize_weight_per_output_channel(weight)
    record = linear_record(input_exponent=-6, weight_scales=scales, output_exponent=-5)
    assert codes.dtype == np.int8
    assert record['output_exponent'] == -5
    assert len(record['multipliers']) == 2
    assert len(record['right_shifts']) == 2
