import numpy as np
import pytest

from requantization import MAX_REQUANT_MULTIPLIER, MAX_REQUANT_RIGHT_SHIFT
from tiny_transformer_export import choose_activation_exponent, linear_int8_rne, quantize_activation, quantize_weight_per_output_channel, linear_record


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


@pytest.mark.parametrize('ratio', [2.0**-20, 2.0**-11, 2.0**-3, 0.37, 1.0, 3.7, 17.0])
def test_multiplier_shift_realizes_representable_ratio_with_int16_bound(ratio):
    record = linear_record(input_exponent=0, weight_scales=np.array([ratio]), output_exponent=0)
    multiplier, shift = record['multipliers'][0], record['right_shifts'][0]
    approximation = multiplier * 2.0**(-shift)
    # Nearest-integer multiplier gives absolute ratio error <= 0.5*2^-shift;
    # dividing by ratio, this is <= 0.5/multiplier for a representable record.
    assert abs(approximation - ratio) / ratio <= 0.5 / multiplier + 1e-15
    assert 0 < multiplier <= MAX_REQUANT_MULTIPLIER
    assert 0 <= shift <= MAX_REQUANT_RIGHT_SHIFT


def test_exact_int8_activation_boundary_maps_to_127():
    exponent = -6
    codes = quantize_activation(np.array([127.0 * 2.0**exponent]), exponent)
    np.testing.assert_array_equal(codes, np.array([127], dtype=np.int8))


def test_linear_requant_matches_float_at_calibrated_scale():
    rng = np.random.default_rng(12)
    x_real = rng.uniform(-0.8, 0.8, (8, 11)).astype(np.float64)
    weight = rng.uniform(-0.2, 0.2, (7, 11)).astype(np.float64)
    e_in, e_out = -7, -5
    x_code = quantize_activation(x_real, e_in)
    weight_code, weight_scales = quantize_weight_per_output_channel(weight)
    record = linear_record(input_exponent=e_in, weight_scales=weight_scales, output_exponent=e_out)
    actual = linear_int8_rne(x_code, weight_code, record)
    expected = x_real @ weight.T
    dequant = actual.astype(np.float64) * 2.0**e_out
    cosine = float((expected.ravel() @ dequant.ravel()) / (np.linalg.norm(expected) * np.linalg.norm(dequant)))
    # Weight quantization plus output RNE is bounded by one output LSB plus
    # accumulated half-weight-LSB terms; this random calibrated case is <3 LSB.
    assert cosine >= 0.99
    assert np.max(np.abs(expected - dequant)) <= 3 * 2.0**e_out


def test_linear_requant_supports_zero_shift_identity_ratio():
    record = {"multipliers": [1], "right_shifts": [0], "input_exponent": 0, "output_exponent": 0}
    actual = linear_int8_rne(np.array([[2, -3]], np.int8), np.array([[4, 5]], np.int8), record)
    np.testing.assert_array_equal(actual, np.array([[-7]], dtype=np.int8))
