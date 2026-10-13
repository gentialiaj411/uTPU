from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from board_config import BoardConfig
from buffer_resident_deployment import (
    deploy_model,
    integer_reference,
    quantize_input,
    run_isa_inference,
)
from fx_importer import FXImportError
from isa_simulator import UTPUISASimulator


def test_boolean_profile_reserves_tiled_int8_mnist_weights_and_fits_estimate():
    board = BoardConfig.boolean_xc7s50()
    assert board.buffer_size == 32768
    assert board.prog_depth == 32768
    assert board.weight_region_words == 27648
    assert board.max_batch_count == 4
    assert board.weight_region_words <= board.buffer_size


def test_deploy_linear_chain_builds_control_only_program_and_matches_isa():
    torch.manual_seed(17)
    model = nn.Sequential(
        nn.Linear(8, 8, bias=False),
        nn.LeakyReLU(negative_slope=0.25),
        nn.Linear(8, 3, bias=False),
    ).eval()
    calibration = torch.linspace(-1.0, 1.0, 40).reshape(5, 8)
    image = torch.linspace(-0.5, 0.5, 8)
    deployment = deploy_model(model, BoardConfig.boolean_xc7s50(), calibration)

    assert deployment.fit_report["fits"] is True
    assert deployment.fit_report["activation_scratch_words"] < 1024
    assert deployment.fit_report["bram_budget"]["estimated_bram36_used"] <= 75
    assert all(layer.weight_words > 0 for layer in deployment.layers)
    assert deployment.control_words < deployment.fit_report["prog_depth_words"]

    simulator = UTPUISASimulator(
        array_size=8,
        buffer_size=deployment.fit_report["buffer_size_words"],
        cfg=deployment.config,
        accumulator_data_width=32,
    )
    for address, word in enumerate(deployment.weight_image_words):
        simulator.buffer[address] = word
    image_q = quantize_input(image, deployment.input_scale)
    expected = integer_reference(image_q, deployment)
    actual, cycles = run_isa_inference(image_q, deployment, simulator)
    assert cycles > 0
    assert np.array_equal(actual, expected)


def test_deploy_rejects_ops_outside_supported_graph_instead_of_fallback():
    class Unsupported(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    with pytest.raises(FXImportError, match="Unsupported call_function node|unsupported"):
        deploy_model(Unsupported(), BoardConfig.boolean_xc7s50(), torch.randn(2, 4))


def test_deploy_rejects_model_over_weight_region_budget():
    board = BoardConfig(name="tiny", prog_depth=8192, buffer_size=64, array_size=8,
                        data_width_bits=8, weight_region_words=20)
    model = nn.Sequential(nn.Linear(8, 8, bias=False)).eval()
    with pytest.raises(ValueError, match="does not fit Boolean unified buffer"):
        deploy_model(model, board, torch.randn(4, 8))


def test_plain_relu_is_not_silently_deployed_as_leaky_relu():
    model = nn.Sequential(
        nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 3, bias=False)
    ).eval()
    with pytest.raises(ValueError, match="plain ReLU.*slope 0.*LeakyReLU slope 0.25"):
        deploy_model(model, BoardConfig.boolean_xc7s50(), torch.randn(4, 8))
