"""Focused structural checks for the deterministic Phase-4 lowering."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "firmware" / "host"))

from lower_tiny_transformer import M, build_program, build_program_stages, build_weight_image


@pytest.fixture(scope="module")
def exported():
    path = ROOT / "bench" / "results" / "tiny_transformer_numerics.pt"
    if not path.is_file():
        pytest.skip("deterministic tiny-transformer checkpoint is unavailable")
    return torch.load(path, map_location="cpu", weights_only=False)


def test_complete_program_is_deterministic_and_within_32k(exported) -> None:
    first = build_program(exported["qstate"], exported["export_schema"])
    second = build_program(exported["qstate"], exported["export_schema"])
    assert first == second
    assert len(first) == 25_551
    assert len(first) <= 32_768
    assert first[-1] == 4  # HALT


def test_lowering_emits_every_transformer_operation_and_safe_images(exported) -> None:
    stages = dict(build_program_stages(exported["qstate"], exported["export_schema"]))
    required = {"pos_add", "rmsnorm1", "qkv", "qkv_to_rm", "scores_head0", "softmax_head2",
                "av_head1", "out_proj", "residual1", "rmsnorm2", "ffn1_leaky_relu", "ffn2",
                "residual2", "final_norm", "logits"}
    assert required <= set(stages)
    assert all(stage for stage in stages.values())
    image = build_weight_image(exported["qstate"], exported["export_schema"])
    assert len(image) == M.x
    assert M.logits_l + 1152 <= 32768
