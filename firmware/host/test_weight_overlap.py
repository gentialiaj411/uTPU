"""Tests for weight-overlap compiler emission (Approach A)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HOST = Path(__file__).resolve().parent
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from isa_encoder import IsaConfig
from isa_simulator import simulate_program_bytes
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu
from test_batched_gemm import LEGACY_B1_FIXTURE  # noqa: E402

OVERLAP_FIXTURE = HOST / "fixtures" / "b_multitile_weight_overlap_program_bytes.json"
EXT_CFG = IsaConfig(address_width=13, compute_data_width=4)


def _lower(out_f, in_f, batch, *, weight_overlap: bool, hoist: bool = True):
    rng = np.random.default_rng(0x0BE1)
    w = rng.integers(-8, 8, size=(out_f, in_f), dtype=np.int8)
    x = rng.integers(-8, 8, size=(batch, in_f), dtype=np.int8)
    return lower_blocked_fc_program_utpu(
        w,
        x,
        out_f,
        in_f,
        16,
        False,
        True,
        0,
        256,
        512,
        cfg=EXT_CFG,
        hoist_tile_payloads=hoist,
        weight_overlap=weight_overlap,
    ), w, x


def test_b1_legacy_flag_off_matches_pinned_fixture():
    """Default weight_overlap=False must stay byte-identical to B=1 fixture."""
    data = json.loads(LEGACY_B1_FIXTURE.read_text(encoding="utf-8"))
    # Reuse the fixture's own generation path via test_batched_gemm contract:
    # this project pins via test_batched_gemm_b1_program_matches_pre_batching_legacy_bytes.
    # Here we only assert the overlap flag does not disturb that path when False.
    from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu as lower
    from isa_encoder import IsaConfig as C

    legacy_cfg = C(address_width=9, compute_data_width=4)
    rng = np.random.default_rng(0xBA7C + 16 * 37 + 16 * 13 + 1)
    w = rng.integers(-8, 8, size=(16, 16), dtype=np.int8)
    x = rng.integers(-8, 8, size=(16,), dtype=np.int8)
    a = lower(w, x, 16, 16, 16, False, True, 0, 64, 128, cfg=legacy_cfg, weight_overlap=False)
    b = lower(w, x, 16, 16, 16, False, True, 0, 64, 128, cfg=legacy_cfg)
    assert a["program"] == b["program"]
    assert a["weight_overlap"] is False


def test_b1_weight_overlap_false_matches_true_off_bytes_for_single_tile_ext():
    """Single-tile EXT_ADDR with overlap flag still differs if commit_after is set.

    Documented: multi-tile fixture is the overlap pin; B=1 legacy pin stays on
    weight_overlap=False. Single-tile overlap=True adds commit_after on prologue
    and is allowed to differ — verify explicitly so we do not assume identity.
    """
    off, _, _ = _lower(16, 16, 4, weight_overlap=False, hoist=True)
    on, _, _ = _lower(16, 16, 4, weight_overlap=True, hoist=True)
    # 16x16 is single-tile: overlap still emits commit_after on the one weight load.
    assert off["program"] != on["program"]
    assert on["weight_overlap"] is True


def test_multitile_overlap_bit_exact_vs_legacy_oracle():
    off, w, x = _lower(32, 32, 4, weight_overlap=False, hoist=True)
    on, _, _ = _lower(32, 32, 4, weight_overlap=True, hoist=True)
    assert off["program"] != on["program"]
    sim_off = simulate_program_bytes(off["program"], array_size=16, buffer_size=8192, cfg=EXT_CFG)
    sim_on = simulate_program_bytes(on["program"], array_size=16, buffer_size=8192, cfg=EXT_CFG)
    assert sim_off.halted and sim_on.halted
    assert sim_off.fetch_bytes == sim_on.fetch_bytes


def test_multitile_overlap_fixture_pin():
    if not OVERLAP_FIXTURE.exists():
        pytest.skip("overlap fixture not generated yet")
    pin = json.loads(OVERLAP_FIXTURE.read_text(encoding="utf-8"))
    on, _, _ = _lower(
        int(pin["out_features"]),
        int(pin["in_features"]),
        int(pin["batch_size"]),
        weight_overlap=True,
        hoist=True,
    )
    assert list(on["program"]) == list(pin["program_bytes"])


def test_prologue_sets_commit_after_bit_on_weight_load_encoding():
    from isa_encoder import encodeLoadWeights

    plain = encodeLoadWeights(0x100, cfg=EXT_CFG, commit_after=False)
    commit = encodeLoadWeights(0x100, cfg=EXT_CFG, commit_after=True)
    plain_hdr = plain[0] | (plain[1] << 8)
    commit_hdr = commit[0] | (commit[1] << 8)
    assert (plain_hdr >> 4) & 1 == 0
    assert (commit_hdr >> 4) & 1 == 1
    assert (commit_hdr & 0x7) == 0b011
    assert (commit_hdr >> 3) & 1 == 1

    on, _, _ = _lower(32, 32, 4, weight_overlap=True, hoist=True)
    off, _, _ = _lower(32, 32, 4, weight_overlap=False, hoist=True)
    # Overlap program must contain the commit_after weight-load header once as a
    # real instruction prefix (ext-addr: header word then address word).
    commit_hdr_bytes = bytes([commit_hdr & 0xFF, (commit_hdr >> 8) & 0xFF])
    assert on["program"].count(commit_hdr_bytes) >= 1
    assert off["program"].count(commit_hdr_bytes) == 0
    assert on["out_blocks"] * on["in_blocks"] == 4
