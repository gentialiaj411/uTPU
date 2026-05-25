"""Phase 5 tests: list scheduler + buffer-slot allocator.

Coverage:

* :class:`BufferSlotAllocator` — deterministic placement, capacity
  enforcement, Belady-style spill, peak-live tracking.
* :class:`BlockedFCScheduler` / :func:`lower_blocked_fc_program_scheduled`
  — bit-exact ``fetch_bytes`` against the legacy
  ``lower_blocked_fc_program_utpu`` across a shape sweep, deterministic
  byte output across re-runs, scheduled cycle/store-byte budget never
  worse than naive, and graceful fall-back to naive emission when
  capacity is too small to hoist all input blocks.
* :class:`ISASimulationResult` — Phase 5 measurement schema lock
  (``store_bytes_total`` / ``redundant_store_bytes`` /
  ``compute_runs`` / ``total_macs`` / ``cycles_per_mac`` /
  ``array_utilization`` populated on every ``run_words`` invocation).
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pytest

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from isa_simulator import simulate_program_bytes  # noqa: E402
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu  # noqa: E402
from scheduler_allocator import (  # noqa: E402
    BlockedFCProblem,
    BlockedFCScheduler,
    BufferAllocationError,
    BufferSlotAllocator,
    LiveInterval,
    Value,
    lower_blocked_fc_program_scheduled,
)


# ---------------------------------------------------------------------------
# Shape sweep used by bit-exactness + cycle-budget tests
# ---------------------------------------------------------------------------

SHAPES: List[Tuple[int, int]] = [
    (16, 16),    # ob=1, ib=1: zero reuse opportunity (boundary case)
    (32, 16),    # ob=2, ib=1: minimal hoist win
    (32, 32),    # ob=2, ib=2
    (64, 32),    # ob=4, ib=2
    (32, 64),    # ob=2, ib=4
    (64, 64),    # ob=4, ib=4
    (128, 64),   # ob=8, ib=4: peak savings region
    (16, 64),    # ob=1, ib=4: hoists but no ob-reuse, still bit-exact
]
ARRAY_SIZE = 16
WEIGHT_ADDR = 256
INPUT_ADDR = 0
RESULT_ADDR = 320


def _gen(out: int, ind: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.integers(low=-8, high=8, size=(out, ind), dtype=np.int8)
    x = rng.integers(low=-8, high=8, size=ind, dtype=np.int8)
    return w, x


# ---------------------------------------------------------------------------
# Allocator unit tests
# ---------------------------------------------------------------------------


def test_allocator_basic_first_fit_packing():
    intervals = [
        LiveInterval(Value("a", 4), first_use_idx=0, last_use_idx=10),
        LiveInterval(Value("b", 4), first_use_idx=1, last_use_idx=5),
        LiveInterval(Value("c", 4), first_use_idx=2, last_use_idx=8),
    ]
    alloc = BufferSlotAllocator(capacity=12, base_offset=0).allocate(
        intervals, op_count=11
    )
    assert alloc.spills == []
    assert sorted(alloc.layout.values()) == [0, 4, 8]
    assert alloc.peak_live_words == 12


def test_allocator_reuses_slot_after_value_dies():
    # 'a' dies at idx 4, 'b' starts at idx 5 -> can land in 'a's slot.
    intervals = [
        LiveInterval(Value("a", 6), first_use_idx=0, last_use_idx=4),
        LiveInterval(Value("b", 6), first_use_idx=5, last_use_idx=9),
    ]
    alloc = BufferSlotAllocator(capacity=6, base_offset=0).allocate(
        intervals, op_count=10
    )
    assert alloc.spills == []
    assert alloc.layout == {"a": 0, "b": 0}
    assert alloc.peak_live_words == 6


def test_allocator_belady_evicts_value_with_latest_next_use():
    # Capacity exactly fits two values, 'c' arrives at idx=4 needing
    # eviction of either 'a' (next use=10) or 'b' (next use=6).
    # Belady picks 'a' (later next use).
    intervals = [
        LiveInterval(Value("a", 4), first_use_idx=0, last_use_idx=10),
        LiveInterval(Value("b", 4), first_use_idx=1, last_use_idx=8),
        LiveInterval(Value("c", 4), first_use_idx=4, last_use_idx=7),
    ]
    next_use_after = {
        ("a", 4): 10,
        ("b", 4): 6,
    }
    alloc = BufferSlotAllocator(capacity=8, base_offset=0).allocate(
        intervals, op_count=11, next_use_after=next_use_after
    )
    assert len(alloc.spills) == 1
    spill = alloc.spills[0]
    assert spill.evicted_value == "a"
    assert spill.replaced_value == "c"
    assert spill.op_idx == 4


def test_allocator_raises_when_value_larger_than_capacity():
    intervals = [
        LiveInterval(Value("big", 16), first_use_idx=0, last_use_idx=4),
    ]
    with pytest.raises(BufferAllocationError):
        BufferSlotAllocator(capacity=8, base_offset=0).allocate(
            intervals, op_count=5
        )


def test_allocator_deterministic_across_runs():
    intervals = [
        LiveInterval(Value(f"v_{i}", 2), first_use_idx=i, last_use_idx=i + 6)
        for i in range(6)
    ]
    a1 = BufferSlotAllocator(capacity=8, base_offset=0).allocate(intervals, op_count=12)
    a2 = BufferSlotAllocator(capacity=8, base_offset=0).allocate(intervals, op_count=12)
    assert a1.layout == a2.layout
    assert [s.evicted_value for s in a1.spills] == [s.evicted_value for s in a2.spills]


def test_allocator_base_offset_is_applied_to_layout():
    intervals = [
        LiveInterval(Value("a", 4), first_use_idx=0, last_use_idx=4),
        LiveInterval(Value("b", 4), first_use_idx=1, last_use_idx=4),
    ]
    alloc = BufferSlotAllocator(capacity=8, base_offset=64).allocate(
        intervals, op_count=5
    )
    assert sorted(alloc.layout.values()) == [64, 68]


# ---------------------------------------------------------------------------
# Scheduler bit-exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out,ind", SHAPES)
def test_scheduled_fetch_bytes_match_naive(out: int, ind: int):
    w, x = _gen(out, ind, seed=0xC0DE + out * 31 + ind)
    naive = lower_blocked_fc_program_utpu(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    sched = lower_blocked_fc_program_scheduled(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    rn = simulate_program_bytes(naive["program"], array_size=ARRAY_SIZE)
    rs = simulate_program_bytes(sched["program"], array_size=ARRAY_SIZE)
    assert rn.fetch_bytes == rs.fetch_bytes, (
        f"shape=({out},{ind}) fetch mismatch: naive[:8]={rn.fetch_bytes[:8]} "
        f"sched[:8]={rs.fetch_bytes[:8]}"
    )


@pytest.mark.parametrize("out,ind", SHAPES)
def test_scheduled_with_relu_matches_naive(out: int, ind: int):
    w, x = _gen(out, ind, seed=0xBEEF + out * 17 + ind)
    naive = lower_blocked_fc_program_utpu(
        w, x, out, ind, ARRAY_SIZE, True, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    sched = lower_blocked_fc_program_scheduled(
        w, x, out, ind, ARRAY_SIZE, True, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    rn = simulate_program_bytes(naive["program"], array_size=ARRAY_SIZE)
    rs = simulate_program_bytes(sched["program"], array_size=ARRAY_SIZE)
    assert rn.fetch_bytes == rs.fetch_bytes


def test_scheduled_program_is_byte_deterministic_across_runs():
    w, x = _gen(64, 64, seed=42)
    sched_a = lower_blocked_fc_program_scheduled(
        w, x, 64, 64, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    sched_b = lower_blocked_fc_program_scheduled(
        w, x, 64, 64, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    assert sched_a["program"] == sched_b["program"]
    assert sched_a["buffer_layout"] == sched_b["buffer_layout"]
    assert sched_a["spill_events"] == sched_b["spill_events"]


# ---------------------------------------------------------------------------
# Scheduler cycle / reload budget (sim-only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out,ind", SHAPES)
def test_scheduled_cycles_never_worse_than_naive(out: int, ind: int):
    w, x = _gen(out, ind, seed=0xACE + out + ind)
    naive = lower_blocked_fc_program_utpu(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    sched = lower_blocked_fc_program_scheduled(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    rn = simulate_program_bytes(naive["program"], array_size=ARRAY_SIZE)
    rs = simulate_program_bytes(sched["program"], array_size=ARRAY_SIZE)
    assert rs.cycle_count_sequential <= rn.cycle_count_sequential
    assert rs.store_bytes_total <= rn.store_bytes_total


def test_scheduled_strictly_reduces_store_bytes_when_ob_gt_one():
    # ob>1 + ib>=1 is the canonical input-reuse case — scheduling MUST
    # remove (ob - 1) * ib * (a // items_per_word) input STORE words
    # from the naive emission.
    out, ind = 128, 64
    w, x = _gen(out, ind, seed=99)
    naive = lower_blocked_fc_program_utpu(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    sched = lower_blocked_fc_program_scheduled(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    rn = simulate_program_bytes(naive["program"], array_size=ARRAY_SIZE)
    rs = simulate_program_bytes(sched["program"], array_size=ARRAY_SIZE)
    out_blocks = sched["out_blocks"]
    in_blocks = sched["in_blocks"]
    in_words = ARRAY_SIZE // 4  # 4 items_per_word at INT4
    expected_saved_bytes = (out_blocks - 1) * in_blocks * in_words * 2
    assert rn.store_bytes_total - rs.store_bytes_total == expected_saved_bytes
    assert rs.cycle_count_sequential < rn.cycle_count_sequential


# ---------------------------------------------------------------------------
# Spill path: undersized capacity falls back to naive bit-by-bit
# ---------------------------------------------------------------------------


def test_undersized_capacity_falls_back_to_naive_bit_for_bit():
    out, ind = 64, 64
    w, x = _gen(out, ind, seed=7)
    naive = lower_blocked_fc_program_utpu(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    # Capacity below in_block_size forces every input block to spill,
    # regressing to per-load STOREs at input_base_addr.
    sched = lower_blocked_fc_program_scheduled(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR,
        buffer_capacity=4,
    )
    assert sched["spill_count"] >= 1
    assert sched["program"] == naive["program"]


def test_capacity_one_below_full_hoist_falls_back_cleanly():
    out, ind = 32, 64
    w, x = _gen(out, ind, seed=8)
    in_blocks = ind // ARRAY_SIZE
    in_block_words = ARRAY_SIZE // 4
    full_hoist_capacity = in_blocks * in_block_words
    sched_below = lower_blocked_fc_program_scheduled(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR,
        buffer_capacity=full_hoist_capacity - 1,
    )
    sched_at = lower_blocked_fc_program_scheduled(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR,
        buffer_capacity=full_hoist_capacity,
    )
    naive = lower_blocked_fc_program_utpu(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    rn = simulate_program_bytes(naive["program"], array_size=ARRAY_SIZE)
    r_at = simulate_program_bytes(sched_at["program"], array_size=ARRAY_SIZE)
    r_below = simulate_program_bytes(sched_below["program"], array_size=ARRAY_SIZE)
    # Above threshold: full hoist (no spills, fewer cycles than naive).
    assert sched_at["spill_count"] == 0
    assert r_at.cycle_count_sequential < rn.cycle_count_sequential
    # Below threshold: spills, falls back to naive byte-for-byte.
    assert sched_below["spill_count"] >= 1
    assert r_below.cycle_count_sequential == rn.cycle_count_sequential
    assert r_below.fetch_bytes == rn.fetch_bytes


# ---------------------------------------------------------------------------
# Measurement schema lock (Phase 5 fields populated on every run)
# ---------------------------------------------------------------------------


def test_simulation_result_exposes_phase5_measurement_fields():
    w, x = _gen(32, 32, seed=11)
    naive = lower_blocked_fc_program_utpu(
        w, x, 32, 32, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    res = simulate_program_bytes(naive["program"], array_size=ARRAY_SIZE)
    expected_fields = {
        "store_bytes_total",
        "redundant_store_bytes",
        "compute_runs",
        "total_macs",
        "cycles_per_mac",
        "array_utilization",
    }
    for field_name in expected_fields:
        assert hasattr(res, field_name), f"missing field {field_name}"
    assert res.store_bytes_total > 0
    assert res.compute_runs > 0
    assert res.total_macs == res.compute_runs * ARRAY_SIZE * ARRAY_SIZE
    assert res.cycles_per_mac > 0.0
    assert 0.0 < res.array_utilization < 1.0


def test_buffer_layout_reports_input_slot_per_block():
    out, ind = 64, 64
    w, x = _gen(out, ind, seed=12)
    sched = lower_blocked_fc_program_scheduled(
        w, x, out, ind, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    in_blocks = sched["in_blocks"]
    layout = sched["buffer_layout"]
    for ib in range(in_blocks):
        assert f"inp_{ib}" in layout, f"missing layout entry for inp_{ib}"
    assert sched["peak_live_words"] == in_blocks * (ARRAY_SIZE // 4)


# ---------------------------------------------------------------------------
# JSON-serializable scheduler metadata (used by the bench artifact)
# ---------------------------------------------------------------------------


def test_scheduler_metadata_is_json_serializable():
    w, x = _gen(64, 64, seed=13)
    sched = lower_blocked_fc_program_scheduled(
        w, x, 64, 64, ARRAY_SIZE, False, True, WEIGHT_ADDR, INPUT_ADDR, RESULT_ADDR
    )
    serialisable = {
        "scheduler": sched["scheduler"],
        "buffer_capacity_words": sched["buffer_capacity_words"],
        "buffer_layout": sched["buffer_layout"],
        "peak_live_words": sched["peak_live_words"],
        "spill_events": sched["spill_events"],
        "spill_count": sched["spill_count"],
    }
    blob = json.dumps(serialisable)
    assert json.loads(blob) == serialisable
