"""Schema + reality-floor tests for ``scheduler_rtl_crosscheck.json``.

This file locks the contract for the Phase 7 remediation P4.1 artifact
emitted by ``run_scheduler_rtl_crosscheck.py``. Two scenarios are
covered:

1. ``status="ok"`` — iverilog was available and ran the testbench; the
   ``headline`` block must include the RTL/sim cycle reduction
   permilles plus the bit-exactness invariant flag.

2. ``status="iverilog_unavailable"`` — the binary was missing; the
   artifact is a stub that records the simulator-side expected values
   so consumers can still reason about what the testbench *would*
   assert.

We additionally enforce reality floors:

* The simulator-side ``expected.cycles_saved`` is positive (otherwise
  the cross-check has nothing to verify).
* The simulator-side ``expected.fetch_bytes_invariant_simulator`` is
  ``true`` (the scheduler's bit-exactness vs naive at the ISA level).
* When ``status="ok"``: RTL_scheduled_cycles < RTL_naive_cycles,
  diff_permille <= tol_permille, and the scheduler invariant flag
  is set.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

REPO_ROOT = os.path.dirname(os.path.dirname(HOST_DIR))
ARTIFACT_PATH = os.path.join(REPO_ROOT, "bench", "results", "scheduler_rtl_crosscheck.json")


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT_PATH):
        pytest.skip(
            f"{ARTIFACT_PATH} not found; run "
            "`python firmware/host/run_scheduler_rtl_crosscheck.py` first."
        )
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_top_level_keys_present(artifact):
    for key in [
        "version",
        "generated_at_utc",
        "status",
        "expected",
        "methodology",
        "tolerance_permille",
        "host",
    ]:
        assert key in artifact, f"missing top-level key: {key}"
    assert artifact["version"] == 1
    assert artifact["status"] in {"ok", "failed", "iverilog_unavailable"}


def test_methodology_documents_headline_and_advisory_split(artifact):
    meth = artifact["methodology"]
    headline = meth["headline_assertions"]
    advisory = meth["advisory_metrics"]
    assert any("strictly less than naive" in s for s in headline)
    assert any("permille" in s for s in headline)
    assert any("RTL_naive fetch_bytes === RTL_scheduled" in s for s in headline)
    # The simulator-vs-RTL byte agreement is recorded as ADVISORY only;
    # it must not have leaked into the headline list.
    assert not any("matches sim" in s.lower() for s in headline)
    assert any("byte agreement" in s for s in advisory)


def test_methodology_summary_explains_permille_window(artifact):
    summary = artifact["methodology"]["summary"]
    assert "permille" in summary
    # Window must be honestly disclosed (we lock the value at 20).
    assert "20" in summary
    assert "2.0%" in summary
    # The simulator's 1-cycle-per-op model is the reason the absolute
    # cycle counts diverge; that must be in the summary.
    assert "1-cycle-per-op" in summary


def test_tolerance_permille_is_20(artifact):
    assert artifact["tolerance_permille"] == 20


# ---------------------------------------------------------------------------
# Simulator-side reality floors (must hold even in stub mode)
# ---------------------------------------------------------------------------


def test_expected_block_has_required_fields(artifact):
    exp = artifact["expected"]
    for key in [
        "shape",
        "array_size",
        "weight_addr",
        "input_addr",
        "result_addr",
        "naive_words",
        "sched_words",
        "naive_cycles",
        "sched_cycles",
        "cycles_saved",
        "reduction_permille",
        "fetch_bytes_n",
        "expected_fetch_bytes",
        "fetch_bytes_invariant_simulator",
    ]:
        assert key in exp, f"missing expected.{key}"
    assert exp["shape"]["out_features"] == 32
    assert exp["shape"]["in_features"] == 32
    assert exp["array_size"] == 16


def test_simulator_cycles_saved_is_positive(artifact):
    exp = artifact["expected"]
    assert exp["cycles_saved"] > 0, (
        f"expected.cycles_saved = {exp['cycles_saved']}; the simulator's "
        "predicted scheduler savings must be positive for the cross-check "
        "to have anything to verify"
    )
    assert exp["sched_cycles"] < exp["naive_cycles"]


def test_simulator_fetch_bytes_invariant_holds(artifact):
    # If this trips, the scheduler is no longer producing bit-exact
    # outputs vs naive at the ISA level — that's a hard scheduler bug
    # to investigate before re-running the RTL cross-check.
    assert artifact["expected"]["fetch_bytes_invariant_simulator"] is True


def test_simulator_reduction_permille_above_tolerance(artifact):
    exp = artifact["expected"]
    tol = artifact["tolerance_permille"]
    # Otherwise the tolerance window swallows the entire effect and the
    # test couldn't tell apart "scheduler works" from "tolerance is wide".
    assert exp["reduction_permille"] >= tol, (
        f"expected.reduction_permille ({exp['reduction_permille']}) must "
        f"be at least the tolerance window ({tol}) so a passing test is "
        "informative; pick a shape with stronger sim savings"
    )


def test_word_count_fits_shipping_prog_depth(artifact):
    # We bind the cross-check to PROG_DEPTH=1024 (the shipping default).
    # If either program grows beyond this, the testbench would silently
    # truncate via $readmemh and the RTL would execute garbage tail
    # instructions.
    exp = artifact["expected"]
    assert exp["naive_words"] <= 1024
    assert exp["sched_words"] <= 1024


# ---------------------------------------------------------------------------
# Status=="ok" reality floors (only when iverilog ran)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def headline(artifact):
    if artifact["status"] != "ok":
        pytest.skip(
            f"status={artifact['status']}; headline checks only run for "
            "a populated artifact (run `python firmware/host/"
            "run_scheduler_rtl_crosscheck.py` on a host with iverilog)"
        )
    assert "headline" in artifact
    return artifact["headline"]


def test_headline_has_required_fields(headline):
    for key in [
        "rtl_naive_cycles",
        "rtl_sched_cycles",
        "rtl_cycles_saved",
        "rtl_reduction_permille",
        "sim_reduction_permille",
        "diff_permille",
        "tol_permille",
        "scheduler_invariant_holds",
    ]:
        assert key in headline, f"missing headline.{key}"


def test_rtl_scheduled_strictly_less_than_naive(headline):
    assert headline["rtl_sched_cycles"] < headline["rtl_naive_cycles"]
    assert headline["rtl_cycles_saved"] > 0


def test_rtl_reduction_within_tolerance_of_simulator(headline):
    diff = headline["diff_permille"]
    tol = headline["tol_permille"]
    assert diff <= tol, (
        f"RTL permille reduction ({headline['rtl_reduction_permille']}) "
        f"diverges from simulator's ({headline['sim_reduction_permille']}) "
        f"by {diff} permille > tolerance {tol} permille — the "
        "sim-only 4.67% cycle-reduction claim is no longer corroborated "
        "by the RTL FSM"
    )


def test_rtl_scheduler_invariant_holds(headline):
    # RTL_naive fetch_bytes === RTL_scheduled fetch_bytes. If this
    # trips, the scheduler's bit-exactness guarantee no longer survives
    # the FSM and the Phase 5 claim must be retracted or scoped.
    assert headline["scheduler_invariant_holds"] is True


# ---------------------------------------------------------------------------
# iverilog_unavailable stub — make sure the stub is informative
# ---------------------------------------------------------------------------


def test_stub_records_instructions_when_iverilog_missing(artifact):
    if artifact["status"] != "iverilog_unavailable":
        pytest.skip("artifact is populated; stub-mode check skipped")
    assert "instructions" in artifact
    assert "iverilog_resolved" in artifact
    # Without a real run we should *not* be advertising RTL numbers.
    assert "headline" not in artifact
