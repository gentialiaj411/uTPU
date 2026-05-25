"""Phase 7 tests: cuBLAS / Inductor baseline harness.

These tests intentionally do **not** gate on a performance number —
the gap depends on the GPU + driver of whichever host regenerates the
artifact. They lock the *contract* of the harness so the writeup and
the CLAIMS_MATRIX row cannot drift from the JSON layout:

* The locked shape set (``run_cublas_baseline.SHAPES``) is exactly the
  six shapes the writeup references and contains documented small /
  medium / large regions so a single number can't be cherry-picked.
* The artifact's top-level keys, methodology block, and per-shape
  layout are identical in ``status="ok"`` and ``status="cuda_unavailable"``
  modes.
* The dtype caveats are present, explicit, and call out the FP32
  Inductor / INT32 cuBLAS mismatches so a downstream reader cannot
  mistake the comparison for like-for-like.
* The Torch subprocess script exists, parses its ``--shapes-json``
  argument, and emits a well-formed stub when CUDA is missing.

Running the harness on the current (no-CUDA) host produces a stub
artifact with ``status="cuda_unavailable"``; these tests validate
*that* stub, plus they validate the schema contract that the live
artifact will satisfy when re-generated on a CUDA host.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from run_cublas_baseline import (  # noqa: E402
    DEFAULT_ITERS,
    DEFAULT_WARMUP,
    OUTPUT_JSON,
    SHAPES,
    SUBPROCESS_SCRIPT,
    _methodology_block,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBPROCESS_PATH = Path(SUBPROCESS_SCRIPT)


# ---------------------------------------------------------------------------
# Locked shape set
# ---------------------------------------------------------------------------


def test_locked_shape_set_is_exactly_six_shapes():
    assert len(SHAPES) == 6
    for entry in SHAPES:
        assert set(entry.keys()) == {"M", "K"}
        assert isinstance(entry["M"], int) and entry["M"] > 0
        assert isinstance(entry["K"], int) and entry["K"] > 0


def test_locked_shape_set_spans_small_medium_large_regions():
    sizes = [s["M"] * s["K"] for s in SHAPES]
    assert min(sizes) <= 16 * 16, "no small-regime shape in SHAPES"
    assert max(sizes) >= 512 * 256, "no large-regime shape in SHAPES"
    assert len({s["M"] * s["K"] for s in SHAPES}) >= 5, (
        "SHAPES has too many duplicates; a single number could be cherry-picked"
    )


def test_locked_shape_set_includes_anchor_shapes():
    pairs = {(s["M"], s["K"]) for s in SHAPES}
    assert (16, 16) in pairs
    assert (512, 512) in pairs


# ---------------------------------------------------------------------------
# Methodology block (independent of CUDA availability)
# ---------------------------------------------------------------------------


def test_methodology_block_contains_required_fields():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    required = {
        "shape_set",
        "shape_set_description",
        "warmup_iterations",
        "timed_iterations",
        "timing_protocol",
        "tflops_definition",
        "dtype_caveats",
        "isolation",
        "rng_seed_per_shape",
        "scope",
    }
    assert required <= set(block.keys())
    assert block["shape_set"] == SHAPES
    assert block["warmup_iterations"] == DEFAULT_WARMUP
    assert block["timed_iterations"] == DEFAULT_ITERS


def test_methodology_dtype_caveats_call_out_inductor_and_cublas_dtypes():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    caveats_blob = " ".join(block["dtype_caveats"]).lower()
    assert "int8" in caveats_blob and "int32" in caveats_blob and "int4" in caveats_blob
    assert "fp32" in caveats_blob or "float32" in caveats_blob
    assert "inductor" in caveats_blob
    assert "cublas" in caveats_blob


def test_methodology_explicitly_disclaims_hardware_claim():
    block = _methodology_block(SHAPES, DEFAULT_WARMUP, DEFAULT_ITERS)
    assert "no physical-board claim" in block["scope"].lower() or \
           "sim/host-measured" in block["scope"].lower()


# ---------------------------------------------------------------------------
# Live artifact schema lock (works for both ok + cuda_unavailable modes)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> Dict[str, Any]:
    if not OUTPUT_JSON.exists():
        pytest.skip(f"baseline artifact not regenerated yet: {OUTPUT_JSON}")
    return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))


def test_artifact_has_top_level_contract_keys(artifact: Dict[str, Any]):
    required = {
        "version",
        "generated_at_utc",
        "git_sha",
        "methodology",
        "status",
        "shapes_requested",
        "per_shape",
        "aggregate",
    }
    assert required <= set(artifact.keys())
    assert artifact["version"] == 1
    assert artifact["status"] in {"ok", "cuda_unavailable"}


def test_artifact_shape_set_matches_locked_constant(artifact: Dict[str, Any]):
    assert artifact["shapes_requested"] == SHAPES
    assert artifact["methodology"]["shape_set"] == SHAPES


def test_artifact_methodology_dtype_caveats_present_in_either_mode(artifact: Dict[str, Any]):
    caveats = artifact["methodology"].get("dtype_caveats")
    assert isinstance(caveats, list) and len(caveats) >= 3


def test_cuda_unavailable_stub_has_actionable_instructions(artifact: Dict[str, Any]):
    if artifact["status"] != "cuda_unavailable":
        pytest.skip("artifact regenerated with status='ok'; stub gate not applicable")
    instr = artifact.get("instructions")
    assert isinstance(instr, list) and len(instr) >= 3
    blob = " ".join(instr).lower()
    assert "cuda" in blob
    assert "run_cublas_baseline.py" in blob


def test_ok_artifact_per_shape_layout_matches_lock(artifact: Dict[str, Any]):
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; live shape gate not applicable")
    per = artifact["per_shape"]
    assert len(per) == len(SHAPES)
    for entry in per:
        assert "shape" in entry and set(entry["shape"].keys()) == {"M", "K", "N"}
        assert "utpu" in entry
        utpu = entry["utpu"]
        for key in [
            "backend", "dtype_W", "dtype_x", "dtype_accum", "dtype_out",
            "kernel_ms", "samples_ms", "int_mac_tflops_median",
            "bit_exact_match_vs_numpy_reference",
        ]:
            assert key in utpu, f"utpu entry missing field {key}"
        for stat in ["mean", "median", "stdev", "min", "max", "p95", "samples"]:
            assert stat in utpu["kernel_ms"], f"utpu.kernel_ms missing {stat}"


def test_ok_artifact_aggregate_gap_keys_match_lock(artifact: Dict[str, Any]):
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; aggregate gate not applicable")
    agg = artifact["aggregate"]
    assert agg is not None
    for key in [
        "cublas_gap_pct_median_of_shapes",
        "cublas_gap_pct_mean_of_shapes",
        "cublas_gap_pct_max_of_shapes",
        "cublas_gap_pct_min_of_shapes",
        "inductor_gap_pct_median_of_shapes",
        "inductor_gap_pct_mean_of_shapes",
        "shapes_compared_vs_cublas",
        "shapes_compared_vs_inductor",
    ]:
        assert key in agg, f"aggregate missing field {key}"


def test_ok_artifact_records_bit_exact_match_per_shape(artifact: Dict[str, Any]):
    """On a real GPU run the uTPU kernel must remain bit-exact vs the
    Python NumPy reference. We do not gate the cuBLAS / Inductor
    numerical agreement here (different dtypes), only our own kernel."""
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; bit-exact gate not applicable")
    for entry in artifact["per_shape"]:
        assert entry["utpu"]["bit_exact_match_vs_numpy_reference"] is True, (
            f"uTPU kernel not bit-exact vs NumPy reference for shape {entry['shape']}"
        )


def test_ok_artifact_cublas_dtype_is_explicit_no_silent_fallback(artifact: Dict[str, Any]):
    """If the Torch build cannot run INT32 cuBLAS GEMV, the subprocess
    falls back to FP32 — but the per-shape entry must record the
    fallback (``dtype_fallback_reason`` non-empty + ``backend`` named
    ``cublas_gemv_fp32_fallback`` + ``dtype_W=="float32"``) so a reader
    cannot mistake the gap for an apples-to-apples INT32 comparison.
    """
    if artifact["status"] != "ok":
        pytest.skip("artifact regenerated as stub; live dtype contract not applicable")
    for entry in artifact["per_shape"]:
        cublas = entry.get("cublas")
        if cublas is None or "error" in cublas:
            continue
        dtype_W = cublas.get("dtype_W")
        if dtype_W == "int32":
            assert cublas.get("backend") == "cublas_gemv_int32"
            assert cublas.get("dtype_accum") == "int32"
            assert "dtype_fallback_reason" not in cublas
        elif dtype_W == "float32":
            assert cublas.get("backend") == "cublas_gemv_fp32_fallback", (
                f"shape {entry['shape']}: fp32 dtype without fp32_fallback "
                f"backend tag — this would silently downgrade the dtype "
                f"comparison; got backend={cublas.get('backend')!r}"
            )
            assert cublas.get("dtype_accum") == "float32"
            reason = cublas.get("dtype_fallback_reason")
            assert isinstance(reason, str) and len(reason) > 0, (
                f"shape {entry['shape']}: fp32 fallback path missing "
                f"dtype_fallback_reason — refusing silent dtype switch."
            )
        else:
            pytest.fail(
                f"shape {entry['shape']}: cublas dtype_W={dtype_W!r} is "
                f"neither int32 (locked dtype) nor float32 (documented "
                f"fallback). Unknown dtype path is not allowed."
            )


# ---------------------------------------------------------------------------
# Torch subprocess (locked entry point)
# ---------------------------------------------------------------------------


def test_torch_subprocess_script_exists_and_is_python():
    assert SUBPROCESS_PATH.exists(), f"missing {SUBPROCESS_PATH}"
    text = SUBPROCESS_PATH.read_text(encoding="utf-8")
    assert "def run_baselines" in text
    assert "--shapes-json" in text


def test_torch_subprocess_emits_well_formed_stub_when_cuda_missing():
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not importable in this environment")
    import torch as torch_mod
    if torch_mod.cuda.is_available():
        pytest.skip("CUDA available; stub path not exercisable")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(SUBPROCESS_PATH),
                "--shapes-json",
                json.dumps([{"M": 16, "K": 16}]),
                "--output",
                str(tmp_path),
                "--warmup",
                "1",
                "--iters",
                "1",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, f"subprocess failed: {proc.stderr}"
        payload = json.loads(tmp_path.read_text(encoding="utf-8"))
        assert payload["status"] in {"cuda_unavailable", "torch_unavailable"}
        assert payload.get("shapes_requested") == [{"M": 16, "K": 16}]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
