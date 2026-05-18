from pathlib import Path

import pytest

from baseline_program_size import compute_unfused_vs_fused_comparison


def test_footprint_baseline_reduction_is_locked():
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "software/model/weights").exists():
        pytest.skip("model weights artifact missing")

    report = compute_unfused_vs_fused_comparison()
    assert int(report["fused_words"]) == 1017
    assert float(report["percent_reduction_unfused_to_fused"]) >= 60.0
