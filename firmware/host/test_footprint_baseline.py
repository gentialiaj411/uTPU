from baseline_program_size import compute_unfused_vs_fused_comparison


def test_footprint_baseline_reduction_is_locked():
    report = compute_unfused_vs_fused_comparison()
    assert int(report["fused_words"]) == 1017
    assert float(report["percent_reduction_unfused_to_fused"]) >= 60.0
