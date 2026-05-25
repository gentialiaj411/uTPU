import json
from pathlib import Path

from cost_model_regression import REGRESSION_SHAPES, replay_regression_report

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "bench" / "results" / "cost_model_regression.json"


def test_cost_model_regression_artifact_schema():
    assert ARTIFACT_PATH.exists(), f"Missing artifact: {ARTIFACT_PATH}"
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    required = {
        "version",
        "generated_at_utc",
        "git_sha",
        "source_calibration_json",
        "shape_source",
        "autotuner_reference",
        "regression_shape_count",
        "regression_shapes",
        "schedule_candidate_count",
        "baseline_coefficients",
        "baseline",
        "methodology",
        "per_shape",
    }
    assert required.issubset(report.keys())
    assert report["regression_shape_count"] == 24
    assert len(report["regression_shapes"]) == 24
    assert report["schedule_candidate_count"] == 16
    assert report["shape_source"] == "firmware/host/calibrate_cost_model.py::_shape_grid() unique layer (in,out) pairs"
    assert report["autotuner_reference"] == "firmware/host/evaluate_pruned_autotuner.py"

    frozen = {(int(s["in_features"]), int(s["out_features"])) for s in report["regression_shapes"]}
    assert frozen == set(REGRESSION_SHAPES)

    baseline = report["baseline"]
    for key in (
        "median_abs_percent_error",
        "p95_abs_percent_error",
        "max_abs_percent_error",
        "regression_tolerance_pct_points",
        "regression_ceiling_abs_percent_error",
    ):
        assert key in baseline
        assert float(baseline[key]) >= 0.0

    assert float(baseline["regression_ceiling_abs_percent_error"]) == float(
        baseline["median_abs_percent_error"] + baseline["regression_tolerance_pct_points"]
    )

    for row in report["per_shape"]:
        assert len(row["schedules"]) == 16
        assert "exhaustive_best_schedule" in row
        assert "cost_model_rank_of_exhaustive_best" in row
        assert 1 <= int(row["cost_model_rank_of_exhaustive_best"]) <= 16


def test_cost_model_regression_replay_within_envelope():
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    replay = replay_regression_report(report)
    assert replay["within_envelope"] is True, (
        "Cost model replay median abs percent error "
        f"{replay['replay_metrics']['median_abs_percent_error']:.4f}% "
        f"exceeds ceiling {replay['regression_ceiling_abs_percent_error']:.4f}% "
        f"(baseline {replay['baseline_median_abs_percent_error']:.4f}% + "
        f"tolerance {replay['regression_tolerance_pct_points']:.4f} pp). "
        "Regenerate bench/results/cost_model_regression.json after intentional cost-model changes."
    )
