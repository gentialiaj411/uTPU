import json

from run_pruning_policy_heldout import DEFAULT_OUTPUT, run


def test_pruning_policy_heldout_schema_and_disjoint_split(tmp_path):
    out = tmp_path / "policy.json"
    payload = run(output=out)
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert payload["version"] == 1
    assert payload["split"]["shape_sets_disjoint"] is True
    assert not set(map(tuple, payload["split"]["train_shapes"])) & set(map(tuple, payload["split"]["test_shapes"]))
    assert len(payload["per_heldout_shape"]) == 5
    for row in payload["per_heldout_shape"]:
        assert row["live_measurement_count_including_baseline"] == row["retained_candidate_count"] + 1
        assert isinstance(row["best_in_menu_retained"], bool)
        assert row["regret_pct"] >= 0.0


def test_committed_pruning_policy_schema_lock():
    blob = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert {"version", "methodology", "split", "fit", "per_heldout_shape", "aggregate"} <= set(blob)
    assert blob["split"]["shape_sets_disjoint"] is True
    assert blob["aggregate"]["heldout_shape_count"] == len(blob["per_heldout_shape"])
