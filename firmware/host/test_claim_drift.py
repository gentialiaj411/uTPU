"""Lock resume-facing numeric claims to committed benchmark artifacts."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTOTUNER = ROOT / "bench" / "results" / "pruned_autotuner_replay.json"
DOCS = [ROOT / "README.md", ROOT / "RESUME_CLAIMS.md", ROOT / "CLAIMS_MATRIX.md", ROOT / "INTERVIEW_GUIDE.md"]
AUDIT = ROOT / "bench" / "fixtures" / "numeric_claim_audit.json"


def test_autotuner_claim_matches_committed_replay_artifact():
    report = json.loads(AUTOTUNER.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["full_candidate_count"] == 16
    assert round(summary["search_reduction_x"], 2) == 3.12
    assert summary["max_quality_regression_pct"] < 0.5
    claim = "3.12x candidate-count reduction over a 16-schedule (threads x unroll) menu on calibration replay, max regression <0.5%"
    for path in (ROOT / "README.md", ROOT / "RESUME_CLAIMS.md", ROOT / "INTERVIEW_GUIDE.md"):
        assert claim in path.read_text(encoding="utf-8"), path


def test_claim_docs_reject_stale_or_unsupported_wording():
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    assert "3.25x" not in text
    assert "4.92` measured candidates" not in text
    assert "87.5%** median modeled FLOP reduction" not in (ROOT / "RESUME_CLAIMS.md").read_text(encoding="utf-8")
    assert "liveness-based memory-plan analysis" in text
    assert "16-bit instruction words; INT4/INT8 datapath" in text


def test_all_four_numeric_claim_docs_have_audited_artifact_and_command_coverage():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["version"] == 1
    for path in DOCS:
        spec = audit["documents"][path.name]
        numeric_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if any(c.isdigit() for c in line)]
        assert len(numeric_lines) == spec["numeric_line_count"], (
            f"{path.name} numeric-claim inventory changed; update {AUDIT.name} with artifact and command coverage"
        )
        assert spec["artifact_roots"] and spec["command"]
        for root in spec["artifact_roots"]:
            assert (ROOT / root).exists(), root
