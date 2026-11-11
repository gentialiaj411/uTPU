"""Lock resume-facing numeric claims to committed benchmark artifacts.

claims/claims.json maps every registered numeric claim string to one artifact field and a formatting rule.
The tests load each artifact, format the value, and require (a) the formatted string to equal the registered
claim and (b) every line of each referenced doc that contains the anchor to contain the claim string. Editing a
digit in a doc, or regenerating an artifact so a value moves, fails here.
"""

import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CLAIMS_PATH = ROOT / "claims" / "claims.json"
DOCS = [ROOT / "README.md", ROOT / "RESUME_CLAIMS.md", ROOT / "CLAIMS_MATRIX.md", ROOT / "INTERVIEW_GUIDE.md"]
AUTOTUNER = ROOT / "bench" / "results" / "pruned_autotuner_replay.json"


def _registry():
    return json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))["claims"]


def _lookup(obj, path):
    """Resolve dot keys, [index] and [key=value] selectors."""
    node = obj
    for part in path.split("."):
        name = re.match(r"[^\[]*", part).group(0)
        if name:
            node = node[name]
        for sel in re.findall(r"\[([^\]]*)\]", part):
            if "=" in sel:
                key, want = sel.split("=", 1)
                node = next(item for item in node if str(item.get(key)) == want)
            else:
                node = node[int(sel)]
    return node


def _format(value, fmt, obj):
    kind = fmt["kind"]
    if kind == "int":
        return f"{int(value)}{fmt.get('suffix', '')}"
    if kind == "ratio":
        den_spec = fmt["denominator"]
        den = len(_lookup(obj, den_spec[4:])) if den_spec.startswith("len:") else _lookup(obj, den_spec)
        return f"{int(value)}/{int(den)}"
    if kind == "upper_bound":
        threshold = float(fmt["threshold"])
        assert float(value) < threshold, f"value {value} is not below {threshold}"
        return f"<{threshold:g}{fmt.get('suffix', '')}"
    if kind == "fixed":
        text = f"{float(value) * fmt.get('scale', 1):.{int(fmt['decimals'])}f}"
        if fmt.get("prefix") and not text.startswith("-"):
            text = fmt["prefix"] + text
        return text + fmt.get("suffix", "")
    raise ValueError(f"unknown format kind {kind!r}")


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


def test_claim_registry_ids_are_unique_and_paths_exist():
    claims = _registry()
    ids = [c["id"] for c in claims]
    assert len(ids) == len(set(ids))
    for c in claims:
        assert (ROOT / c["artifact"]).is_file(), c["artifact"]
        for ref in c["refs"]:
            assert (ROOT / ref["doc"]).is_file(), ref["doc"]


@pytest.mark.parametrize("claim", _registry(), ids=lambda c: c["id"])
def test_registered_claim_matches_artifact_and_doc(claim):
    obj = json.loads((ROOT / claim["artifact"]).read_text(encoding="utf-8"))
    formatted = _format(_lookup(obj, claim["field"]), claim["format"], obj)
    assert formatted == claim["claim"], f"{claim['artifact']}:{claim['field']} formats to {formatted!r}"
    for ref in claim["refs"]:
        lines = (ROOT / ref["doc"]).read_text(encoding="utf-8").splitlines()
        matched = [line for line in lines if ref["anchor"] in line]
        assert matched, f"{ref['doc']}: no line contains anchor {ref['anchor']!r}"
        stale = [line for line in matched if claim["claim"] not in line]
        assert not stale, f"{ref['doc']}: lines with anchor {ref['anchor']!r} lack {claim['claim']!r}: {stale[:2]}"


def test_frontend_evidence_bundle_matches_generator():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import build_frontend_data as builder
    finally:
        sys.path.pop(0)
    claims, artifacts_read = builder.collect_claims()
    evidence = json.loads((ROOT / "frontend" / "public" / "data" / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["claims"] == json.loads(json.dumps(claims)), "evidence.json is stale; rerun tools/build_frontend_data.py"
    assert evidence["meta"]["claim_count"] == len(claims)
    assert evidence["meta"]["artifact_count"] == len(artifacts_read)
