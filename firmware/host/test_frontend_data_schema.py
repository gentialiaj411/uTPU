"""Schema lock for frontend evidence bundle — every claim must be fenced."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_JSON = REPO_ROOT / "frontend" / "public" / "data" / "evidence.json"
BUILD_SCRIPT = REPO_ROOT / "tools" / "build_frontend_data.py"
VALID_TIERS = frozenset({"sim", "ci", "synth", "silicon"})


def _load_evidence() -> dict:
    if not EVIDENCE_JSON.is_file():
        subprocess.check_call([sys.executable, str(BUILD_SCRIPT)], cwd=str(REPO_ROOT))
    return json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))


def test_evidence_bundle_exists():
    blob = _load_evidence()
    assert blob.get("claims"), "claims list must be non-empty"


def test_every_claim_has_tier_and_artifact():
    blob = _load_evidence()
    errors: list[str] = []
    for claim in blob["claims"]:
        cid = claim.get("id", "<missing-id>")
        tier = claim.get("tier", "")
        if tier not in VALID_TIERS:
            errors.append(f"{cid}: invalid tier {tier!r}")
        artifact = claim.get("source_artifact")
        if not artifact:
            errors.append(f"{cid}: missing source_artifact")
            continue
        path = REPO_ROOT / artifact
        if not path.is_file():
            errors.append(f"{cid}: artifact not on disk: {artifact}")
    assert not errors, "schema lock failures:\n" + "\n".join(errors)


def test_tier_legend_present():
    blob = _load_evidence()
    keys = {t["key"] for t in blob.get("tiers", [])}
    assert keys == VALID_TIERS


def test_no_silicon_claims_yet():
    blob = _load_evidence()
    silicon = [c for c in blob["claims"] if c.get("tier") == "silicon"]
    assert silicon == [], "no artifact currently maps to silicon tier"


def test_validate_claims_rejects_invalid_tier():
    tools_dir = REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import build_frontend_data as bfd  # noqa: E402

    sample = REPO_ROOT / "bench" / "results" / "fusion_payoff.json"
    if not sample.is_file():
        pytest.skip("fusion_payoff.json not present")
    bad_claim = {
        "id": "guardrail_probe",
        "tier": "not_a_tier",
        "source_artifact": "bench/results/fusion_payoff.json",
        "value": 1,
    }
    with pytest.raises(SystemExit):
        bfd.validate_claims([bad_claim])


def test_build_script_exits_zero():
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
