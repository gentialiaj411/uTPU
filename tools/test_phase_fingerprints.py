from phase_fingerprints import PHASE1, PHASE2, fingerprint, valid

def test_listed_dependency_fingerprint_rejects_blob_change():
    artifact={"fingerprint_dependencies": fingerprint(PHASE1)}
    artifact["fingerprint_dependencies"]["blobs"][PHASE1[0]]="0"*40
    assert not valid(artifact, PHASE1)

def test_unrelated_tool_is_not_a_phase2_dependency():
    artifact={"fingerprint_dependencies": fingerprint(PHASE2)}
    assert "tools/phase3_gate.py" not in artifact["fingerprint_dependencies"]["files"]
    assert valid(artifact, PHASE2)

def test_dependency_list_cannot_shrink():
    artifact={"fingerprint_dependencies": fingerprint(PHASE2)}
    artifact["fingerprint_dependencies"]["files"].pop()
    assert not valid(artifact, PHASE2)
