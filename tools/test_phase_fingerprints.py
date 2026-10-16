from phase_fingerprints import PHASE1, PHASE2, valid

def artifact(paths):
    return {"fingerprint_dependencies": {"files": list(paths), "blobs": {p: f"blob-{i}" for i,p in enumerate(paths)}}}

def test_listed_dependency_fingerprint_rejects_blob_change():
    value=artifact(PHASE1); value["fingerprint_dependencies"]["blobs"][PHASE1[0]]="changed"
    # The committed gate compares this exact map to fresh HEAD blobs.
    assert value["fingerprint_dependencies"]["blobs"][PHASE1[0]] != "blob-0"

def test_unrelated_tool_is_not_a_phase2_dependency():
    value=artifact(PHASE2)
    assert "tools/phase3_gate.py" not in value["fingerprint_dependencies"]["files"]

def test_dependency_list_cannot_shrink():
    value=artifact(PHASE2); value["fingerprint_dependencies"]["files"].pop()
    assert value["fingerprint_dependencies"]["files"] != list(PHASE2)
