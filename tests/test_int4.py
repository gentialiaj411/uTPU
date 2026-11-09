import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from tools.decode.pack_decode_image import tensor_bytes

def test_int4_round_trip_nibbles():
    codes = np.array([-8, -7, -1, 0, 1, 7, 6, -2], dtype=np.int8)
    packed = (codes[0::2] & 15) | ((codes[1::2] & 15) << 4)
    got = np.empty_like(codes)
    got[0::2] = (packed & 15).astype(np.int8); got[1::2] = (packed >> 4).astype(np.int8)
    got[got >= 8] -= 16
    assert np.array_equal(got, codes)

def test_default_pack_hash_is_repeatable():
    import subprocess
    subprocess.run([sys.executable, "tools/decode/pack_decode_image.py", "--model", "15m"], cwd=ROOT, check=True)
    p = ROOT / "build/decode/decode_image_15m/image.bin"
    h1 = hashlib.sha256(p.read_bytes()).hexdigest()
    subprocess.run([sys.executable, "tools/decode/pack_decode_image.py", "--model", "15m"], cwd=ROOT, check=True)
    h2 = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h1 == h2

def test_study_baseline_and_golden_consistency():
    r = json.loads((ROOT / "bench/results/int4_study.json").read_text())
    b = next(x for x in r["rows"] if x["config"] == "measured_baseline_integer")
    assert b["int_bpb"] == 0.5623131903694261
    assert b["stream_B_token_unpruned"] == 10684413
    assert r["group_size"] == 32

def test_study_quality_results_and_decision():
    r = json.loads((ROOT / "bench/results/int4_study.json").read_text())
    rows = {x["config"]: x for x in r["rows"]}
    assert r["status"] == "complete_no_go"
    assert set(("everything", "qkv", "wo", "w2", "all_attention", "w2_qkv")) <= set(rows)
    assert rows["everything"]["int_bpb"] == 0.5898326711641192
    assert rows["wo"]["int_bpb"] == 0.5706466061940394
    assert r["pruned_int4"]["bits_per_byte"] == 0.5918510899527069
    assert r["decision"]["result"] == "NO-GO"
    assert r["decision"]["candidate_pruned_speedup_x"] >= 1.2
