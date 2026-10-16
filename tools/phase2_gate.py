"""Machine-check the Phase-2 vector ISA completion contract."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from phase_fingerprints import PHASE2, fingerprint
RESULT=ROOT/"bench/results/tiny_transformer_vec_isa.json"
AUDIT=ROOT/"bench/results/isa_opcode_audit.json"; HASHES=ROOT/"bench/results/legacy_isa_hashes.json"
HOST=ROOT/"build/host_suite/summary.json"
TESTS=["firmware/host/test_tiny_transformer_vec_isa.py","firmware/host/test_isa_simulator.py","firmware/host/test_descriptor_isa_tile_gemm.py","firmware/host/test_buffer_resident_lowering.py","firmware/host/test_buffer_resident_deployment.py"]
def main():
    bad=[]
    if subprocess.run([sys.executable,"tools/isa_opcode_audit.py"],cwd=ROOT).returncode: bad.append("ISA opcode audit")
    if subprocess.run([sys.executable,"tools/phase1_gate.py"],cwd=ROOT).returncode: bad.append("phase1 gate")
    text=(ROOT/"docs/transformer_on_utpu_design.md").read_text(encoding="utf-8")
    for marker in ("## 5. Frozen vector descriptor ISA (Phase 2)","### 5.1 Parameter records","### 5.2 Subop shapes, aliasing, and decode errors"):
        if marker not in text: bad.append("design marker "+marker)
    if subprocess.run([sys.executable,"-m","pytest",*TESTS,"-q","-p","no:cacheprovider"],cwd=ROOT).returncode: bad.append("vector/legacy pytest")
    if subprocess.run([sys.executable,"firmware/host/run_isa_rtl_bitmatch.py"],cwd=ROOT).returncode: bad.append("ISA/RTL bitmatch")
    if not RESULT.is_file(): bad.append("vector result missing")
    else:
        r=json.loads(RESULT.read_text(encoding="utf-8")); source=fingerprint(PHASE2)
        if r.get("source_tree")!=source: bad.append("vector result source fingerprint")
        if r.get("sequences_used",0)<512: bad.append("fewer than 512 sequences")
        if r.get("rsqrt_lut_bin_coverage",{}).get("covered")!=192: bad.append("rsqrt coverage")
        for name,item in {**r.get("per_op",{}),**r.get("stages",{})}.items():
            if item.get("mismatches") != 0: bad.append(f"mismatches {name}")
    if not AUDIT.is_file() or json.loads(AUDIT.read_text(encoding="utf-8")).get("collisions") != []: bad.append("opcode collisions")
    if not HASHES.is_file(): bad.append("legacy hash artifact missing")
    elif not json.loads(HASHES.read_text(encoding="utf-8")).get("unchanged",False): bad.append("legacy hashes changed")
    if not HOST.is_file(): bad.append("host suite summary missing")
    dirty=subprocess.check_output(["git","status","--porcelain","--","firmware/host","tools"],cwd=ROOT,text=True).strip()
    if dirty: bad.append("uncommitted firmware/host or tools changes")
    for item in bad: print("UNMET:",item)
    return bool(bad)
if __name__=="__main__": raise SystemExit(main())
