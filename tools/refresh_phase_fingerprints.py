"""Refresh only provenance fields after narrowing phase dependency scopes."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from phase_fingerprints import PHASE1, PHASE2, fingerprint
ROOT=Path(__file__).resolve().parents[1]
def refresh(path: Path, deps: tuple[str,...]) -> None:
    data=json.loads(path.read_text(encoding="utf-8")); data["source_tree"]=fingerprint(deps)
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("phase1","phase2"));a=p.parse_args()
    if a.phase=="phase1":
        for n in ("tiny_transformer_numerics.json","tiny_transformer_divergence.json"): refresh(ROOT/"bench/results"/n,PHASE1)
    else:
        refresh(ROOT/"bench/results/tiny_transformer_vec_isa.json",PHASE2)
    return 0
if __name__=="__main__": raise SystemExit(main())
