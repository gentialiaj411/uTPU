"""Generate GEMV differential evidence; requires iverilog for RTL execution."""
from __future__ import annotations
import argparse, json, shutil, subprocess
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=Path('bench/results/decode_gemv_rtl.json')); a=ap.parse_args()
    sim=shutil.which('iverilog'); rtl=Path('rtl/decode/gemv_unit.sv')
    out={'runner':'tools/decode/run_gemv_rtl.py','rtl_sources':[str(rtl)],'real_vectors':2*7*16,'random_edge_cases':2000,'mismatches':None,'cycles_per_byte':None,'status':'not_run' if sim is None else 'not_implemented','simulator':sim or 'missing','blocker':'iverilog is not installed' if sim is None else 'runner compilation pending'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
