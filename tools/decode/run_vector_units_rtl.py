"""Vector-unit RTL runner manifest; fails closed when simulator is unavailable."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
UNITS=('act_quant','rmsnorm','rope','silu_mul','residual')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=Path('bench/results')); a=ap.parse_args(); sim=shutil.which('iverilog')
 for u in UNITS:
  o={'unit':u,'real_cases':2000,'random_cases':2000,'mismatches':None,'status':'not_run' if not sim else 'not_implemented','simulator':sim or 'missing','blocker':'iverilog is not installed' if not sim else 'runner compilation pending'}
  (a.output_dir/f'decode_{u}_rtl.json').write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o))
if __name__=='__main__': main()
