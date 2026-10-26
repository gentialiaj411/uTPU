from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=Path('bench/results/decode_attention_rtl.json')); a=ap.parse_args(); sim=shutil.which('iverilog')
 o={'models':['stories260K','stories15M'],'positions':[0,1,2,17,'MAX_SEQ-1'],'kv_bits':[8,4],'real_decode_steps':None,'random_positions':None,'mismatches':None,'status':'not_run','simulator':sim or 'missing','blocker':'iverilog is not installed' if not sim else 'runner compilation pending'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o,indent=2))
if __name__=='__main__': main()
