from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=Path('bench/results/decode_e2e_rtl.json')); a=ap.parse_args(); sim=shutil.which('iverilog')
 o={'models':{'stories260K':{'prompts':5,'generated_tokens':64,'kv_bits':[8,4]},'stories15M':{'prompts':1,'generated_tokens':8,'kv_bits':[8,4]}},'throttles_bytes_per_cycle':[0.5,1,2],'cycles_per_token':None,'token_mismatches':None,'logit_mismatches':None,'decoded_text':None,'status':'not_run','simulator':sim or 'missing','blocker':'iverilog is not installed' if sim is None else 'runner compilation pending'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o,indent=2))
if __name__=='__main__': main()
