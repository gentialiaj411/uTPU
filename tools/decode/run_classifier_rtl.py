from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=Path('bench/results/decode_classifier_rtl.json')); a=ap.parse_args(); sim=shutil.which('iverilog')
 o={'models':['stories260K','stories15M'],'real_tokens':{'stories260K':200,'stories15M':200},'full_logits_260k':20,'full_logits_15m':20,'argmax_matches':None,'logit_mismatches':None,'status':'not_run','simulator':sim or 'missing','blocker':'iverilog is not installed' if sim is None else 'runner compilation pending'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o,indent=2))
if __name__=='__main__': main()
