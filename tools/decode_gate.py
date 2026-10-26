from __future__ import annotations
import json,sys
from pathlib import Path
from tools.decode.test_decode_structure import check
ROOT=Path(__file__).resolve().parents[1]
def main():
 r=check(ROOT); unmet=[]
 required=['decode_gemv_rtl','decode_act_quant_rtl','decode_rmsnorm_rtl','decode_rope_rtl','decode_silu_mul_rtl','decode_residual_rtl','decode_attention_rtl','decode_classifier_rtl','decode_top_rtl','decode_e2e_rtl']
 for n in required:
  p=ROOT/'bench/results'/f'{n}.json'
  if not p.exists(): unmet.append(f'missing:{p.name}')
  elif json.loads(p.read_text()).get('status')!='pass': unmet.append(f'{n}:not_pass')
 if r.get('status')!='pass': unmet.extend(r.get('errors',[]))
 out={'exit_code':1 if unmet else 0,'structural':r,'unmet':unmet}; print(json.dumps(out,indent=2)); sys.exit(out['exit_code'])
if __name__=='__main__': main()
