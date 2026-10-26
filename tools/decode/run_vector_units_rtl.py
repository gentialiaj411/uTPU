from __future__ import annotations
import argparse,json,random
from pathlib import Path
from sim import compile_and_run
ROOT=Path(__file__).resolve().parents[2]
def run_unit(name,cases,body,decl,check):
 tb=ROOT/f'build/decode/{name}_tb.sv'; lines=[f'module {name}_tb; logic clk=0,rst=1,start=0,busy,done; integer errors=0; {decl} dut(clk,rst,start,{body}); always #1 clk=~clk; initial begin #2 rst=0;']
 for args,exp in cases: lines += [f'{args}; start=1; @(posedge clk); #1; start=0; @(posedge clk); #1; {check.format(exp=exp)}']
 lines += [f'$display("SUMMARY cases={len(cases)} errors=%0d",errors); $finish; end endmodule']; tb.write_text('\n'.join(lines)); rc,out=compile_and_run([ROOT/'rtl/decode/vector_units.sv'],f'{name}_tb',tb); summary=next((x for x in out.splitlines() if x.startswith('SUMMARY')),''); return rc,summary
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=ROOT/'bench/results'); a=ap.parse_args(); rng=random.Random(11)
 specs=[]
 for name in ('act_quant','rmsnorm','rope','silu_mul','residual'):
  cases=[]
  count = 512 if name=='rope' else (3000 if name=='rmsnorm' else 4000)
  for _ in range(count):
   if name=='act_quant':
    v=rng.randint(-100000,100000); rec=1<<24; cases.append((f'value={v}; reciprocal={rec}',max(-128,min(127,v))))
   elif name=='rmsnorm':
    v=rng.randint(-10000,10000); g=rng.randint(0,4096); cases.append((f'value={v}; gamma={g}',(v*g)>>12))
   elif name=='rope':
    x,y,c,s=[rng.randint(-10000,10000) for _ in range(4)]; cases.append((f'x={x}; y={y}; c={c}; s={s}',((x*c-y*s)>>15,(x*s+y*c)>>15)))
   elif name=='silu_mul':
    x,y=[rng.randint(-10000,10000) for _ in range(2)]; cases.append((f'silu={x}; w3={y}',x*y))
   else:
    x,y=[rng.randint(-100000,100000) for _ in range(2)]; cases.append((f'a={x}; b={y}',x+y))
  if name=='act_quant': decl='logic signed [31:0] value; logic [31:0] reciprocal; logic signed [7:0] code; decode_act_quant'; body='value,reciprocal,busy,done,code'; check='if(code !== {exp}) errors=errors+1;'
  elif name=='rmsnorm': decl='logic signed [31:0] value,gamma,out_value; decode_rmsnorm'; body='value,gamma,busy,done,out_value'; check='if(out_value !== {exp}) errors=errors+1;'
  elif name=='rope': decl='logic signed [31:0] x,y,c,s,out_x,out_y; decode_rope'; body='x,y,c,s,busy,done,out_x,out_y'; check='if(out_x !== {exp[0]} || out_y !== {exp[1]}) errors=errors+1;'
  elif name=='silu_mul': decl='logic signed [31:0] silu,w3,out_value; decode_silu_mul'; body='silu,w3,busy,done,out_value'; check='if(out_value !== {exp}) errors=errors+1;'
  else: decl='logic signed [31:0] a,b,out_value; decode_residual'; body='a,b,busy,done,out_value'; check='if(out_value !== {exp}) errors=errors+1;'
  rc,summary=run_unit(name,cases,body,decl,check); err=0 if rc==0 and 'errors=0' in summary else 1; o={'status':'pass' if not err else 'fail','cases_run':len(cases),'mismatches':err,'cycles_per_case':2,'stdout_summary':summary}; (a.output_dir/f'decode_{name}_rtl.json').write_text(json.dumps(o,indent=2)+'\n'); print(name,o)
if __name__=='__main__': main()
