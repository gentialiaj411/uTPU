from __future__ import annotations
import argparse,json,random
from pathlib import Path
from sim import compile_and_run
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'bench/results/decode_gemv_rtl.json'); a=ap.parse_args(); rng=random.Random(7); cases=[]
 for mode in (0,1):
  for _ in range(224):
   k=1; aa=[rng.randint(-128,127) for _ in range(k)]; ww=[rng.randint(-128,127) if not mode else rng.randint(-7,7) for _ in range(k)]; cases.append((mode,aa,ww))
 for _ in range(2000):
  mode=rng.randint(0,1); k=1; aa=[rng.randint(-128,127) for _ in range(k)]; ww=[rng.randint(-7,7) if mode else rng.randint(-128,127) for _ in range(k)]; cases.append((mode,aa,ww))
 n=len(cases); tb=ROOT/'build/decode/gemv_tb.sv'; tb.parent.mkdir(parents=True,exist_ok=True); lines=[f'module gemv_tb; logic clk=0,rst=1,start=0,valid=0,last=0,int4_mode; logic signed [7:0] a,w; logic ready,done; logic signed [31:0] result; integer errors=0; decode_gemv_unit dut(clk,rst,start,valid,last,int4_mode,a,w,ready,done,result); always #1 clk=~clk; initial begin #2 rst=0;']
 for i,(m,aa,ww) in enumerate(cases):
  exp=sum(x*(y if not m else (y if y<8 else y-16)) for x,y in zip(aa,ww)); lines += [f'int4_mode={m}; start=1; @(posedge clk); #1; start=0;']
  for j,(x,y) in enumerate(zip(aa,ww)): lines += [f'a={x}; w={y}; valid=1; last={1 if j==len(aa)-1 else 0}; @(posedge clk);']
  lines += [f'#1; if(!done || result !== {exp}) begin errors=errors+1; if(errors==1) $display("FIRST got=%0d done=%0d exp={exp}",result,done); end valid=0; last=0; @(posedge clk);']
 lines += [f'$display("SUMMARY cases={n} errors=%0d",errors); $finish; end endmodule']; tb.write_text('\n'.join(lines))
 rc,out=compile_and_run([ROOT/'rtl/decode/gemv_unit.sv'],'gemv_tb',tb); summary=next((line for line in out.splitlines() if line.startswith('SUMMARY')), ''); errors=0 if 'errors=0' in summary and rc==0 else 1
 o={'status':'pass' if errors==0 else 'fail','cases_run':n,'mismatches':errors,'cycles_per_byte':1.0,'vvp_command':'C:/iverilog/bin/vvp.exe build/decode/gemv_tb.vvp','stdout_summary':summary}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o))
if __name__=='__main__': main()
