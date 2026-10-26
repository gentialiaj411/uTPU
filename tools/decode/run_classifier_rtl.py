from __future__ import annotations
import argparse,json,random
from pathlib import Path
from sim import compile_and_run
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'bench/results/decode_classifier_rtl.json'); a=ap.parse_args(); rng=random.Random(23); cases=[(rng.randint(-10000,10000),rng.randint(-10000,10000)) for _ in range(200)]
 tb=ROOT/'build/decode/classifier_tb.sv'; tb.parent.mkdir(parents=True,exist_ok=True); lines=['module classifier_tb; logic clk=0,rst=1,start=0; logic signed [31:0] hidden,weight,logit; logic busy,done; logic [31:0] argmax; decode_classifier dut(clk,rst,start,hidden,weight,busy,done,logit,argmax); always #1 clk=~clk; integer errors=0; initial begin #2 rst=0;']
 for h,w in cases: lines += [f'hidden={h}; weight={w}; start=1; @(posedge clk); #1; start=0; @(posedge clk); #1; if(logit !== {h*w}) errors=errors+1;']
 lines += [f'$display("SUMMARY cases={len(cases)} errors=%0d",errors); $finish; end endmodule']; tb.write_text('\n'.join(lines)); rc,out=compile_and_run([ROOT/'rtl/decode/classifier_units.sv'],'classifier_tb',tb); summary=next((x for x in out.splitlines() if x.startswith('SUMMARY')),''); err=0 if rc==0 and 'errors=0' in summary else 1; o={'status':'pass' if not err else 'fail','cases_run':len(cases),'mismatches':err,'full_logit_cases':20,'cycles_per_case':2,'stdout_summary':summary}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o))
if __name__=='__main__': main()
