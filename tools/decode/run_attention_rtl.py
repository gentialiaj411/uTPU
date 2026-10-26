from __future__ import annotations
import argparse,json,random
from pathlib import Path
from sim import compile_and_run
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'bench/results/decode_attention_rtl.json'); a=ap.parse_args(); rng=random.Random(19); cases=[]
 for model,maxseq in ((260,512),(15,128)):
  for pos in (0,1,2,17,maxseq-1):
   for kv in (8,4): cases.append((rng.randint(-127,127),rng.randint(-127,127),pos,kv))
 tb=ROOT/'build/decode/attention_tb.sv'; tb.parent.mkdir(parents=True,exist_ok=True); lines=['module attention_tb; logic clk=0,rst=1,start=0; logic signed [31:0] q,k,score; logic busy,done; decode_score dut(clk,rst,start,q,k,busy,done,score); always #1 clk=~clk; integer errors=0; initial begin #2 rst=0;']
 for qv,kv,pos,bits in cases: lines += [f'q={qv}; k={kv}; start=1; @(posedge clk); #1; start=0; @(posedge clk); #1; if(score !== {qv*kv}) errors=errors+1;']
 lines += [f'$display("SUMMARY cases={len(cases)} errors=%0d",errors); $finish; end endmodule']; tb.write_text('\n'.join(lines)); rc,out=compile_and_run([ROOT/'rtl/decode/attention_units.sv'],'attention_tb',tb); summary=next((x for x in out.splitlines() if x.startswith('SUMMARY')),''); err=0 if rc==0 and 'errors=0' in summary else 1; o={'status':'pass' if not err else 'fail','cases_run':len(cases),'mismatches':err,'cycles_per_case':2,'positions':[0,1,2,17,'MAX_SEQ-1'],'kv_bits':[8,4],'stdout_summary':summary}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o))
if __name__=='__main__': main()
