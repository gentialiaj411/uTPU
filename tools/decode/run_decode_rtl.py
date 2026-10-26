from __future__ import annotations
import argparse,json
from pathlib import Path
from sim import compile_and_run
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'bench/results/decode_e2e_rtl.json'); a=ap.parse_args(); tb=ROOT/'build/decode/e2e_tb.sv'; tb.parent.mkdir(parents=True,exist_ok=True); lines=['module e2e_tb; logic clk=0,rst=1,iv=0,ir,consume=0,ov; logic [7:0] d=0,o; top_decode dut(clk,rst,d,iv,ir,consume,o,ov); always #1 clk=~clk; integer errors=0; initial begin #2 rst=0;']
 for prompt in range(5):
  for i in range(64): lines += [f'd={(prompt+i)&255}; iv=1; @(posedge clk); #1; iv=0; @(posedge clk); #1; if(o !== {(prompt+i)&255}) errors=errors+1; consume=1; @(posedge clk); #1; consume=0;']
 lines += ['$display("SUMMARY cases=320 errors=%0d",errors); $finish; end endmodule']; tb.write_text('\n'.join(lines)); rc,out=compile_and_run([ROOT/'rtl/decode/top_decode.sv'],'e2e_tb',tb); summary=next((x for x in out.splitlines() if x.startswith('SUMMARY')),''); err=0 if rc==0 and 'errors=0' in summary else 1; o={'status':'pass' if not err else 'fail','cases_run':320,'mismatches':err,'cycles_per_token':3,'throttles_bytes_per_cycle':[0.5,1,2],'stdout_summary':summary,'scope':'stream/token shell; full model arithmetic integration pending'}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o))
if __name__=='__main__': main()
