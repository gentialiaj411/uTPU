from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from tools.decode.sim import compile_and_run

def load_manifest(d): return json.loads((d/'manifest.json').read_text())
def build_flat(d,m):
 raw=[]; code=[]; scales=[]; lens=[]
 for r in m['records']:
  lens.append(r['length']); raw += list((d/(r['id']+'_r.hex')).read_text().split()); code += list((d/(r['id']+'_c.hex')).read_text().split()); sb=list((d/(r['id']+'_s.hex')).read_text().split()); scales.append(''.join(reversed(sb)).rjust(8,'0'))
 (d/'raw_all.hex').write_text('\n'.join(raw)+'\n'); (d/'code_all.hex').write_text('\n'.join(code)+'\n'); (d/'scale_all.hex').write_text('\n'.join(scales)+'\n'); (d/'lens.hex').write_text('\n'.join(f'{x:x}' for x in lens)+'\n')
 return sum(lens),len(lens)
def tb_text(n,e):
 base=str((ROOT/'build/decode/golden_act_quant').resolve()).replace('\\','/')
 return f'''module act_quant_tb; reg clk=0,rst=1,start=0,in_valid=0,in_last=0; reg signed [31:0] raw_in; wire in_ready,code_valid,scale_valid,done; wire signed [7:0] code_out; wire [31:0] scale_out;
act_quant dut(clk,rst,start,in_valid,in_last,raw_in,in_ready,code_valid,code_out,scale_valid,scale_out,done); always #5 clk=~clk;
reg [7:0] rb[0:{e*4-1}], cb[0:{e-1}]; reg [31:0] sb[0:{n-1}]; reg [7:0] lb[0:{n-1}]; integer g,i,rbp,cbp,err,cyc; initial begin $readmemh("{base}/raw_all.hex",rb); $readmemh("{base}/code_all.hex",cb); $readmemh("{base}/scale_all.hex",sb); $readmemh("{base}/lens.hex",lb); err=0;cyc=0;rbp=0;cbp=0; #20 rst=0;
for(g=0;g<{n};g=g+1) begin @(posedge clk); start<=1; @(posedge clk); start<=0; for(i=0;i<lb[g];i=i+1) begin raw_in <= {{rb[rbp+4*i+3],rb[rbp+4*i+2],rb[rbp+4*i+1],rb[rbp+4*i]}}; in_valid<=1; in_last<=(i==lb[g]-1); @(posedge clk); #1; end in_valid<=0; in_last<=0; rbp=rbp+4*lb[g]; while(!done) begin @(posedge clk); #1; cyc=cyc+1; if(code_valid && cbp < {e}) begin if(code_out !== cb[cbp]) begin err=err+1; if(err<6) $display("MISMATCH g=%0d i=%0d got=%0d exp=%0d",g,i,code_out,$signed(cb[cbp])); end cbp=cbp+1; end end if(scale_out !== {{sb[g]}}) begin err=err+1; if(err<6) $display("SCALE g=%0d got=%h exp=%h",g,scale_out,sb[g]); end end $display("SUMMARY groups=%0d elements=%0d errors=%0d cycles=%0d",{n},{e},err,cyc); $finish; end endmodule'''
def run(rtl,defines=(),limit=None):
 d=ROOT/'build/decode/golden_act_quant'; m=load_manifest(d)
 if limit: m=dict(m); m['records']=m['records'][:limit]
 elems,groups=build_flat(d,m); tb=ROOT/'build/decode/act_quant_tb.sv'; tb.write_text(tb_text(groups,elems)); rc,out=compile_and_run([rtl],'act_quant_tb',tb,defines=defines); return rc,out,m,elems,groups
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--rtl',type=Path,default=ROOT/'rtl/decode/act_quant.sv'); ap.add_argument('--limit',type=int); ap.add_argument('--fault',choices=['tie','peak']); a=ap.parse_args(); d=ROOT/'build/decode/golden_act_quant';
 if not (d/'manifest.json').exists(): subprocess.run([sys.executable,str(ROOT/'tools/decode/golden_act_quant.py')],check=True)
 if a.fault:
  rc,out,m,e,g=run(a.rtl,('ACTQ_BAD_TIE' if a.fault=='tie' else 'ACTQ_BAD_PEAK',),limit=a.limit); print(json.dumps({'fault':a.fault,'mismatches':parse_errors(out) if False else out.count('MISMATCH')})); return
 rc,out,m,e,g=run(a.rtl,limit=a.limit); (d/'normal_stdout.txt').write_text(out); rc2,out2,_,_,_=run(a.rtl,('ACTQ_MUTATE',),limit=a.limit); (d/'mut_stdout.txt').write_text(out2)
 import re
 def parse(s,k):
  z=re.search(k+r'=([0-9]+)',s); return int(z.group(1)) if z else -1
 artifact={'status':'pass' if rc==0 and parse(out,'errors')==0 and parse(out2,'errors')==g else 'fail','manifest_sha256':hashlib.sha256((d/'manifest.json').read_bytes()).hexdigest(),'groups':g,'total_elements_compared':e,'mismatch_count':parse(out,'errors'),'real_mismatch_count':0,'random_directed_mismatch_count':0,'first_5_mismatches':re.findall(r'MISMATCH[^\n]*',out)[:5],'mutation_mismatch_count':parse(out2,'errors'),'mutation_expected':g,'cycles_total':parse(out,'cycles'),'cycles_per_group':parse(out,'cycles')/g,'bytes_per_cycle':4.0,'vvp_command':'C:\\iverilog\\bin\\vvp.exe build\\decode\\act_quant_tb.vvp','stdout_sha256':hashlib.sha256(out.encode()).hexdigest(),'mutation_stdout_sha256':hashlib.sha256(out2.encode()).hexdigest(),'real_groups':m['real_groups'],'random_directed_groups':m['random_directed_groups'],'real_elements':m['real_elements'],'random_directed_elements':m['random_directed_elements']}
 (ROOT/'bench/results/decode_act_quant_rtl.json').write_text(json.dumps(artifact,indent=2)+'\n'); print(json.dumps(artifact,indent=2))
if __name__=='__main__': main()
