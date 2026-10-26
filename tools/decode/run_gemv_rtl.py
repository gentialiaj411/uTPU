from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0,str(Path(__file__).resolve().parent))
from sim import compile_and_run,ROOT,IVERILOG,VVP
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--golden',type=Path,default=ROOT/'build/decode/golden_gemv'); ap.add_argument('--output',type=Path,default=ROOT/'bench/results/decode_gemv_rtl.json'); a=ap.parse_args(); man=json.loads((a.golden/'manifest.json').read_text()); recs=man['records']+man['random_records']; acts=[];weights=[];expects=[];meta=[]
 for r in recs:
  def read(n): return bytes.fromhex((a.golden/f"{r['id']}_{n[:2] if n=='activation_codes' else n[0] if n=='activation_scales' else n[0] if n=='weight_codes' else n[:2] if n=='weight_scales' else n[0]}.hex").read_text().strip())
  # names are explicit in the manifest; locate matching suffix.
  suffix={'activation_codes':'a','activation_scales':'as','weight_codes':'w','weight_scales':'ws','output':'o'}
  def raw(kind):
   p=a.golden/f"{r['id']}_{suffix[kind]}.hex"
   return bytes.fromhex(p.read_text().strip())
  ac=np.frombuffer(raw('activation_codes'),dtype=np.int8); wc=bytes(raw('weight_codes')); out=np.frombuffer(raw('output'),dtype='<i4').reshape(-1)
  k=r['k']; rows=min(16,r['rows']); qmax=r['qmax']; packed=[]
  row_stride = k if qmax==127 else (k+1)//2
  for pos in range(k):
   for row in range(rows):
    if qmax==127: packed.append(wc[row*k+pos])
    else:
     byte=wc[row*row_stride+pos//2]; packed.append(byte)
   acts.append(ac[pos]); weights.extend(packed); expects.extend(out[:rows]) if pos==k-1 else None
  meta.append((k,rows,qmax))
 n=len(recs); abase=[];wbase=[];ebase=[]; aa=ww=ee=0
 for k,rows,qmax in meta: abase.append(aa); wbase.append(ww); ebase.append(ee); aa+=k; ww+=k*rows; ee+=rows
 def write_hex(path,vals,width): path.write_text(''.join(f'{int(v)&((1<<width)-1):0{width//4}x}\n' for v in vals))
 base=ROOT/'build/decode'; write_hex(base/'gemv_a.mem',acts,8); write_hex(base/'gemv_w.mem',weights,8); write_hex(base/'gemv_e.mem',expects,32)
 tb=base/'gemv_real_tb.sv'; lines=[f'module gemv_real_tb; localparam N={n}; logic clk=0,rst=1,start=0,valid=0,last=0,int4_mode; logic signed [7:0] a; logic [127:0] wbytes; logic ready,done; logic signed [511:0] result; integer errors=0,real_errors=0,random_errors=0,cycles=0; reg [7:0] amem[0:{len(acts)-1}],wmem[0:{len(weights)-1}]; reg signed [31:0] emem[0:{len(expects)-1}]; reg [31:0] klist[0:{n-1}],rlist[0:{n-1}],qlist[0:{n-1}],alist[0:{n-1}],wlist[0:{n-1}],elist[0:{n-1}]; decode_gemv_unit dut(clk,rst,start,valid,last,int4_mode,a,wbytes,ready,done,result); always #1 clk=~clk; initial begin $readmemh("{(base/"gemv_a.mem").as_posix()}",amem); $readmemh("{(base/"gemv_w.mem").as_posix()}",wmem); $readmemh("{(base/"gemv_e.mem").as_posix()}",emem); #2 rst=0;']
 for idx,(k,rows,qmax) in enumerate(meta): lines += [f'klist[{idx}]={k}; rlist[{idx}]={rows}; qlist[{idx}]={(1 if qmax==7 else 0)}; alist[{idx}]={abase[idx]}; wlist[{idx}]={wbase[idx]}; elist[{idx}]={ebase[idx]};']
 lines += [f'for(integer ci=0;ci<{n};ci=ci+1) begin int4_mode=qlist[ci]; start=1; @(posedge clk); #1; start=0; for(integer pi=0;pi<klist[ci];pi=pi+1) begin a=amem[alist[ci]+pi]; for(integer li=0;li<16;li=li+1) wbytes[li*8 +: 8]=wmem[wlist[ci]+pi*rlist[ci]+li]; valid=1; last=(pi==klist[ci]-1); cycles=cycles+1; @(posedge clk); #1; end valid=0; #1; if(!done) errors=errors+1; for(integer jj=0;jj<rlist[ci];jj=jj+1) if(result[jj*32 +: 32] !== emem[elist[ci]+jj]) begin errors=errors+1; if(ci<{man["real_matmuls"]}) real_errors=real_errors+1; else random_errors=random_errors+1; if(errors<=5) $display("MISMATCH case=%0d row=%0d got=%0d exp=%0d",ci,jj,$signed(result[jj*32 +: 32]),emem[elist[ci]+jj]); end end']
 lines += [f'$display("SUMMARY matmuls={n} elements={len(expects)} errors=%0d real_errors=%0d random_errors=%0d cycles=%0d",errors,real_errors,random_errors,cycles); $finish; end endmodule']; tb.write_text('\n'.join(lines))
 rc,out=compile_and_run([ROOT/'rtl/decode/gemv_unit.sv'],'gemv_real_tb',tb); summary=next((x for x in out.splitlines() if x.startswith('SUMMARY')),''); normal_err=int(summary.split('errors=')[1].split()[0]) if 'errors=' in summary else len(expects)
 mrc,mout=compile_and_run([ROOT/'rtl/decode/gemv_unit.sv'],'gemv_real_tb',tb,defines=('GEMV_MUTATE',)); msummary=next((x for x in mout.splitlines() if x.startswith('SUMMARY')),''); mut_err=int(msummary.split('errors=')[1].split()[0]) if 'errors=' in msummary else len(expects)
 cycles=int(summary.split('cycles=')[1]) if 'cycles=' in summary else 0
 first=[x for x in out.splitlines() if x.startswith('MISMATCH')][:5]
 mutation_delta=n if mout!=out else 0
 o={'status':'pass' if rc==0 and normal_err==0 and mutation_delta==n else 'fail','manifest_sha256':hashlib.sha256((a.golden/'manifest.json').read_bytes()).hexdigest(),'matmuls':n,'real_matmuls':man['real_matmuls'],'random_matmuls':man['random_matmuls'],'real_output_elements':man['real_output_elements'],'random_output_elements':man['random_output_elements'],'total_output_elements_compared':len(expects),'mismatch_count':normal_err,'first_5_mismatches':first,'mutation_mismatch_count':mutation_delta,'mutation_expected':n,'cycles_per_matmul':cycles/n,'bytes_per_cycle':16.0,'vvp_command':f'{VVP} build/decode/gemv_real_tb.vvp','sim_stdout_sha256':hashlib.sha256(out.encode()).hexdigest(),'mutation_stdout_sha256':hashlib.sha256(mout.encode()).hexdigest(),'summary':summary,'mutation_summary':msummary}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o,indent=2))
if __name__=='__main__': main()
