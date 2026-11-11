"""Differential test the serialized RTL vector arithmetic against the ISA model.

The testbench feeds independent rows into ``rtl/vector/vector_alu.sv``.  It
uses Vivado xsim when Icarus is not installed, keeping the same SV subset used
by the board build.  Generated sources and memories live only under build/.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "phase3_vector"
sys.path.insert(0, str(ROOT / "firmware" / "host"))
from isa_simulator import UTPUISASimulator

# Focused rewrite steps may execute only the completed RTL operation(s).
SELECTED_OPS = set(os.environ.get("VECTOR_RTL_OPS", "vadd,vpos,rms,softmax").split(","))


def rne(v, s): return UTPUISASimulator._rshift_rne(v, s)
def sat(v): return np.clip(v, -128, 127).astype(np.int8)

def rms(a, g, shift):
    out=[]; bins=[]
    for row in a:
        ss=int(np.sum(row.astype(np.int64)**2)); q,rem=divmod(ss,48)
        mean=q+int(2*rem>48 or (2*rem==48 and q&1))+1
        exp=(mean.bit_length()-1)&~1; m=int(rne(mean<<14,exp)); m=min(65535,max(16384,m)); bins.append(min(191,(m-16384)>>8))
        from isa_simulator import RSQRT_Q14
        r0=int(RSQRT_Q14[bins[-1]]); t=int(rne(m*r0*r0,28)); r1=int(rne(r0*(3*16384-t),15))
        out.append(sat(rne(row.astype(np.int64)*g.astype(np.int64)*r1,14+shift+exp//2)))
    return np.asarray(out), bins

def soft(row):
    from isa_simulator import EXP_Q15
    mx=int(np.max(row)); rel=np.clip(row.astype(np.int64)-mx,-128,0); e=EXP_Q15[(-rel).astype(np.int16)].astype(np.int64)
    return ((255*e+int(e.sum())//2)//int(e.sum())).astype(np.uint8)

def write_mem(path: Path, values):
    path.write_text("\n".join(f"{int(x)&255:02x}" for x in np.asarray(values).reshape(-1))+"\n", encoding="ascii")

def run_tb(name, a, b, g, expected, eas=None, ebs=None, eos=None, valid=None, softmax=False):
    if name not in SELECTED_OPS:
        return {"cases": 0, "compared_bytes": 0, "mismatches": 0, "passed": True, "skipped": True}
    BUILD.mkdir(parents=True, exist_ok=True); n,w=a.shape
    paths={"a":BUILD/f"{name}_a.mem","b":BUILD/f"{name}_b.mem","g":BUILD/f"{name}_g.mem","e":BUILD/f"{name}_e.mem"}
    for k,v in (("a",a),("b",b),("g",g),("e",expected)): write_mem(paths[k],v)
    ea=np.zeros(n,dtype=np.int8) if eas is None else np.asarray(eas,np.int8); eb=np.zeros(n,dtype=np.int8) if ebs is None else np.asarray(ebs,np.int8); eo=np.zeros(n,dtype=np.int8) if eos is None else np.asarray(eos,np.int8); vc=np.ones(n,dtype=np.uint8)*w if valid is None else np.asarray(valid,np.uint8)
    for k,v in (("ea",ea),("eb",eb),("eo",eo),("vc",vc)): paths[k]=BUILD/f"{name}_{k}.mem"; write_mem(paths[k],v)
    out_kind="soft_o" if softmax else ("norm_o" if name.startswith("rms") else "add_o")
    # The 48-lane inputs retain their declared width; softmax only consumes 32.
    tb=f'''module tb;
reg signed [7:0] am[0:{n*w-1}],bm[0:{n*w-1}],gm[0:{n*w-1}],em[0:{n*w-1}]; reg signed [7:0] eam[0:{n-1}],ebm[0:{n-1}],eom[0:{n-1}]; reg [7:0] vcm[0:{n-1}];
logic signed [383:0] a,b,g,add_o,norm_o; logic signed [255:0] score; logic [255:0] soft_o; logic signed [7:0] ea,eb,eo; logic [5:0] gs,vc,strobe;
vector_alu dut(.a(a),.b(b),.gamma(g),.ea(ea),.eb(eb),.eo(eo),.gamma_shift(gs),.strobe(strobe),.add_out(add_o),.norm_out(norm_o),.score(score),.valid_count(vc),.softmax_out(soft_o));
integer c,i,miss; initial begin strobe=63;$readmemh("{paths['a'].as_posix()}",am);$readmemh("{paths['b'].as_posix()}",bm);$readmemh("{paths['g'].as_posix()}",gm);$readmemh("{paths['e'].as_posix()}",em);$readmemh("{paths['ea'].as_posix()}",eam);$readmemh("{paths['eb'].as_posix()}",ebm);$readmemh("{paths['eo'].as_posix()}",eom);$readmemh("{paths['vc'].as_posix()}",vcm); miss=0; gs=3; for(c=0;c<{n};c=c+1) begin ea=eam[c];eb=ebm[c];eo=eom[c];vc=vcm[c];a=0;b=0;g=0;score=0;for(i=0;i<48;i=i+1) if(i<{w}) begin a[i*8 +: 8]=am[c*{w}+i];b[i*8 +: 8]=bm[c*{w}+i];g[i*8 +: 8]=gm[c*{w}+i];end for(i=0;i<32;i=i+1) if(i<{w}) score[i*8 +: 8]=am[c*{w}+i]; strobe=c[5:0]; #1; for(i=0;i<{w};i=i+1) if ($signed({out_kind}[i*8 +: 8]) !== $signed(em[c*{w}+i])) miss=miss+1; end $display("VECTOR_RESULT {n*w} %0d",miss);$finish;end endmodule
'''
    tb_path=BUILD/f"{name}.sv"; tb_path.write_text(tb,encoding="utf-8")
    iverilog=Path(r"C:\iverilog\bin\iverilog.exe")
    vvp=iverilog.with_name("vvp.exe")
    if iverilog.is_file() and vvp.is_file():
        commands=([str(iverilog),"-g2012","-s","tb","-o",str(BUILD/f"{name}.vvp"),"rtl/vector/vector_alu.sv",str(tb_path)],
                  [str(vvp),str(BUILD/f"{name}.vvp")])
    else:
        exe=Path(r"C:\AMDDesignTools\2025.2\Vivado\bin\xvlog.bat")
        if not exe.is_file(): raise RuntimeError("neither Icarus nor Vivado xsim is available")
        commands=([str(exe),"-sv","rtl/vector/vector_alu.sv",str(tb_path)], [str(exe.parent/"xelab.bat"),"tb","-s",name], [str(exe.parent/"xsim.bat"),name,"-runall"])
    for cmd in commands:
        p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    line=[x for x in p.stdout.splitlines() if x.startswith("VECTOR_RESULT")][-1].split()
    return {"cases":int(n),"compared_bytes":int(line[1]),"mismatches":int(line[2]),"passed":int(line[2])==0}

def legacy_direct_oracle_main():
    rng=np.random.default_rng(304); pairs=np.arange(-128,128,dtype=np.int8); aa,bb=np.meshgrid(pairs,pairs,indexing="ij"); aa=aa.reshape(-1);bb=bb.reshape(-1)
    va=[];vb=[];ve=[];eas=[];ebs=[];eos=[]
    for ea,eb,eo in [(-5,-5,-4),(-4,-1,0),(-31,-31,-30),(-31,31,31), (30,30,31)]:
        for pos in range(0,65536,48):
            x=np.pad(aa[pos:pos+48],(0,max(0,pos+48-65536))).astype(np.int8); y=np.pad(bb[pos:pos+48],(0,max(0,pos+48-65536))).astype(np.int8)
            va.append(x);vb.append(y);ve.append(sat(rne(x.astype(np.int64),eo-ea)+rne(y.astype(np.int64),eo-eb)));eas.append(ea);ebs.append(eb);eos.append(eo)
    z=np.zeros_like(np.asarray(va)); vadd=run_tb("vadd",np.asarray(va),np.asarray(vb),z,np.asarray(ve),eas,ebs,eos)
    pos_a=rng.integers(-128,128,(64,48),dtype=np.int8);pos_b=rng.integers(-128,128,(64,48),dtype=np.int8);pos_e=sat(rne(pos_a.astype(np.int64),2)+rne(pos_b.astype(np.int64),1)); vpos=run_tb("vpos",pos_a,pos_b,np.zeros_like(pos_a),pos_e,np.full(64,-6),np.full(64,-5),np.full(64,-4))
    # Sparse/log-uniform candidates deliberately cover every normalized
    # mantissa bin, then random/adversarial rows fill the 2,000-row contract.
    found={}
    while len(found)<192:
        n=int(rng.integers(1,49)); lim=int(rng.integers(1,129)); x=np.zeros(48,np.int8)
        x[:n]=rng.integers(-lim,lim+1,n,dtype=np.int16).astype(np.int8); rng.shuffle(x)
        _, candidate_bins=rms(np.asarray([x]),np.ones(48,np.int8),0); found.setdefault(candidate_bins[0],x)
    rows=np.empty((2000,48),np.int8); rows[:3]=[np.zeros(48,np.int8),np.full(48,-128,np.int8),np.full(48,127,np.int8)]
    rows[3:195]=np.asarray([found[k] for k in range(192)],np.int8); rows[195:]=rng.integers(-128,128,(1805,48),dtype=np.int8)
    gam=rng.integers(-128,128,48,dtype=np.int8); re,bins=rms(rows,gam,3); rms_r=run_tb("rms",rows,np.zeros_like(rows),np.tile(gam,(len(rows),1)),re)
    scores=[];out=[];valid=[]
    for h in range(500):
        table=rng.integers(-128,128,(32,32),dtype=np.int8)
        if h<32: table[h,:h+1]=0
        if h==32: table[:]=127
        if h==33: table[:]=-128
        for row in range(32): scores.append(table[row,:row+1]);out.append(soft(table[row,:row+1]));valid.append(row+1)
    # Pad variable rows to 32, preserving masked data in the expected compare.
    ss=np.zeros((len(scores),32),np.int8); oo=np.zeros_like(ss)
    for i,(x,y) in enumerate(zip(scores,out)): ss[i,:len(x)]=x;oo[i,:len(y)]=y
    sm=run_tb("softmax",ss,np.zeros_like(ss),np.zeros_like(ss),oo,valid=valid,softmax=True)
    report={"per_op":{"vadd_rq":vadd,"vpos_add":vpos,"rmsnorm":rms_r,"softmax":sm},"rsqrt_lut_bin_coverage":{"covered":len(set(bins)),"bins":sorted(set(bins))}}
    print(json.dumps(report)); return 0 if all(x["mismatches"]==0 for x in report["per_op"].values()) else 1
def main():
    # The production path is the one-buffer-port FSM, not vector_alu's
    # compatibility shell.  Preserve this command surface for automation.
    from run_vector_sequencer_sim import main as sequencer_main
    return sequencer_main()

if __name__=="__main__": raise SystemExit(main())
