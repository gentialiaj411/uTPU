"""Real ISA/full-top smoke evidence for the Phase-4a VEC additions."""
from __future__ import annotations
import json, re, struct, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'firmware'/'host'))
from isa_encoder import IsaConfig, encodeHalt, encode_vec_vlayout, encode_vec_vmatmul
from isa_simulator import UTPUISASimulator
from run_phase3_evidence import FULL_TOP_SOURCES, IVERILOG, VVP
CFG=IsaConfig(address_width=16,compute_data_width=8,descriptor_isa=True,vector_unit=True)

def words(b): return list(struct.unpack('<'+'H'*(len(b)//2),b))
def put(mem,base,a):
 a=np.asarray(a).reshape(-1)
 for i in range(0,len(a),2): mem[base+i//2]=(int(a[i])&255)|((int(a[i+1])&255)<<8)

def full_top(name,program,mem,expect):
 build=ROOT/'build'/'phase4a'; build.mkdir(parents=True,exist_ok=True)
 ip,pp,ep=(build/f'{name}_{x}.mem' for x in ('input','prog','expect'))
 ip.write_text('\n'.join(f'{x:04x}' for x in mem)+'\n'); pp.write_text('\n'.join(f'{x:04x}' for x in program)+'\n')
 ep.write_text('\n'.join(f'{((v<<16)|a):08x}' for a,v in expect.items())+'\n')
 tb=build/f'{name}.sv'; tb.write_text(f'''module tb; logic clk=0,rst=0,rx=1; wire tx,led_rst; integer i,c,miss; bit seen;
reg [15:0] m[0:32767],p[0:{len(program)-1}]; reg [31:0] e[0:{len(expect)-1}];
top #(.ARRAY_SIZE(8),.BUFFER_SIZE(32768),.COMPUTE_DATA_WIDTH(8),.ACCUMULATOR_DATA_WIDTH(32),.EXT_ADDR_EN(1),.PROG_DEPTH(1024),.FIFO_WIDTH(256),.MAX_BATCH_COUNT(4),.QUANTIZER_PIPE_DEPTH(1),.VECTOR_UNIT_EN(1)) dut(.clk(clk),.rst(rst),.rx(rx),.tx(tx),.led_rst(led_rst)); always #5 clk=~clk;
task push(input [7:0]v); @(posedge clk);dut.fifo_in.mem[dut.fifo_in.w_ptr[7:0]]=v;dut.fifo_in.w_ptr=dut.fifo_in.w_ptr+1;@(posedge clk);endtask
initial begin $readmemh("{ip.as_posix()}",m);$readmemh("{pp.as_posix()}",p);$readmemh("{ep.as_posix()}",e);for(i=0;i<32768;i=i+1)dut.u_unified_buffer.bank_mem[i%32][i/32]=m[i];repeat(8)@(posedge clk);rst=1;repeat(8)@(posedge clk);push(8'ha1);push({len(program)});push(0);for(i=0;i<{len(program)};i=i+1)begin push(p[i][7:0]);push(p[i][15:8]);end for(c=0;c<10000&&dut.current_state!=dut.WAIT_START_STATE;c=c+1)@(posedge clk);push(8'ha2);for(c=0;c<200000&&dut.current_state!=dut.HALT_STATE;c=c+1)@(posedge clk);miss=0;for(i=0;i<{len(expect)};i=i+1)if(dut.u_unified_buffer.bank_mem[e[i][15:0]%32][e[i][15:0]/32]!==e[i][31:16])miss=miss+1;$display("P4A {name} %0d %0d",miss,c);$finish;end endmodule''')
 exe=build/f'{name}.vvp'; subprocess.run([str(IVERILOG),'-g2012','-DICARUS','-s','tb','-o',str(exe),*FULL_TOP_SOURCES[:-1],str(tb)],cwd=ROOT,check=True,capture_output=True,text=True)
 out=subprocess.check_output([str(VVP),str(exe)],cwd=ROOT,text=True); m=re.search(rf'P4A {name} (\d+) (\d+)',out)
 if not m or int(m.group(1)): raise RuntimeError(out)
 return int(m.group(2))

def main():
 mem=[0]*32768; q=np.array([[2,-1,3,1],[1,2,-2,0],[-1,3,1,2],[2,1,0,-2]],np.int8); k=np.array([[1,1,0,2],[2,-1,1,0],[0,3,-2,1],[-1,2,2,1]],np.int8)
 put(mem,100,q);put(mem,200,k);mem[300:308]=[5,200,0x8c63,0x404,4,1,0,0]
 pg=words(encode_vec_vmatmul(400,100,300,16,causal=True,cfg=CFG)+encodeHalt(cfg=CFG)); sim=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);sim.buffer[:]=mem;sim.run_words(pg,preserve_buffer=True); expect={400+i:sim.buffer[400+i] for i in range(8)}; c1=full_top('qk',pg,mem,expect)
 mem=[0]*32768; x=np.arange(-8,8,dtype=np.int8).reshape(4,4);put(mem,100,x);mem[300:308]=[6,0x63,0,0x404,0,0,0,0];mem[308:316]=[6,0x63,3,0x404,0,0,0,0]
 pg=words(encode_vec_vlayout(400,100,300,16,cfg=CFG)+encode_vec_vlayout(500,400,308,16,cfg=CFG)+encodeHalt(cfg=CFG));sim=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);sim.buffer[:]=mem;sim.run_words(pg,preserve_buffer=True);expect={500+i:sim.buffer[500+i] for i in range(8)}; c2=full_top('layout',pg,mem,expect)
 out={'status':'passed','command':'python tools/run_phase4a_evidence.py','matmul':{'cases':1,'mismatches':0,'cycles':c1},'layout':{'cases':1,'mismatches':0,'cycles':c2},'full_top':True}; p=ROOT/'bench'/'results'/'phase4a_vector_evidence.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));
if __name__=='__main__': main()
