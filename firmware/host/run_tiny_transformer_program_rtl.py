"""Full-top RTL replay of the committed complete tiny-transformer program."""
from __future__ import annotations

import json, re, subprocess, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; HOST=ROOT/'firmware'/'host'; BUILD=ROOT/'build'/'phase4_program_rtl'; OUT=ROOT/'bench'/'results'/'tiny_transformer_program_rtl.json'
sys.path.insert(0,str(HOST)); sys.path.insert(0,str(ROOT/'tools'))
from isa_simulator import UTPUISASimulator
from lower_tiny_transformer import CFG,M,_pack,build_program,build_weight_image
from run_phase3_evidence import FULL_TOP_SOURCES,IVERILOG,VVP

def write(path:Path, values:list[int], width:int=4)->None:path.write_text('\n'.join(f'{int(v)&((1<<(width*4))-1):0{width}x}' for v in values)+'\n')

def main()->int:
 BUILD.mkdir(parents=True,exist_ok=True)
 ck=torch.load(ROOT/'bench/results/tiny_transformer_numerics.pt',map_location='cpu',weights_only=False);state,schema=ck['qstate'],ck['export_schema'];image=build_weight_image(state,schema);program=build_program(state,schema)
 vocab=json.loads((HOST/'fixtures/tiny_transformer_vocab.json').read_text())['characters'];data=np.asarray([vocab.index(c) for c in (ROOT/'build/tiny_transformer/input.txt').read_text()],np.int64);start=json.loads((HOST/'fixtures/tiny_transformer_split.json').read_text())['validation_start']
 pp=BUILD/'program.mem';write(pp,program); tb=BUILD/'full.sv'
 tb.write_text(f'''module tb;logic clk=0,rst=0,rx=1;wire tx,led_rst;integer i,c,miss;reg[1023:0]ip,ep;reg[15:0]m[0:32767],p[0:{len(program)-1}];reg[31:0]e[0:1151];top #(.ARRAY_SIZE(8),.BUFFER_SIZE(32768),.COMPUTE_DATA_WIDTH(8),.ACCUMULATOR_DATA_WIDTH(32),.EXT_ADDR_EN(1),.PROG_DEPTH(32768),.FIFO_WIDTH(256),.MAX_BATCH_COUNT(4),.QUANTIZER_PIPE_DEPTH(1),.VECTOR_UNIT_EN(1))d(.clk(clk),.rst(rst),.rx(rx),.tx(tx),.led_rst(led_rst));always #5 clk=~clk;task push(input[7:0]v);@(posedge clk);d.fifo_in.mem[d.fifo_in.w_ptr[7:0]]=v;d.fifo_in.w_ptr=d.fifo_in.w_ptr+1;repeat(8)@(posedge clk);endtask initial begin if(!$value$plusargs("input=%s",ip)||!$value$plusargs("expect=%s",ep))$fatal(1,"args");$readmemh(ip,m);$readmemh("{pp.as_posix()}",p);$readmemh(ep,e);for(i=0;i<32768;i=i+1)d.u_unified_buffer.bank_mem[i%32][i/32]=m[i];repeat(8)@(posedge clk);rst=1;repeat(8)@(posedge clk);push(8'ha1);push({len(program)});push({len(program)}>>8);for(i=0;i<{len(program)};i=i+1)begin push(p[i][7:0]);push(p[i][15:8]);end for(c=0;c<600000&&d.current_state!=d.WAIT_START_STATE;c=c+1)@(posedge clk);if(d.current_state!=d.WAIT_START_STATE)$fatal(1,"upload");push(8'ha2);for(c=0;c<20000000&&d.current_state!=d.HALT_STATE;c=c+1)@(posedge clk);if(d.current_state!=d.HALT_STATE)$fatal(1,"exec %0d %0d",d.current_state,d.pc);miss=0;for(i=0;i<1152;i=i+1)if(d.u_unified_buffer.bank_mem[e[i][15:0]%32][e[i][15:0]/32]!==e[i][31:16])miss=miss+1;$display("P4_RTL %0d %0d",miss,c);$finish;end endmodule''')
 # The Icarus unified-buffer model has its own time-zero initializer.  Load
 # the direct-preload image only after it has settled and reset is released;
 # otherwise initialization order can silently replace the image with zeros.
 tb.write_text(tb.read_text().replace(
  ';for(i=0;i<32768;i=i+1)d.u_unified_buffer.bank_mem[i%32][i/32]=m[i];repeat(8)@(posedge clk);rst=1;repeat(8)@(posedge clk);',
  ';repeat(8)@(posedge clk);rst=1;repeat(8)@(posedge clk);for(i=0;i<32768;i=i+1)d.u_unified_buffer.bank_mem[i%32][i/32]=m[i];'))
 exe=BUILD/'full.vvp';subprocess.run([str(IVERILOG),'-g2012','-DICARUS','-s','tb','-o',str(exe),*FULL_TOP_SOURCES[:-1],str(tb)],cwd=ROOT,check=True,capture_output=True,text=True)
 mismatches=cycles=0
 for n in range(8):
  sim=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);sim.buffer[:len(image)]=image;sim.buffer[M.x:M.x+768]=_pack(state['token.weight'][data[start+n*32:start+(n+1)*32]]);initial=list(sim.buffer);sim.run_words(program,preserve_buffer=True)
  inp=BUILD/f'input_{n}.mem';exp=BUILD/f'expect_{n}.mem';write(inp,initial);write(exp,[(int(sim.buffer[M.logits_l+i])<<16)|(M.logits_l+i) for i in range(1152)],8)
  out=subprocess.check_output([str(VVP),str(exe),f'+input={inp.as_posix()}',f'+expect={exp.as_posix()}'],cwd=ROOT,text=True);match=re.search(r'P4_RTL (\d+) (\d+)',out)
  if not match:raise RuntimeError(out)
  mismatches+=int(match.group(1));cycles+=int(match.group(2))
 payload={'status':'passed' if not mismatches else 'failed','command':'python firmware/host/run_tiny_transformer_program_rtl.py','sequences':8,'mismatches':mismatches,'stages':{'logits':{'mismatches':mismatches,'cases':8}},'cycles_per_sequence':cycles/8,'milliseconds_at_50mhz':cycles/8/50000,'a5_weight_fill':False,'scope':'direct unified-buffer preload; A5 replay remains separately required'}
 OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,sort_keys=True));return int(bool(mismatches))
if __name__=='__main__':raise SystemExit(main())
