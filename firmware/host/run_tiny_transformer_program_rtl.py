"""Full-top RTL replay of the committed complete tiny-transformer program."""
from __future__ import annotations

import json, os, re, subprocess, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; HOST=ROOT/'firmware'/'host'; BUILD=ROOT/'build'/'phase4_program_rtl'; OUT=ROOT/'bench'/'results'/'tiny_transformer_program_rtl.json'
sys.path.insert(0,str(HOST)); sys.path.insert(0,str(ROOT/'tools'))
from isa_simulator import UTPUISASimulator
from lower_tiny_transformer import CFG,M,_pack,build_program,build_program_stages,build_weight_image
from run_phase3_evidence import FULL_TOP_SOURCES,IVERILOG,VVP

def write(path:Path, values:list[int], width:int=4)->None:path.write_text('\n'.join(f'{int(v)&((1<<(width*4))-1):0{width}x}' for v in values)+'\n')

def main()->int:
 BUILD.mkdir(parents=True,exist_ok=True)
 ck=torch.load(ROOT/'bench/results/tiny_transformer_numerics.pt',map_location='cpu',weights_only=False);state,schema=ck['qstate'],ck['export_schema'];image=build_weight_image(state,schema);program=build_program(state,schema)
 vocab=json.loads((HOST/'fixtures/tiny_transformer_vocab.json').read_text())['characters'];data=np.asarray([vocab.index(c) for c in (ROOT/'build/tiny_transformer/input.txt').read_text()],np.int64);start=json.loads((HOST/'fixtures/tiny_transformer_split.json').read_text())['validation_start']
 pp=BUILD/'program.mem';write(pp,program); wf=BUILD/'weights.mem';write(wf,image); tb=BUILD/'full.sv'; stages=build_program_stages(state,schema); bounds=[];pc0=0
 for name,blob in stages: pc0+=len(blob); bounds.append(pc0)
 tb.write_text(f'''module tb;logic clk=0,rst=0,rx=1;wire tx,led_rst;integer i,c,miss,s,j,fd,a5_bytes,a5_timeout;reg[1023:0]ip,ep;reg[15:0]m[0:32767],p[0:{len(program)-1}],w[0:{len(image)-1}];reg[31:0]e[0:1151];integer b[0:{len(bounds)-1}];logic seen[0:{len(bounds)-1}];top #(.ARRAY_SIZE(8),.BUFFER_SIZE(32768),.COMPUTE_DATA_WIDTH(8),.ACCUMULATOR_DATA_WIDTH(32),.EXT_ADDR_EN(1),.PROG_DEPTH(32768),.FIFO_WIDTH(256),.MAX_BATCH_COUNT(4),.QUANTIZER_PIPE_DEPTH(1),.VECTOR_UNIT_EN(1))d(.clk(clk),.rst(rst),.rx(rx),.tx(tx),.led_rst(led_rst));always #5 clk=~clk;initial begin {''.join(f'b[{i}]={v};seen[{i}]=0;' for i,v in enumerate(bounds))} end
 // Byte-level FIFO injection with backpressure; this bypasses the UART PHY.
 // Phase 5 supplies the bit-level UART simulation.
 task push(input[7:0]v);begin @(negedge clk);while(d.fifo_in.full) @(negedge clk);d.fifo_in.mem[d.fifo_in.w_ptr[7:0]]=v;d.fifo_in.w_ptr=d.fifo_in.w_ptr+1;@(negedge clk);end endtask
 task dump_stage(input integer k);begin fd=$fopen($sformatf("{(BUILD/'stage_%0d.txt').as_posix()}",k),"w");for(j=0;j<32768;j=j+1)$fdisplay(fd,"%0d %04x",j,d.u_unified_buffer.bank_mem[j%32][j/32]);$fclose(fd);end endtask
 always @(posedge clk) if($test$plusargs("trace")) for(s=0;s<{len(bounds)};s=s+1) if(!seen[s]&&d.current_state==d.FETCH_BRAM_STATE&&d.pc==b[s]) begin seen[s]=1;dump_stage(s);$display("STAGE %0d PC %0d",s,d.pc);end
 initial begin if(!$value$plusargs("input=%s",ip)||!$value$plusargs("expect=%s",ep))$fatal(1,"args");$readmemh(ip,m);$readmemh("{pp.as_posix()}",p);$readmemh("{wf.as_posix()}",w);$readmemh(ep,e);repeat(8)@(posedge clk);rst=1;repeat(8)@(posedge clk);for(i=0;i<32768;i=i+1)d.u_unified_buffer.bank_mem[i%32][i/32]=m[i];if($test$plusargs("a5"))begin a5_bytes={len(image)}*2+5;a5_timeout=a5_bytes*64;for(i=0;i<{len(image)};i=i+1)m[i]=0;for(i=0;i<32768;i=i+1)d.u_unified_buffer.bank_mem[i%32][i/32]=m[i];push(8'ha5);push(0);push(0);push({len(image)});push({len(image)}>>8);for(i=0;i<{len(image)};i=i+1)begin push(w[i][7:0]);push(w[i][15:8]);end for(c=0;c<a5_timeout&&d.current_state!=d.WAIT_START_STATE;c=c+1)@(posedge clk);if(d.current_state!=d.WAIT_START_STATE)$fatal(1,"a5 fill timeout bytes=%0d cycles=%0d",a5_bytes,c);end push(8'ha1);push({len(program)});push({len(program)}>>8);for(i=0;i<{len(program)};i=i+1)begin push(p[i][7:0]);push(p[i][15:8]);end for(c=0;c<600000&&d.current_state!=d.WAIT_START_STATE;c=c+1)@(posedge clk);if(d.current_state!=d.WAIT_START_STATE)$fatal(1,"upload");push(8'ha2);for(c=0;c<20000000&&d.current_state!=d.HALT_STATE;c=c+1)@(posedge clk);if(d.current_state!=d.HALT_STATE)$fatal(1,"exec %0d %0d",d.current_state,d.pc);miss=0;for(i=0;i<1152;i=i+1)if(d.u_unified_buffer.bank_mem[e[i][15:0]%32][e[i][15:0]/32]!==e[i][31:16])miss=miss+1;$display("P4_RTL %0d %0d",miss,c);$finish;end endmodule''')
 # Backpressure can make the byte stream substantially slower than the raw
 # byte count; retain a byte-count-derived bound with generous protocol
 # headroom, rather than a fixed wall-clock constant.
 # The byte-level FIFO injector bypasses UART framing.  Its measured fill rate is
 # about 256 cycles/byte; 400 cycles/byte gives margin, while real 115200-baud
 # hardware is much slower (about 4,340 cycles/byte), so this is not a PHY limit.
 tb.write_text(tb.read_text().replace('a5_timeout=a5_bytes*64', 'a5_timeout=a5_bytes*400'))
 # The Icarus unified-buffer model has its own time-zero initializer.  Load
 # the direct-preload image only after it has settled and reset is released;
 # otherwise initialization order can silently replace the image with zeros.
 tb.write_text(tb.read_text().replace(
  ';for(i=0;i<32768;i=i+1)d.u_unified_buffer.bank_mem[i%32][i/32]=m[i];repeat(8)@(posedge clk);rst=1;repeat(8)@(posedge clk);',
  ';repeat(8)@(posedge clk);rst=1;repeat(8)@(posedge clk);for(i=0;i<32768;i=i+1)d.u_unified_buffer.bank_mem[i%32][i/32]=m[i];'))
 exe=BUILD/'full.vvp';subprocess.run([str(IVERILOG),'-g2012','-DICARUS','-s','tb','-o',str(exe),*FULL_TOP_SOURCES[:-1],str(tb)],cwd=ROOT,check=True,capture_output=True,text=True)
 mismatches=cycles=0
 count=int(os.environ.get('UTPU_SEQUENCE_COUNT','8'))
 for n in range(count):
  sim=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);sim.buffer[:len(image)]=image;sim.buffer[M.x:M.x+768]=_pack(state['token.weight'][data[start+n*32:start+(n+1)*32]]);initial=list(sim.buffer);sim.run_words(program,preserve_buffer=True)
  if os.environ.get('UTPU_STAGE_TRACE')=='1':
   prefix=0
   for si,(sn,blob) in enumerate(stages):
    prefix+=len(blob);ss=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);ss.buffer[:len(initial)]=initial;ss.run_words(program[:prefix]+program[-1:],preserve_buffer=True)
    (BUILD/f'isa_stage_{n}_{si}.txt').write_text(''.join(f'{a} {int(v)&0xffff:04x}\n' for a,v in enumerate(ss.buffer)))
  inp=BUILD/f'input_{n}.mem';exp=BUILD/f'expect_{n}.mem';write(inp,initial);write(exp,[(int(sim.buffer[M.logits_l+i])<<16)|(M.logits_l+i) for i in range(1152)],8)
  args=[str(VVP),str(exe),f'+input={inp.as_posix()}',f'+expect={exp.as_posix()}']
  if n == 0 and os.environ.get('UTPU_SKIP_A5') != '1': args.append('+a5')
  if os.environ.get('UTPU_STAGE_TRACE')=='1': args.append('+trace')
  try:
   out=subprocess.check_output(args,cwd=ROOT,text=True,stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as exc:
   raise RuntimeError(exc.output) from exc
  match=re.search(r'P4_RTL (\d+) (\d+)',out)
  if not match:raise RuntimeError(out)
  seq_miss=int(match.group(1));seq_cycles=int(match.group(2));print(f'P4_RTL_SEQ {n} {seq_miss} {seq_cycles}');mismatches+=seq_miss;cycles+=seq_cycles
 a5_used = count >= 1 and os.environ.get('UTPU_SKIP_A5') != '1'
 payload={'status':'passed' if not mismatches else 'failed','command':'python firmware/host/run_tiny_transformer_program_rtl.py','sequences':count,'bytes_compared':1152*count,'mismatches':mismatches,'stages':{'logits':{'bytes_compared':1152*count,'mismatches':mismatches,'cases':count}},'cycles_per_sequence':cycles/count,'milliseconds_at_50mhz':cycles/count/50000,'a5_weight_fill':a5_used,'a5_sequences':1 if a5_used else 0,'scope':'sequence 0 weights filled through UART MAGIC_BUF_FILL (A5) using a backpressured byte-level FIFO injector that bypasses the UART PHY; remaining sequences use direct preload; Phase 5 adds bit-level UART simulation' if a5_used else 'direct unified-buffer preload; A5 replay remains separately required'}
 OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,sort_keys=True));return int(bool(mismatches))
if __name__=='__main__':raise SystemExit(main())
