"""Executable Phase-4a VMATMUL/VLAYOUT ISA and serialized-RTL evidence."""
from __future__ import annotations
import json, re, struct, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; HOST=ROOT/'firmware'/'host'; BUILD=ROOT/'build'/'phase4a'
sys.path.insert(0,str(HOST))
from isa_encoder import IsaConfig, encodeHalt, encode_vec_vlayout, encode_vec_vmatmul
from isa_simulator import UTPUISASimulator
from run_phase3_evidence import FULL_TOP_SOURCES, IVERILOG, VVP
from tiny_transformer_reference import TinyTransformerConfig, integer_decoder_trace
CFG=IsaConfig(address_width=16,compute_data_width=8,descriptor_isa=True,vector_unit=True)

def words(x): return list(struct.unpack('<'+'H'*(len(x)//2),x))
def pack(x):
 x=np.asarray(x).reshape(-1).astype(np.uint8); return [int(x[i])|(int(x[i+1])<<8) for i in range(0,len(x),2)]
def put(m,b,x): m[b:b+len(pack(x))]=pack(x)
def rec(tr,au,bu,causal): return [5,200,0x0421|(int(tr)<<15),0x0202,int(au)|(int(bu)<<1)|(int(causal)<<2),1,0,0]
def run_sim(a,b,tr,au,bu,causal):
 m=[0]*32768;put(m,100,a);put(m,200,b);m[400:408]=rec(tr,au,bu,causal)
 s=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);s.buffer[:]=m
 s.run_words(words(encode_vec_vmatmul(300,100,400,4,causal=causal,cfg=CFG)+encodeHalt(cfg=CFG)),preserve_buffer=True)
 return np.asarray(s.buffer[300:302],np.uint16)
def write_bytes(p,x): p.write_text('\n'.join(f'{int(v)&255:02x}' for v in np.asarray(x).reshape(-1))+'\n',encoding='ascii')

def rtl_matmul(cases):
 """One compiled sequencer run: 2,048 ISA-generated random expectations."""
 BUILD.mkdir(parents=True,exist_ok=True);n=len(cases)
 ap,bp,ep,fp=[BUILD/f'mat_{x}.mem' for x in ('a','b','expected','flags')]
 write_bytes(ap,np.asarray([x[0].reshape(-1) for x in cases],np.int8));write_bytes(bp,np.asarray([x[1].reshape(-1) for x in cases],np.int8))
 write_bytes(ep,np.asarray([[int(x[6][0])&255,int(x[6][0])>>8,int(x[6][1])&255,int(x[6][1])>>8] for x in cases],np.uint8))
 write_bytes(fp,np.asarray([int(x[2])|(int(x[3])<<1)|(int(x[4])<<2)|(int(x[5])<<3) for x in cases],np.uint8))
 tb=f'''module tb;
reg clk=0,rst=1,start=0,causal;reg[3:0]subop=4;reg[15:0]dst=300,src0=100,param=400,length=4,buf_rdata;wire busy,done,error,we,re,se;wire[15:0]addr,wdata;reg[15:0]mem[0:32767];reg[7:0]a[0:{n*4-1}],b[0:{n*4-1}],e[0:{n*4-1}],f[0:{n-1}];integer i,k,cy,miss;
vector_sequencer #(.ADDRESS_SIZE(16),.BUFFER_WORDS(32768))d(.clk(clk),.rst(rst),.start(start),.subop(subop),.causal(causal),.dst(dst),.src0(src0),.param(param),.length(length),.buf_rdata(buf_rdata),.busy(busy),.done(done),.error(error),.buf_we(we),.buf_re(re),.buf_store_en(se),.buf_addr(addr),.buf_wdata(wdata));always #5 clk=~clk;always@(posedge clk)begin if(re)buf_rdata<=mem[addr];if(we)mem[addr]<=wdata;end
initial begin for(i=0;i<32768;i=i+1)mem[i]=0;buf_rdata=0;$readmemh("{ap.as_posix()}",a);$readmemh("{bp.as_posix()}",b);$readmemh("{ep.as_posix()}",e);$readmemh("{fp.as_posix()}",f);miss=0;cy=0;repeat(3)@(posedge clk);rst=0;for(k=0;k<{n};k=k+1)begin
mem[100]={{a[k*4+1],a[k*4]}};mem[101]={{a[k*4+3],a[k*4+2]}};mem[200]={{b[k*4+1],b[k*4]}};mem[201]={{b[k*4+3],b[k*4+2]}};mem[400]=5;mem[401]=200;mem[402]=16'h0421|(f[k][0]<<15);mem[403]=16'h0202;mem[404]={{13'd0,f[k][3],f[k][2],f[k][1]}};mem[405]=1;mem[406]=0;mem[407]=0;causal=f[k][3];@(negedge clk);start=1;@(negedge clk);start=0;while(!done&&cy<2000000)begin@(posedge clk);cy=cy+1;end;if(error||!done)miss=miss+1;if(mem[300]!=={{e[k*4+1],e[k*4]}})miss=miss+1;if(mem[301]!=={{e[k*4+3],e[k*4+2]}})miss=miss+1;end $display("P4A_MAT {n} %0d %0d",cy,miss);$finish;end endmodule'''
 sp=BUILD/'matmul_random.sv';sp.write_text(tb,encoding='utf-8');exe=BUILD/'matmul_random.vvp'
 for c in ([str(IVERILOG),'-g2012','-s','tb','-o',str(exe),'rtl/vector/vector_serial_alu.sv','rtl/vector/restoring_divider.sv','rtl/vector/vector_sequencer.sv',str(sp)],[str(VVP),str(exe)]):
  p=subprocess.run(c,cwd=ROOT,text=True,capture_output=True)
  if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 z=re.search(r'P4A_MAT (\d+) (\d+) (\d+)',p.stdout)
 if not z:raise RuntimeError(p.stdout)
 c,cy,mis=map(int,z.groups());return {'cases':c,'compared_bytes':c*4,'mismatches':mis,'cycles':cy,'passed':mis==0}

def layouts():
 rng=np.random.default_rng(4105);cases=mis=rounds=0;mode_cases={str(x):0 for x in range(8)}
 for r,c in ((1,2),(3,6),(5,10),(9,14),(31,30)):
  stride=c+2+(c&1);src=rng.integers(-128,128,(r,stride),dtype=np.int8);src[:,c:]=0
  for mode in range(6):
   m=[0]*32768;put(m,100,src);m[3000:3008]=[6,(r-1)|((c-1)<<5),mode,stride|(stride<<8),0,0,0,0];s=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);s.buffer[:]=m;s.run_words(words(encode_vec_vlayout(900,100,3000,r*c,cfg=CFG)+encodeHalt(cfg=CFG)),preserve_buffer=True);cases+=1
   mode_cases[str(mode)]+=1
  # Defined LIN inverse plus tails/stride holes has to be a true identity.
  m=[0]*32768;put(m,100,src);m[3000:3008]=[6,(r-1)|((c-1)<<5),0,stride|(stride<<8),0,0,0,0];m[3008:3016]=[6,(r-1)|((c-1)<<5),3,stride|(stride<<8),0,0,0,0];s=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);s.buffer[:]=m;s.run_words(words(encode_vec_vlayout(900,100,3000,r*c,cfg=CFG)+encode_vec_vlayout(1600,900,3008,r*c,cfg=CFG)+encodeHalt(cfg=CFG)),preserve_buffer=True);mis+=int(np.count_nonzero(np.asarray(s.buffer[1600:1600+(r*stride+1)//2])!=np.asarray(pack(src))));rounds+=1
 # Modes 6/7 are the only physical legacy-LOAD boundary maps.  These shapes
 # cover every byte lane, two tile groups, and the 32x48 transformer shape.
 for r,c in ((4,8),(8,16),(32,48)):
  src=rng.integers(-128,128,(r,c),dtype=np.int8);m=[0]*32768;put(m,100,src)
  m[12000:12008]=[6,(r-1)|((c-1)<<5),6,c|(c<<8),0,0,0,0];m[12008:12016]=[6,(r-1)|((c-1)<<5),7,c|(c<<8),0,0,0,0]
  s=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);s.buffer[:]=m;s.run_words(words(encode_vec_vlayout(2500,100,12000,r*c,cfg=CFG)+encode_vec_vlayout(5000,2500,12008,r*c,cfg=CFG)+encodeHalt(cfg=CFG)),preserve_buffer=True)
  got=np.frombuffer(np.asarray(s.buffer[5000:5000+r*c//2],dtype='<u2').tobytes(),dtype=np.int8).reshape(r,c);mis+=int(np.count_nonzero(got!=src));cases+=2;rounds+=1;mode_cases['6']+=1;mode_cases['7']+=1
 return {'cases':cases,'mode_cases':mode_cases,'round_trips':rounds,'mismatches':mis,'tails_and_padding':True,'physical_legacy_load_round_trips':3,'passed':mis==0}

def reference_stages():
 import torch
 ck=torch.load(ROOT/'bench/results/tiny_transformer_numerics.pt',map_location='cpu',weights_only=False);schema,state,cfg=ck['export_schema'],ck['qstate'],TinyTransformerConfig(**ck['config'])
 chars=json.loads((HOST/'fixtures/tiny_transformer_vocab.json').read_text())['characters'];data=np.asarray([chars.index(x) for x in (ROOT/'build/tiny_transformer/input.txt').read_text(encoding='utf-8')],np.int64);start=json.loads((HOST/'fixtures/tiny_transformer_split.json').read_text())['validation_start'];score=av=mis=0
 for q in range(512):
  t=integer_decoder_trace(data[start+q*32:start+(q+1)*32],state,schema,cfg)
  for h in range(3):
   for a,b,tr,au,want,kind in ((t['q'][0,h],t['k'][0,h],True,False,t['scores'][0,h],'scores'),(t['softmax_probs'][0,h],t['v'][0,h],False,True,t['av'][0,:,h*16:(h+1)*16],'av')):
    mm,kk,nn=(32,16,32) if tr else (32,32,16);m=[0]*32768;put(m,100,a);put(m,700,b);line=schema['linears'][kind];m[2000:2008]=[5,700,(mm-1)|((kk-1)<<5)|((nn-1)<<10)|(int(tr)<<15),kk|((kk if tr else nn)<<8),int(au)|4,int(line['multipliers'][0]),int(line['right_shifts'][0]),0];s=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);s.buffer[:]=m;s.run_words(words(encode_vec_vmatmul(1000,100,2000,mm*nn,causal=True,cfg=CFG)+encodeHalt(cfg=CFG)),preserve_buffer=True);raw=np.asarray(s.buffer[1000:1000+mm*nn//2],dtype='<u2').tobytes();got=np.frombuffer(raw,dtype=np.int8).reshape(mm,nn);exp=np.asarray(want,np.int8).copy();
    if tr:exp[np.triu_indices(32,1)]=0;score+=1
    else:av+=1
    mis+=int(np.count_nonzero(got!=exp))
 return {'sequences':512,'score_cases':score,'av_cases':av,'mismatches':mis,'passed':mis==0}

def full_top():
 # Full production top dispatch, separately from the sequencer-only corpus.
 m=[0]*32768;q=np.array([[2,-1],[127,-128]],np.int8);k=np.array([[1,1],[-128,127]],np.int8);put(m,100,q);put(m,200,k);m[400:408]=rec(True,False,False,True);pg=words(encode_vec_vmatmul(300,100,400,4,causal=True,cfg=CFG)+encodeHalt(cfg=CFG));s=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);s.buffer[:]=m;s.run_words(pg,preserve_buffer=True);expect={300+i:int(s.buffer[300+i]) for i in range(2)}
 ip,pp,ep=[BUILD/f'full_{x}.mem' for x in ('input','prog','expect')];ip.write_text('\n'.join(f'{x:04x}' for x in m)+'\n');pp.write_text('\n'.join(f'{x:04x}' for x in pg)+'\n');ep.write_text('\n'.join(f'{(v<<16)|a:08x}' for a,v in expect.items())+'\n')
 tb=BUILD/'full.sv';tb.write_text(f'''module tb;logic clk=0,rst=0,rx=1;wire tx,led_rst;integer i,c,miss;reg[15:0]m[0:32767],p[0:{len(pg)-1}];reg[31:0]e[0:{len(expect)-1}];top #(.ARRAY_SIZE(8),.BUFFER_SIZE(32768),.COMPUTE_DATA_WIDTH(8),.ACCUMULATOR_DATA_WIDTH(32),.EXT_ADDR_EN(1),.PROG_DEPTH(1024),.FIFO_WIDTH(256),.MAX_BATCH_COUNT(4),.QUANTIZER_PIPE_DEPTH(1),.VECTOR_UNIT_EN(1))d(.clk(clk),.rst(rst),.rx(rx),.tx(tx),.led_rst(led_rst));always #5 clk=~clk;task push(input[7:0]v);@(posedge clk);d.fifo_in.mem[d.fifo_in.w_ptr[7:0]]=v;d.fifo_in.w_ptr=d.fifo_in.w_ptr+1;@(posedge clk);endtask initial begin $readmemh("{ip.as_posix()}",m);$readmemh("{pp.as_posix()}",p);$readmemh("{ep.as_posix()}",e);for(i=0;i<32768;i=i+1)d.u_unified_buffer.bank_mem[i%32][i/32]=m[i];repeat(8)@(posedge clk);rst=1;repeat(8)@(posedge clk);push(8'ha1);push({len(pg)});push(0);for(i=0;i<{len(pg)};i=i+1)begin push(p[i][7:0]);push(p[i][15:8]);end for(c=0;c<10000&&d.current_state!=d.WAIT_START_STATE;c=c+1)@(posedge clk);push(8'ha2);for(c=0;c<200000&&d.current_state!=d.HALT_STATE;c=c+1)@(posedge clk);miss=0;for(i=0;i<{len(expect)};i=i+1)if(d.u_unified_buffer.bank_mem[e[i][15:0]%32][e[i][15:0]/32]!==e[i][31:16])miss=miss+1;$display("P4A_TOP %0d %0d",miss,c);$finish;end endmodule''')
 exe=BUILD/'full.vvp';subprocess.run([str(IVERILOG),'-g2012','-DICARUS','-s','tb','-o',str(exe),*FULL_TOP_SOURCES[:-1],str(tb)],cwd=ROOT,check=True,capture_output=True,text=True);o=subprocess.check_output([str(VVP),str(exe)],cwd=ROOT,text=True);z=re.search(r'P4A_TOP (\d+) (\d+)',o)
 if not z or int(z.group(1)):raise RuntimeError(o)
 return {'cases':1,'mismatches':0,'cycles':int(z.group(2)),'passed':True}

def decode_errors():
 """Execute malformed subop-4/5 descriptors; each must fail before writes."""
 out={}
 def check(name,subop,record,*,dst=300,length=4,causal=False):
  m=[0]*32768;put(m,100,np.array([1,2,3,4],np.int8));put(m,200,np.array([1,2,3,4],np.int8));m[400:408]=record
  blob=(encode_vec_vmatmul(dst,100,400,length,causal=causal,cfg=CFG) if subop==4 else encode_vec_vlayout(dst,100,400,length,cfg=CFG))+encodeHalt(cfg=CFG)
  try:
   s=UTPUISASimulator(array_size=8,buffer_size=32768,cfg=CFG);s.buffer[:]=m;s.run_words(words(blob),preserve_buffer=True);out[name]={'passed':False,'mismatches':1}
  except ValueError: out[name]={'passed':True,'mismatches':0}
 valid=rec(False,False,False,False)
 bad=valid.copy();bad[0]=6;check('matmul_bad_record_kind',4,bad)
 bad=valid.copy();bad[7]=1;check('matmul_reserved_record_bits',4,bad)
 bad=valid.copy();bad[3]=0x0101;check('matmul_invalid_stride',4,bad)
 bad=valid.copy();bad[4]=4;check('matmul_illegal_causal_shape',4,bad,causal=True)
 check('matmul_prohibited_overlap',4,valid,dst=100)
 layout=[6,0x0021,0,0x0202,0,0,0,0]
 bad=layout.copy();bad[0]=5;check('layout_bad_record_kind',5,bad)
 bad=layout.copy();bad[4]=1;check('layout_reserved_record_bits',5,bad)
 bad=layout.copy();bad[3]=0x0101;check('layout_invalid_stride',5,bad)
 bad=[6,0x0022,6,0x0202,0,0,0,0];check('layout_physical_shape',5,bad,length=6)
 check('layout_prohibited_overlap',5,layout,dst=100)
 return {'classes':out,'cases':len(out),'mismatches':sum(x['mismatches'] for x in out.values()),'passed':all(x['passed'] for x in out.values())}

def main():
 rng=np.random.default_rng(4104);families=[(tr,au,bu,False)for tr in(0,1)for au in(0,1)for bu in(0,1)]+[(1,au,bu,True)for au in(0,1)for bu in(0,1)]+[(0,1,bu,True)for bu in(0,1)];cases=[]
 for i in range(2048):
  tr,au,bu,ca=families[i%len(families)];a=rng.integers(0 if au else -128,256 if au else 128,(2,2),dtype=np.uint8 if au else np.int8);b=rng.integers(0 if bu else -128,256 if bu else 128,(2,2),dtype=np.uint8 if bu else np.int8)
  if i==0:a=np.array([[127,-128],[127,-128]],np.int8);b=np.array([[127,-128],[-128,127]],np.int8);tr,au,bu,ca=1,0,0,1
  cases.append((a.view(np.int8),b.view(np.int8),tr,au,bu,ca,run_sim(a,b,tr,au,bu,ca)))
 rtl=rtl_matmul(cases);lay=layouts();ref=reference_stages();top=full_top();dec=decode_errors();out={'status':'passed','command':'python tools/run_phase4a_evidence.py','seed':4104,'matmul':{'sim_cases':len(cases),'rtl':rtl,'families':len(families),'edge_cases':['max_magnitude','K=1','M=N=1','causal_row_0']},'layout':lay,'reference_score_av':ref,'full_top':top,'decode_errors':dec}
 if any((rtl['mismatches'],lay['mismatches'],ref['mismatches'],top['mismatches'],dec['mismatches'])):out['status']='failed'
 (ROOT/'bench/results/phase4a_vector_evidence.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True));return int(out['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
