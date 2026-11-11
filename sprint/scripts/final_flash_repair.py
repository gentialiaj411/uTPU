import os,sys,subprocess,time,json,shutil,datetime
from pathlib import Path
D=Path('sprint/data');state=json.loads((D/'engineering_status.json').read_text());(D/'engineering_pre_flashfix_status.json').write_text(json.dumps(state,indent=2))
deadline=json.loads((D/'wall_deadline.json').read_text())['unix'];os.environ['UTPU_SIM']='verilator'
def log(s):
 with Path('SPRINT_LOG.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' '+s+'\n')
def run(cmd,label,timeout=1800):
 p=D/(label+'.log')
 if p.exists():shutil.copyfile(p,D/(label+'.pre_idle_fix.log'))
 t=time.perf_counter();log('FINAL REPAIR '+subprocess.list2cmdline(cmd))
 with p.open('w') as f:
  try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=min(timeout,max(1,deadline-time.time()-120)));rc=r.returncode
  except subprocess.TimeoutExpired:rc=124
 state[label]=dict(command=cmd,rc=rc,seconds=time.perf_counter()-t);(D/'engineering_status.json').write_text(json.dumps(state,indent=2));log(label+' EXIT '+str(rc));return rc
parallel=[]
def launch(cmd,label):
 f=(D/(label+'.log')).open('w');p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);parallel.append((p,f,cmd,label,time.perf_counter()))
launch([sys.executable,'sprint/scripts/final_qspi_validation.py','verilator','bench/results/decode_qspi_rtl.json'],'final_qspi_verilator')
launch([sys.executable,'sprint/scripts/final_qspi_validation.py','icarus','sprint/data/qspi_icarus.json'],'final_qspi_icarus')
# Independent generated-input namespaces prevent compiler/memory-file collisions.
for backend in ['icarus','verilator']:
 launch([sys.executable,'sprint/scripts/postfix_decode_pair.py',backend],'postfix_decode_'+backend)
if run([sys.executable,'tools/board/run_uart_rtl.py','--sim','verilator'],'regen_uart',600)==0:shutil.copyfile('bench/results/boolean_uart_rtl.json',D/'uart_verilator.json')
run([sys.executable,'tools/board/run_uart_rtl.py','--sim','icarus','--output','sprint/data/uart_icarus.json'],'final_uart_icarus',1200)
for unit in ['gemv','top','board','qspi','actq','rms','vec','attn','attn15']:
 if time.time()>deadline-180:break
 if run([sys.executable,'tools/decode/fault_inject.py',unit],'fault_'+unit,1800):break
run([sys.executable,'tools/decode/run_decode_rtl.py','--model','260k','--kv-bits','8','--flash','qspi','--gen','8','--sim','verilator','--output','bench/results/decode_e2e_rtl_kv8_p1_qspi.json'],'regen_e2e_kv8_p1_qspi',300)
run([sys.executable,'tools/decode/run_decode_rtl.py','--model','15m','--kv-bits','4','--flash','qspi','--gen','1','--sim','verilator','--output','sprint/data/15m_metadata.json'],'final_15m_qspi',900)
for p,f,cmd,label,t in parallel:
 try:rc=p.wait(timeout=max(1,deadline-time.time()-120))
 except subprocess.TimeoutExpired:p.kill();rc=124
 f.close();state[label]=dict(command=cmd,rc=rc,seconds=time.perf_counter()-t)
 (D/'engineering_status.json').write_text(json.dumps(state,indent=2))
 if label=='final_qspi_verilator' and rc==0:shutil.copyfile('bench/results/decode_qspi_rtl.json',D/'qspi_verilator.json')
run([sys.executable,'sprint/scripts/compare.py'],'compare_final',60)
run([sys.executable,'tools/decode_gate.py'],'strict_gate_final',180)
state['P6']=dict(status='software validation complete; inspect final strict unmet for physical jobs')
(D/'engineering_status.json').write_text(json.dumps(state,indent=2))
(D/'final_flash_repair_complete.json').write_text(json.dumps(dict(timestamp=datetime.datetime.now().isoformat())))
