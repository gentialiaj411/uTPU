import subprocess,sys,time,json,datetime,csv,os
from pathlib import Path
D=Path('sprint/data');deadline=time.monotonic()+max(1,json.loads((D/'wall_deadline.json').read_text())['unix']-time.time())
state={}
def log(msg):
 with Path('SPRINT_LOG.md').open('a') as f:f.write('\n- '+datetime.datetime.now().isoformat()+' '+msg+'\n')
def run(cmd,label,timeout=2700):
 log('RUN '+subprocess.list2cmdline(cmd)+' -> sprint/data/'+label+'.log');t=time.perf_counter()
 with (D/(label+'.log')).open('w') as f:
  try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=min(timeout,max(1,deadline-time.monotonic())));rc=r.returncode
  except subprocess.TimeoutExpired:rc=124
 state[label]=dict(command=cmd,rc=rc,seconds=time.perf_counter()-t);(D/'engineering_status.json').write_text(json.dumps(state,indent=2));log(label+' EXIT '+str(rc));return rc
# Preserve baseline provenance: no metadata/source-format changes until baseline jobs terminate.
while not (D/'crosscheck.csv').exists() and time.monotonic()<deadline:time.sleep(30)
while not (D/'15m_timing.json').exists() and time.monotonic()<deadline:time.sleep(30)
while time.monotonic()<deadline:
 try:
  timing=json.loads((D/'15m_timing.json').read_text())
  if len(timing)>=2 or any(x['returncode'] for x in timing):break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(30)
while not (D/'deterministic_recovery_complete.json').exists() and time.monotonic()<deadline:time.sleep(30)
run([sys.executable,'sprint/scripts/compare.py'],'compare_final',60)
rows=list(csv.DictReader((D/'crosscheck_comparison.csv').open()))
passed=len(rows)==11 and all(x['identical']=='True' and x['icarus']==x['verilator']=='pass' for x in rows)
if not passed:
 state['P4']=dict(status='BLOCKED',reason='P1 full backend equivalence acceptance unmet; no metadata RTL applied')
 log('P4 BLOCKED by P1 full backend equivalence; arithmetic and packed format preserved.')
else:
 backups={str(p):p.read_bytes() for p in map(Path,['tools/decode/pack_decode_image.py','rtl/decode/top_decode.sv','rtl/board/boolean_core.sv','tools/decode/fault_inject.py'])}
 (D/'metadata_baseline.json').write_text(json.dumps({str(p):json.loads(p.read_text()) for p in [D/'e2e_qspi_verilator.json',D/'15m_small.json']}))
 rc=run([sys.executable,'tools/decode/run_decode_rtl.py','--model','15m','--kv-bits','4','--flash','qspi','--gen','1','--sim','verilator','--output','sprint/data/15m_qspi_before_metadata.json'],'15m_qspi_before_metadata',1200)
 if not rc:rc=run([sys.executable,'sprint/scripts/implement_metadata.py'],'metadata_apply',180)
 if not rc:
  rc=run([sys.executable,'-m','pytest','sprint/scripts/test_compact_metadata.py','-q'],'metadata_python_exactness',180)
 if not rc:
  os.environ['UTPU_SIM']='verilator'
  checks=[('260k_metadata',['--model','260k','--kv-bits','4','--flash','qspi','--gen','8']),('15m_metadata',['--model','15m','--kv-bits','4','--flash','qspi','--gen','1'])]
  for name,args in checks:
   rc=run([sys.executable,'tools/decode/run_decode_rtl.py',*args,'--sim','verilator','--output',f'sprint/data/{name}.json'],name,2700)
   if rc:break
 if rc:
  for name,content in backups.items():Path(name).write_bytes(content)
  state['P4']=dict(status='TIMEBOXED',reason='metadata smoke failed; original bytes restored')
 else:state['P4']=dict(status='IN-PROGRESS',reason='smokes pass; regeneration/fault acceptance required')
# Regenerate fast artifact producers, never edit result JSONs by hand.
backend='verilator' if passed else 'icarus';os.environ['UTPU_SIM']=backend
jobs=[('gemv',['tools/decode/run_gemv_rtl.py']),('gemv15',['tools/decode/run_gemv_rtl.py','--golden','build/decode/golden_gemv15','--output','bench/results/decode_gemv15_rtl.json']),('actq',['tools/decode/run_act_quant_rtl.py']),('rms',['tools/decode/run_rms_rtl.py']),('vec',['tools/decode/run_vec_rtl.py']),('attn',['tools/decode/run_attn_rtl.py']),('attn15',['tools/decode/run_attn15_rtl.py']),('qspi',['tools/decode/run_qspi_rtl.py']),('uart',['tools/board/run_uart_rtl.py'])]
for name,cmd in jobs:
 if time.monotonic()>deadline-2700:break
 if name=='qspi' and passed and run([sys.executable,'sprint/scripts/promote_fresh_qspi.py'],'regen_qspi',180)==0:continue
 if run([sys.executable,*cmd,'--sim',backend],'regen_'+name,1800):break
# Fault suites depend on every model/numeric source: produce them under final inputs.
for unit in ['gemv','top','board','qspi','actq','rms','vec','attn','attn15']:
 if time.monotonic()>deadline-2700:break
 if run([sys.executable,'tools/decode/fault_inject.py',unit],'fault_'+unit,1800):break
# Regenerate final e2e artifacts through their producers when the accelerated backend passed equivalence.
if passed:
 e2es=[('kv8_p1',['--model','260k','--kv-bits','8']),('kv4_p1',['--model','260k','--kv-bits','4']),('kv8_p2',['--model','260k','--kv-bits','8','--period','2']),('kv8_p4',['--model','260k','--kv-bits','8','--period','4']),('kv8_p1_qspi',['--model','260k','--kv-bits','8','--flash','qspi']),('15m_kv4_p1',['--model','15m','--kv-bits','4'])]
 for name,args in e2es:
  if time.monotonic()>deadline-1800:break
  if run([sys.executable,'tools/decode/run_decode_rtl.py',*args,'--gen','8','--sim',backend,'--output','bench/results/decode_e2e_rtl_'+name+'.json'],'regen_e2e_'+name,1800):break
# Use the reserved final regeneration window for any still-unrun fault suite.
for unit in ['gemv','top','board','qspi','actq','rms','vec','attn','attn15']:
 if 'fault_'+unit in state:continue
 if time.monotonic()>deadline-300:break
 if run([sys.executable,'tools/decode/fault_inject.py',unit],'fault_'+unit,1800):break
# Keep strict gate strict; stale physical artifacts are expected until OWNER_RUN2.
run([sys.executable,'tools/decode_gate.py'],'strict_gate_final',180)
state['P6']=dict(status='PARTIAL',note='Physical jobs NOT RUN; inspect strict gate unmet. No gate relaxed.')
(D/'engineering_status.json').write_text(json.dumps(state,indent=2))
log('Engineering driver finished; inspect final gate and package acceptance before reporting.')
