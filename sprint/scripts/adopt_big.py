import ctypes,time,json
from pathlib import Path
K=ctypes.windll.kernel32;K.OpenProcess.restype=ctypes.c_void_p
h=K.OpenProcess(0x100000,False,38420);deadline=time.monotonic()+45*60
while h and K.WaitForSingleObject(ctypes.c_void_p(h),0)==258 and time.monotonic()<deadline:time.sleep(15)
rows=json.loads(Path('sprint/data/15m_timing.json').read_text())
p=Path('sprint/data/15m_big.json')
if p.exists():
 o=json.loads(p.read_text());rows.append(dict(label='15m_big',returncode=0 if o.get('status')=='pass' else 1,wall_seconds=o['wall_seconds'],command=['python','tools/decode/run_decode_rtl.py','--model','15m','--kv-bits','4','--gen','32','--prompts','8','--sim','verilator','--output','sprint/data/15m_big.json']))
else:rows.append(dict(label='15m_big',returncode=124,wall_seconds=None,note='adopted job did not complete within extended wall cap'))
Path('sprint/data/15m_timing.json').write_text(json.dumps(rows,indent=2))
