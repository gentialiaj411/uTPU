import json,time
from pathlib import Path
p=Path('sprint/data/wall_deadline.json')
if not p.exists():p.write_text(json.dumps(dict(unix=time.time()+2.25*3600)))
p=Path('sprint/scripts/engineering_driver.py');s=p.read_text().replace('deadline=time.monotonic()+3*3600',"deadline=time.monotonic()+max(1,json.loads((D/'wall_deadline.json').read_text())['unix']-time.time())")
s=s.replace("while not (D/'backend_recovery_complete.json').exists()", "while not (D/'deterministic_recovery_complete.json').exists()")
p.write_text(s)
p=Path('tools/decode/sim.py');s=p.read_text();needle='if SIMULATOR not in ("icarus", "verilator"):'
s=s.replace(needle,'''for i, arg in enumerate(sys.argv):
    if arg.startswith('--sim='):
        SIMULATOR = arg.split('=',1)[1]
        del sys.argv[i]
        break
'''+needle);p.write_text(s)
p=Path('sprint/scripts/finalize.py');s=p.read_text();start=s.index('jobs={');end=s.index('\n',start);s=s[:start]+"jobs={33636:'argmax_bound',new['engineering']:'engineering_launcher'}"+s[end:];s=s.replace('time.monotonic()-start>2.5*3600',"time.time()>json.loads(Path('sprint/data/wall_deadline.json').read_text())['unix']+300");p.write_text(s)
