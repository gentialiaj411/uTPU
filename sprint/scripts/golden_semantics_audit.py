from pathlib import Path
import subprocess,json,datetime
r=subprocess.run(['git','diff','e8d9282','--exit-code','--','firmware/host/numerics_v2/runq_runtime.py','firmware/host/numerics_v2/fixed.py'],capture_output=True,text=True)
assert r.returncode==0,'golden numeric semantics changed'
Path('sprint/data/golden_semantics_audit.json').write_text(json.dumps(dict(status='pass',command='git diff e8d9282 --exit-code -- firmware/host/numerics_v2/runq_runtime.py firmware/host/numerics_v2/fixed.py',timestamp=datetime.datetime.now().isoformat()),indent=2))
print('SUMMARY golden_runtime_and_fixed_source_unchanged=1')
