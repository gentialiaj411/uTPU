from pathlib import Path
p=Path('tools/decode/provenance.py');s=p.read_text().replace('import glob, hashlib, subprocess, sys, platform','import glob, hashlib, subprocess, sys, platform\nfrom functools import lru_cache')
s=s.replace('def iverilog_version() -> str:', '''def iverilog_version() -> str:
    return _iverilog_version_cached(sha256_file(IVERILOG) if IVERILOG.exists() else 'missing')

@lru_cache(maxsize=8)
def _iverilog_version_cached(executable_digest) -> str:''');p.write_text(s)
p=Path('tools/decode/sim.py');s=p.read_text().replace('import os, sys, re, shlex, hashlib, json','import os, sys, re, shlex, hashlib, json\nfrom functools import lru_cache')
s=s.replace('def simulator_identity():','''@lru_cache(maxsize=8)
def _version(executable, digest, linux):
    cmd = ['wsl','-d','Ubuntu','--',linux_path(executable),'--version'] if linux else [str(executable),'-V']
    r = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    if r.returncode: raise RuntimeError('simulator version query failed')
    return (r.stdout+r.stderr).strip()

def simulator_identity():''')
s=s.replace('version = subprocess.run([str(exe), "-V"], capture_output=True, text=True).stderr.strip()', 'version = _version(exe, hashlib.sha256(exe.read_bytes()).hexdigest(), False)')
s=s.replace('version = subprocess.run(["wsl", "-d", "Ubuntu", "--", linux_path(exe), "--version"], capture_output=True, text=True).stdout.strip()', 'version = _version(exe, hashlib.sha256(exe.read_bytes()).hexdigest(), True)')
p.write_text(s)
