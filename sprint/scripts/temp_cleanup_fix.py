from pathlib import Path
p=Path('tools/decode/sim.py');s=p.read_text();needle='def run_verilator(sources'
i=s.index(needle)
s=s[:i]+'''def _cleanup_verilator_build(directory):
    script = "import sys,shutil;from pathlib import Path;p=Path(sys.argv[1]);root=Path('/tmp/utpu-perf');assert p.parent.resolve()==root.resolve() and p.name.endswith('_obj');shutil.rmtree(p) if p.exists() else None"
    subprocess.run(['wsl','-d','Ubuntu','--','python3','-c',script,directory],capture_output=True,timeout=120,check=True)

'''+s[i:]
s=s.replace('if result.returncode: raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout, stderr=result.stderr)', 'if result.returncode:\n        _cleanup_verilator_build(obj_linux)\n        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout, stderr=result.stderr)')
s=s.replace('        return r.returncode,r.stdout+r.stderr', '        _cleanup_verilator_build(obj_linux)\n        return r.returncode,r.stdout+r.stderr')
s=s.replace('    except subprocess.TimeoutExpired as e:\n        return TIMEOUT_RC,_text(e.stdout)', '    except subprocess.TimeoutExpired as e:\n        _cleanup_verilator_build(obj_linux)\n        return TIMEOUT_RC,_text(e.stdout)')
p.write_text(s)
