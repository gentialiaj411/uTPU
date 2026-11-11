from pathlib import Path
p=Path('tools/decode/sim.py');s=p.read_text();s=s.replace('if result.returncode: raise RuntimeError("Verilator compile failed: "+(result.stdout+result.stderr)[-4000:])', 'if result.returncode: raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout, stderr=result.stderr)');p.write_text(s)
