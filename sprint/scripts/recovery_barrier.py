from pathlib import Path
p=Path('sprint/scripts/engineering_driver.py');s=p.read_text().replace('deadline=time.monotonic()+4*3600','deadline=time.monotonic()+3*3600')
s=s.replace("run([sys.executable,'sprint/scripts/compare.py'],'compare_final',60)","while not (D/'backend_recovery_complete.json').exists() and time.monotonic()<deadline:time.sleep(30)\nrun([sys.executable,'sprint/scripts/compare.py'],'compare_final',60)")
p.write_text(s)
