import time,subprocess,sys,json
from pathlib import Path
while not Path('sprint/data/deterministic_recovery_complete.json').exists() and time.time()<json.loads(Path('sprint/data/wall_deadline.json').read_text())['unix']:time.sleep(15)
subprocess.run([sys.executable,'sprint/scripts/engineering_driver.py'])
