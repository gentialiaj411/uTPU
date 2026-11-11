import json
from pathlib import Path
p=Path('sprint/data/wall_deadline.json');o=json.loads(p.read_text());o['unix']+=1800;p.write_text(json.dumps(o))
