from pathlib import Path
p=Path('sprint/scripts/report.py');s=p.read_text().replace("emit('## Summary');", "emit('P4 RTL remains EXPERIMENTAL pending complete fault and physical closure; measured simulation results are not board performance claims.\\n')\nemit('## Summary');");p.write_text(s)
