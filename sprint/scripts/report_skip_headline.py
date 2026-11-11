from pathlib import Path
p=Path('sprint/scripts/report.py');s=p.read_text();s=s.replace("f'P3 {go}; '+", "f'P3 {go}; MEASURED max mean row skip '+(f'{max(statistics.mean(b[\"skip\"]) for b in bounds.values()):.6f}' if bounds else 'N/A')+' over the final screening variants; '+")
p.write_text(s)
