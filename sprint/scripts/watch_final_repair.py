from pathlib import Path
p=Path('sprint/scripts/finalize.py');s=p.read_text().replace("new['engineering']:'engineering_launcher'", "new['engineering']:'final_flash_repair'");p.write_text(s)
