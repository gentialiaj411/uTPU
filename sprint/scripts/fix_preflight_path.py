from pathlib import Path
p=Path('sprint/scripts/metadata_preflight.py');s=p.read_text().replace("source=captured['tools/decode/pack_decode_image.py']", "assert len(captured)==1\nsource=next(iter(captured.values()))");p.write_text(s)
