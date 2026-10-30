from pathlib import Path
p=Path('sprint/scripts/implement_metadata.py');s=p.read_text();s=s.replace("s=s.replace('mult < (1 << 32)','mult < (1 << 16)')", "s=s.replace('mult < (1 << 32)','mult < (1 << 15)')")
p.write_text(s)
p=Path('sprint/scripts/report.py');s=p.read_text();s=s.replace('Packer asserts q fits u16.', 'Packer conservatively requires q<32768 although stored in u16, because the unchanged GEMV interprets the expanded 32-bit multiplier as signed. The measured model range fits this constraint; accepting arbitrary u16 would be unsound.')
p.write_text(s)
