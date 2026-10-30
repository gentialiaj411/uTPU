from pathlib import Path
p=Path('tools/decode/sim.py');s=p.read_text().replace('    BUILD.mkdir(parents=True, exist_ok=True)','    BUILD.mkdir(parents=True, exist_ok=True)\n    testbench = Path(tb)\n    text = testbench.read_text()\n    if "`timescale" not in text and "timeunit" not in text:\n        testbench.write_text("`timescale 1ns/1ps\\n" + text)');p.write_text(s)
