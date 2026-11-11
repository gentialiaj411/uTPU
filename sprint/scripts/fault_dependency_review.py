from pathlib import Path
p=Path('tools/decode/provenance.py');s=p.read_text().replace('FAULT_SOURCES = [*NUMERICS, *M260, *M15, *LUTS, ', "FAULT_SOURCES = [*NUMERICS, *M260, *M15, *LUTS, 'tools/decode/pack_decode_image.py', 'tools/decode/golden_*.py', ");p.write_text(s)
p=Path('tools/decode/sim.py');s=p.read_text().replace('Small iverilog/vvp compile-and-run helper for decode unit tests.','Shared Icarus/Verilator compile-and-run helper for decode tests.');p.write_text(s)
p=Path('tools/decode/test_gate.py');s=p.read_text().replace("[*pv.M260,*pv.M15,*pv.NUMERICS]", "[*pv.M260,*pv.M15,*pv.NUMERICS,'tools/decode/pack_decode_image.py','tools/decode/golden_*.py']");p.write_text(s)
