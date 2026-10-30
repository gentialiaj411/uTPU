from pathlib import Path
# Prepared but not applied until all baseline runs finish, preserving honest input provenance.
p=Path('tools/decode/pack_decode_image.py');s=p.read_text()
s=s.replace('    mult, shift = np.asarray(rec[\'multiplier\'], dtype=np.int64), np.asarray(rec[\'shift\'], dtype=np.int64)',"    from firmware.host.numerics_v2.fixed import rshift_rne\n    mult, shift = rshift_rne(np.asarray(rec['multiplier'], dtype=np.int64), 16), np.asarray(rec['shift'], dtype=np.int64)")
s=s.replace('mult < (1 << 32)','mult < (1 << 16)').replace("m.to_bytes(4, 'little')", "m.to_bytes(2, 'little')")
s=s.replace('gcls * TILE_ROWS * 5','gcls * TILE_ROWS * 3')
s=s.replace('multiplier u32, shift u8','effective multiplier u16, shift u8')
p.write_text(s)
p=Path('rtl/decode/top_decode.sv');s=p.read_text().replace('CLS_G * 80','CLS_G * 48')
s=s.replace("assign sc_we = byte_ok && bc == 3'd4 && ((state == E_SCALES) || (state == G_SCALES));", "assign sc_we = byte_ok && (((state == E_SCALES) && bc == 3'd4) || ((state == G_SCALES) && bc == 3'd2));")
s=s.replace('m[sc_wg] <= btmp;',"m[sc_wg] <= (state == G_SCALES) ? {btmp[31:16], 16'b0} : btmp;")
needle="            G_SCALES: if (byte_ok) begin\n                if (bc == 3'd4)"
assert needle in s;s=s.replace(needle,needle.replace("3'd4","3'd2"));p.write_text(s)
# Board/top default address values must come from the freshly packed matching model.
import sys,json,re
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('tools/decode').resolve()))
from tools.decode.pack_decode_image import pack,load_export_15m
from tools.decode.harness import load_runtime_260k
for rtl,exp in [('rtl/decode/top_decode.sv',load_runtime_260k()[0]),('rtl/board/boolean_core.sv',load_export_15m())]:
 params=pack(exp)[1];p=Path(rtl);s=p.read_text()
 for key in ['EMB_SCALE_BASE','LAYERS_BASE','IMAGE_BYTES','CLS_TILE_BYTES']:
  s,count=re.subn(r'\b'+key+r' = \d+',key+' = '+str(params[key]),s);assert count==1,(rtl,key,count)
 p.write_text(s)
p=Path('tools/decode/fault_inject.py');s=p.read_text().replace("    'top': ('tools/decode/run_decode_rtl.py --prompts 1 --gen 8', 'rtl/decode/top_decode.sv', {", "    'top': ('tools/decode/run_decode_rtl.py --prompts 1 --gen 8', 'rtl/decode/top_decode.sv', {\n        'missing_effective_multiplier_expansion': ('rtl', \"{btmp[31:16], 16'b0}\", \"{16'b0, btmp[31:16]}\"),")
p.write_text(s)
