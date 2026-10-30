import json,re
from pathlib import Path
params=json.loads(Path('build/decode/decode_image_15m/params.json').read_text());text=Path('rtl/board/boolean_core.sv').read_text()
for key in ['EMB_SCALE_BASE','LAYERS_BASE','IMAGE_BYTES','CLS_TILE_BYTES']:
 value=int(re.search(r'\b'+key+r' = (\d+)',text)[1]);assert value==params[key],(key,value,params[key])
Path('sprint/data/final_image_default_check.json').write_text(json.dumps(dict(status='pass',image_bytes=params['IMAGE_BYTES'],flash_base=0x280000,end_exclusive=0x280000+params['IMAGE_BYTES'],bytes_free=(1<<24)-0x280000-params['IMAGE_BYTES']),indent=2))
print('SUMMARY board_defaults_match_packed_image=1 image_bytes='+str(params['IMAGE_BYTES']))
