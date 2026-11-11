import sys,json,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path('tools/decode').resolve()))
import provenance as pv
source=Path('sprint/data/qspi_verilator.json');dest=Path('bench/results/decode_qspi_rtl.json');obj=json.loads(source.read_text())
errors=pv.stale(obj['provenance'],'decode_qspi_rtl')+pv.missing_coverage(obj['provenance'],pv.ARTIFACT_SOURCES['decode_qspi_rtl'],'decode_qspi_rtl')
assert obj['status']=='pass' and not errors,errors
shutil.copyfile(source,dest)
assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(dest.read_bytes()).digest()
print('SUMMARY promoted unchanged producer output; complete clean/mutation suite already run under current inputs')
