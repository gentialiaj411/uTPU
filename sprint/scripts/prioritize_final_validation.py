from pathlib import Path
p=Path('sprint/scripts/engineering_driver.py');s=p.read_text();s=s.replace("['gemv','actq','rms','vec','attn','attn15','top','qspi','board']", "['gemv','top','board','qspi','actq','rms','vec','attn','attn15']")
needle='# Keep strict gate strict;'
insert="""# Use the reserved final regeneration window for any still-unrun fault suite.
for unit in ['gemv','top','board','qspi','actq','rms','vec','attn','attn15']:
 if 'fault_'+unit in state:continue
 if time.monotonic()>deadline-300:break
 if run([sys.executable,'tools/decode/fault_inject.py',unit],'fault_'+unit,1800):break
"""
s=s.replace(needle,insert+needle);p.write_text(s)
