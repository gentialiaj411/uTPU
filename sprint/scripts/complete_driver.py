from pathlib import Path
p=Path('sprint/scripts/engineering_driver.py');s=p.read_text();needle='# Keep strict gate strict;'
insert='''# Regenerate final e2e artifacts through their producers when the accelerated backend passed equivalence.
if passed:
 e2es=[('kv8_p1',['--model','260k','--kv-bits','8']),('kv4_p1',['--model','260k','--kv-bits','4']),('kv8_p2',['--model','260k','--kv-bits','8','--period','2']),('kv8_p4',['--model','260k','--kv-bits','8','--period','4']),('kv8_p1_qspi',['--model','260k','--kv-bits','8','--flash','qspi']),('15m_kv4_p1',['--model','15m','--kv-bits','4'])]
 for name,args in e2es:
  if time.monotonic()>deadline-1800:break
  if run([sys.executable,'tools/decode/run_decode_rtl.py',*args,'--gen','8','--sim',backend,'--output','bench/results/decode_e2e_rtl_'+name+'.json'],'regen_e2e_'+name,1800):break
'''
s=s.replace(needle,insert+needle);p.write_text(s)
