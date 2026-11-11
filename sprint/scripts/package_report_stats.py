from pathlib import Path
p=Path('sprint/scripts/report.py');s=p.read_text();needle="emit('## P5 serial overlap')"
insert="""if (D/'metadata_preflight.csv').exists():
 checked=list(csv.DictReader((D/'metadata_preflight.csv').open()))
 emit('Independent readonly preflight (MEASURED): python sprint/scripts/metadata_preflight.py intercepts the planned packer write, decodes the actual compact bytes independently, and compares all effective multipliers, shifts and two integer golden forwards per tensor. This does not modify production files or substitute for RTL validation. Artifacts metadata_preflight.csv and planned_packer_normalized_sha256.txt.')
 table(['Model','Tensors MEASURED','Serialized bytes saved MEASURED','All metadata / forwards equal MEASURED'],[[model,sum(r['model']==model for r in checked),sum(int(r['saved_bytes']) for r in checked if r['model']==model),all(r['all_metadata_equal']==r['integer_forward_equal']=='True' for r in checked if r['model']==model)] for model in ['260k','15m']])
"""
s=s.replace(needle,insert+needle)
needle="emit('## OWNER_RUN2.ps1')"
insert="""emit('Per-package source diff counts below are DERIVED by summing git diff --numstat e8d9282 for the listed groups; package scripts/data are separately visible in the complete stat.')
package_paths={'P1':['tools/decode/sim.py','tools/decode/run_*_rtl.py','tools/board/run_uart_rtl.py'],'P2':['tools/decode/provenance.py','tools/decode/test_gate.py'],'P3':['sprint/scripts/argmax_bound.py'],'P4':['tools/decode/pack_decode_image.py','rtl/decode/top_decode.sv','rtl/board/boolean_core.sv','tools/decode/fault_inject.py'],'P5':['sprint/P5_PREFETCH_REVIEW.md'],'P6':['OWNER_RUN2.ps1','bench/results/']}
stats=[]
for package,paths in package_paths.items():
 data=subprocess.run(['git','diff','e8d9282','--numstat','--',*paths],capture_output=True,text=True).stdout.splitlines();adds=dels=0
 for row in data:
  fields=row.split('\\t')
  if len(fields)>=3 and fields[0].isdigit():adds+=int(fields[0]);dels+=int(fields[1])
 stats.append([package,len(data),adds,dels])
table(['Package','Files DERIVED','Lines added DERIVED','Lines removed DERIVED'],stats)
"""
s=s.replace(needle,insert+needle)
p.write_text(s)
