from pathlib import Path
p=Path('sprint/scripts/report.py');s=p.read_text().replace('Static fault-target audit:', 'Static RTL-kind fault-target audit:').replace('confirms every RTL mutation needle', 'confirms every RTL-kind mutation needle')
s=s.replace('Checked product bounds reject INT64 overflow rather than silently wrap.', 'For the pinned classifier, bound_domain_proof.json certifies the actual shift range lies in the hardware six-bit shift domain and the worst legal INT32 activation-scale L1 product fits INT64. Per-input checks also reject overflow rather than silently wrap. Coefficient products use the limb construction described below.')
p.write_text(s)
p=Path('sprint/scripts/argmax_bound.py');s=p.read_text().replace('assert np.all(SH>=0) and np.all(M>=0)', 'assert np.all((SH>=0)&(SH<=63)) and np.all(M>=0)');p.write_text(s)
