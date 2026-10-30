from pathlib import Path
p=Path('sprint/scripts/report.py');s=p.read_text();needle="emit('# uTPU engineering sprint report')"
insert="""meta=load('15m_metadata.json',{})
if meta.get('cycles_per_token_mean'):
 summary[6]=f'P4 {p4status}; 15M QSPI bytes/token {meta[\"stream_bytes_per_token\"]:,}, clocks/token {meta[\"cycles_per_token_mean\"]:,.1f} (MEASURED); {50e6/meta[\"cycles_per_token_mean\"]:.5f} tok/s at 50MHz (DERIVED).'
"""
s=s.replace(needle,insert+needle)
s=s.replace("emit('## P5 serial overlap')", "emit('Static fault-target audit: python sprint/scripts/fault_needle_audit.py confirms every RTL mutation needle still matches the current source after compaction. This is a source-integrity check, not a substitute for executing mutations; data/fault_needle_audit.csv.\\n')\nemit('## P5 serial overlap')")
p.write_text(s)
