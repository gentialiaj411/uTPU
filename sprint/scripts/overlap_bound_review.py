from pathlib import Path
p=Path('sprint/scripts/report.py');s=p.read_text();needle="emit('## P6 regeneration and strict gate')"
insert="""overlap=load('e2e_qspi_icarus.json',{})
if overlap.get('stream_bytes_per_token') and overlap.get('cycles_per_token_mean'):
 floor=overlap['stream_bytes_per_token']/.25;ceiling=100*(overlap['cycles_per_token_mean']/floor-1)
 emit(f'Independent overlap limit DERIVED for the measured short 260K QSPI workload: bytes/rate={overlap[\"stream_bytes_per_token\"]}/0.25={floor:.1f} clocks/token versus measured {overlap[\"cycles_per_token_mean\"]:.1f}. Even eliminating every non-bandwidth clock would improve throughput by at most {ceiling:.4f} percent. Source e2e_qspi_icarus.json, command in crosscheck.py. This does not bound long-context or faster-stream cases; it explains why another prefetch enable is a poor short-case optimization.')
"""
s=s.replace(needle,insert+needle);p.write_text(s)
