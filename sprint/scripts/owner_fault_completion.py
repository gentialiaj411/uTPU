from pathlib import Path
p=Path('OWNER_RUN2.ps1');s=p.read_text();s=s.replace("Invoke-Logged -Name 'fault_qspi' -Exe $py -CommandArgs @('tools/decode/fault_inject.py','qspi')", "foreach ($unit in @('qspi','rms','vec','attn','attn15')) {\n    Invoke-Logged -Name ('fault_' + $unit) -Exe $py -CommandArgs @('tools/decode/fault_inject.py',$unit)\n}")
s=s.replace('Finish the timeboxed full QSPI mutation sweep before physical closure.', 'Finish timeboxed fault-provenance regeneration before physical closure.');p.write_text(s)
p=Path('sprint/scripts/report.py');s=p.read_text().replace('finishes the timeboxed QSPI fault sweep, then runs','finishes timeboxed fault sweeps, then runs');p.write_text(s)
