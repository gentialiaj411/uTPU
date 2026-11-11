from pathlib import Path
p=Path('OWNER_RUN2.ps1');s=p.read_text();needle="$synthJobs = @("
s=s.replace(needle,"# Finish the timeboxed full QSPI mutation sweep before physical closure.\n$env:UTPU_SIM = 'icarus'\nInvoke-Logged -Name 'fault_qspi' -Exe $py -CommandArgs @('tools/decode/fault_inject.py','qspi')\n"+needle);p.write_text(s)
p=Path('sprint/scripts/report.py');s=p.read_text().replace('TIMEBOXED; metadata smokes passed but full fault/suite acceptance incomplete','TIMEBOXED; large regression and unit suites pass, complete fault acceptance unfinished');s=s.replace("recovery=load('fast_fault_recovery.json',{})", "recovery=load('fast_fault_recovery.json',{})\nrecovery.update(load('remaining_fault_recovery.json',{}))")
s=s.replace('contains sequential logged Vivado jobs and final gate', 'finishes the timeboxed QSPI fault sweep, then runs sequential logged Vivado jobs and final gate')
p.write_text(s)
