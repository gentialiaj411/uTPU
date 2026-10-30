from pathlib import Path
p=Path('sprint/scripts/report.py');s=p.read_text();needle="emit('## Evidence rules and environment')"
insert="""emit('Additional confirmed verification findings: sparse QSPI used multiple stateful RNG calls in task arguments, whose evaluation order differed across simulators; explicit ordered sampling now preserves the full stimulus set while making it deterministic. UART queue polling raced active-edge monitor updates; post-edge sampling removes the four-cycle summary skew without changing checks. The initial QSPI/UART compile failures were also traced to a full 7.7GB tmpfs and zero-byte makefiles, not RTL errors. Failed outputs and disk evidence are retained. Finally, the original research 50-prompt record list contained only nine distinct eight-token prefixes; P3 now requires 50 distinct prefixes and records them explicitly.\\n')
"""
s=s.replace(needle,insert+needle)
s=s.replace("times={r['label']:r for r in csv.DictReader((D/'crosscheck.csv').open())}", "times={r['label']:r for r in csv.DictReader((D/'crosscheck.csv').open())}\n for recovery in load('deterministic_recovery_complete.json',[]):times[recovery['label']]=recovery")
p.write_text(s)
