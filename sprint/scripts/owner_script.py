from pathlib import Path
old=(Path('../uTPU-integ/OWNER_RUN.ps1')).read_text();head=old[:old.index('$faultUnits')];tail=old[old.index('$synthJobs'):];tail=tail[:tail.index("Invoke-Logged -Name 'e2e_15m")]+"Invoke-Logged -Name 'strict_gate' -Exe $py -CommandArgs @('tools/decode_gate.py')\nWrite-Output 'OWNER_RUN2 complete'\n"
bootstrap="""Invoke-Logged -Name 'pack_15m' -Exe $py -CommandArgs @('tools/decode/pack_decode_image.py','--model','15m')
Invoke-Logged -Name 'wrap_15m' -Exe $py -CommandArgs @('tools/decode/gen_15m_wrap.py')
Invoke-Logged -Name 'wrap_qspi' -Exe $py -CommandArgs @('tools/decode/gen_qspi_wrap.py')
"""
Path('OWNER_RUN2.ps1').write_text((head+bootstrap+tail).replace('build/owner_logs','build/owner_logs2'))
