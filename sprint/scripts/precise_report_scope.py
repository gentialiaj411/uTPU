from pathlib import Path
p=Path('sprint/scripts/report.py');s=p.read_text().replace('maximum position127.', 'full sequence length128; the last classifier evaluation forwards input position126, and predicts the final generated token at index127.')
s=s.replace('The readonly preflight independently decodes every model tensor', 'The readonly preflight independently decodes every model tensor')
s=s.replace('Command python sprint/scripts/argmax_bound.py.', 'Command python sprint/scripts/argmax_bound.py. Input model/text/prompt/numeric hashes are in sprint_input_hashes.json.')
s=s.replace('decodes the actual compact bytes independently,', 'loads the original packer from integration commit e8d9282, decodes the actual compact bytes independently, verifies all packed weight-code payloads are unchanged,')
s=s.replace('No confirmed real bug;', 'No confirmed real bug;')
needle="emit('## P2 provenance')"
insert="""emit('Final model-compatibility repair: after compaction changed reset timing, Verilator retained an old flash array-data nibble during a later RDID despite correct instruction/register state, whereas Icarus passed all reset checks. Setting only the inactive flash_o value to zero when flash_oe=0 removes redundant internal tri-state encoding. The external bus remains Z under its explicit enable, active X/tV windows and all protocol/timing checks are unchanged. The single-timed-process prototype failed and was not adopted. Evidence: uart_compact_failure.json, uart_reset_detail_trace.log, uart_reset_icarus_trace.log and uart_reset_idle_data_trace.log. Full final-model revalidation commands/status are in engineering_status.json.\\n')
"""
s=s.replace(needle,insert+needle)
p.write_text(s)
