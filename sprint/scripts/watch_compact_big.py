from pathlib import Path
p=Path('sprint/scripts/finalize.py');s=p.read_text();needle="jobs={33636:'argmax_bound',new['engineering']:'engineering_launcher'}";s=s.replace(needle,needle+"\nextra=json.loads(Path('sprint/data/compact_big_job.json').read_text());jobs[extra['pid']]='compact_big'");p.write_text(s)
p=Path('sprint/scripts/report.py');s=p.read_text();s=s.replace("small=load('15m_small.json',{});big=load('15m_big.json',{});", "small=load('15m_small.json',{});big=load('15m_big.json',{});compact_big=load('15m_big_compact.json',{});")
s=s.replace("summary=[", "if p4status=='DONE' and compact_big.get('status')!='pass':p4status='TIMEBOXED; standard suite passed, post-compaction large regression incomplete'\nsummary=[")
s=s.replace("for n,o in [('small',small),('8x32',big)]", "for n,o in [('small baseline',small),('8x32 baseline',big),('8x32 compact image',compact_big)]")
s=s.replace("emit('Validation commands: python -m pytest", "emit('Post-compaction large regression command: python sprint/scripts/compact_big.py. It uses an isolated generated-input directory and an unused compile tag to prevent collisions with standard regeneration; the producer records this wrapper as an extra provenance dependency. Artifact 15m_big_compact.json.\\n')\nemit('Validation commands: python -m pytest")
p.write_text(s)
