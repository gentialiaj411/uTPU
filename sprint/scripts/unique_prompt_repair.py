from pathlib import Path
p=Path('sprint/scripts/argmax_bound.py');s=p.read_text();old="prompts=json.loads((R.parent/'uTPU-research/research/data/prompts.json').read_text())\nrecords=[];deadline=time.monotonic()+90*60"
new="""old_prompts=json.loads((R.parent/'uTPU-research/research/data/prompts.json').read_text())
prompts=[];first_indices=[]
for i,prompt in enumerate(old_prompts):
 if prompt not in prompts:prompts.append(prompt);first_indices.append(i)
reused=len(prompts)
from tools.llama2c.tokenizer import Tokenizer
tokenizer=Tokenizer(R/'models/llama2c/tokenizer.bin',32000)
for story in (R/'models/llama2c/TinyStories-valid.txt').read_text(encoding='utf-8').split('<|endoftext|>'):
 if story.strip():
  candidate=tokenizer.encode(story.strip()[:160])[:8]
  if len(candidate)==8 and candidate not in prompts:prompts.append(candidate)
 if len(prompts)==50:break
assert len(prompts)==50 and len({tuple(x) for x in prompts})==50
(D/'argmax_prompts.json').write_text(json.dumps(prompts))
records=[]
if (D/'argmax_repeated_prefix_dataset.csv').exists():
 mapping={old:new for new,old in enumerate(first_indices)}
 with (D/'argmax_repeated_prefix_dataset.csv').open() as source:
  for row in csv.DictReader(source):
   if int(row['prompt']) in mapping:
    row['prompt']=mapping[int(row['prompt'])];records.append(row)
deadline=time.monotonic()+90*60"""
assert old in s;s=s.replace(old,new).replace('for pi,prompt in enumerate(prompts[:50]):\n rt.reset()', 'for pi,prompt in enumerate(prompts[:50]):\n if pi<reused:continue\n rt.reset()')
s+='\n(D/"argmax_unique_complete.json").write_text(json.dumps(dict(unique_prompts=len(prompts),records=len(records))))\n'
p.write_text(s)
p=Path('sprint/.gitignore');p.write_text(p.read_text()+'data/argmax_repeated_prefix_dataset.csv\n')
p=Path('sprint/scripts/report.py');s=p.read_text();s=s.replace('first 50 research prompts derived from TinyStories validation, eight prompt tokens', '50 distinct eight-token prefixes from TinyStories validation (argmax_prompts.json); a diversity audit found the original research 50-record prompt list had only nine unique prefixes, so nine complete measurements were reused and 41 distinct prompts were added. Eight prompt tokens')
p.write_text(s)
