from pathlib import Path
p=Path('tools/decode/run_decode_rtl.py');s=p.read_text();s=s.replace('    prompts = prompt_bank[:a.prompts]', '''    if a.prompts < 1 or a.gen < 1:
        ap.error('prompts and gen must be positive')
    if a.prompts > len(prompt_bank):
        from tools.llama2c.tokenizer import Tokenizer
        tokenizer = Tokenizer(ROOT / 'models/llama2c/tokenizer.bin', 32000) if a.model == '15m' else Tokenizer(ROOT / 'models/llama2c/stories260K/tok512.bin', 512)
        stories = (ROOT / 'models/llama2c/TinyStories-valid.txt').read_text(encoding='utf-8').split('<|endoftext|>')
        for story in stories:
            if story.strip():
                candidate = tokenizer.encode(story.strip()[:160])[:8]
                if candidate not in prompt_bank: prompt_bank.append(candidate)
            if len(prompt_bank) >= a.prompts: break
    if len(prompt_bank) < a.prompts:
        ap.error('not enough distinct validation prompts')
    prompts = prompt_bank[:a.prompts]
    if max(map(len,prompts)) + a.gen - 1 > (64 if a.model == '260k' else 128):
        ap.error('prompt plus generation exceeds hardware context')''');p.write_text(s)
for p in [*Path('tools/decode').glob('run_*_rtl.py'),Path('tools/board/run_uart_rtl.py')]:
 s=p.read_text();needle='    a = ap.parse_args()';needle2='    args = ap.parse_args()'
 line="    ap.add_argument('--sim', choices=('icarus','verilator'), default=__import__('sim').SIMULATOR, help='simulation backend (default icarus)')\n"
 if "add_argument('--sim'" not in s:
  if needle in s:s=s.replace(needle,line+needle)
  elif needle2 in s:s=s.replace(needle2,line+needle2)
 p.write_text(s)
