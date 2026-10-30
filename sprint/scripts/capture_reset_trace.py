from pathlib import Path
p=Path('sprint/scripts/uart_reset_diagnostic.py');s=p.read_text();s=s.replace('m.main()', '''base=m.compile_and_run
Path('sprint/data/uart_reset_trace.log').write_text('')
def traced(*args,**kwargs):
    rc,out=base(*args,**kwargs)
    with Path('sprint/data/uart_reset_trace.log').open('a') as f:f.write('\\nRUN '+str(args[2])+'\\n'+out)
    return rc,out
m.compile_and_run=traced
m.main()''');p.write_text(s)
