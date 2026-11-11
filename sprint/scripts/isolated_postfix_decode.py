import sys,os,shutil
from pathlib import Path
backend,flash,output=sys.argv[1:4];assert backend in ('icarus','verilator') and flash in ('ideal','qspi')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/decode'));os.environ['UTPU_SIM']=backend
sys.argv=['tools/decode/run_decode_rtl.py','--model','260k','--kv-bits','4','--gen','8','--sim',backend,'--flash',flash,'--output',output]
import sim,harness
isolated=ROOT/'build/decode'/('postfix_'+backend+'_'+flash);isolated.mkdir(parents=True,exist_ok=True);harness.BUILD=isolated
base=sim.compile_and_run
def isolated_compile(sources,top,tb,*args,**kwargs):
 copy=Path(tb).with_name('postfix_'+backend+'_'+flash+'_'+Path(tb).name);shutil.copyfile(tb,copy)
 return base(sources,top,copy,*args,**kwargs)
sim.compile_and_run=isolated_compile
import run_decode_rtl as runner
runner.BUILD=isolated;sources=runner.e2e_sources;runner.e2e_sources=lambda model,mode:[*sources(model,mode),'sprint/scripts/isolated_postfix_decode.py']
runner.main()
