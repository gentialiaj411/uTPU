import sys,os,shutil
from pathlib import Path
backend=sys.argv[1];output=sys.argv[2];assert backend in ('icarus','verilator')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/decode'));os.environ['UTPU_SIM']=backend
sys.argv=['tools/decode/run_qspi_rtl.py','--output',output]
import sim,harness
isolated=ROOT/'build/decode'/('final_qspi_'+backend);isolated.mkdir(parents=True,exist_ok=True);harness.BUILD=isolated
base=sim.compile_and_run
def isolated_compile(sources,top,tb,*args,**kwargs):
    copy=Path(tb).with_name('final_'+backend+'_'+Path(tb).name);shutil.copyfile(tb,copy)
    return base(sources,top,copy,*args,**kwargs)
sim.compile_and_run=isolated_compile
import run_qspi_rtl as runner
runner.BUILD=isolated
runner.ARTIFACT_SOURCES['decode_qspi_rtl']=[*runner.ARTIFACT_SOURCES['decode_qspi_rtl'],'sprint/scripts/final_qspi_validation.py']
runner.main()
