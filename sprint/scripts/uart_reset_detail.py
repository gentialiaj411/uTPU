import os,sys,re,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2];os.chdir(R);sys.path.insert(0,str(R/'tools/decode'));os.environ['UTPU_SIM']='verilator'
from sim import compile_and_run
text=Path('build/decode/uart_robust_fast_tb.sv').read_text();isolated=Path('build/decode/uart_reset_detail');isolated.mkdir(parents=True,exist_ok=True)
for source in re.findall(r'"([A-Za-z]:/[^"\n]*uart_[^"\n]*\.mem)"',text):
 p=Path(source);dest=isolated/p.name;shutil.copyfile(p,dest);text=text.replace(source,dest.resolve().as_posix())
text=text.replace('rbit=%0d",cycles', 'rbit=%0d f_o=%h padT=%h padO=%h regbyte=%h ID0=%h",cycles').replace('flash.ridx,flash.rbit);','flash.ridx,flash.rbit,f_o,pad_io_t,pad_io_o,flash.regbyte,flash.id_bytes[0]);')
tb=isolated/'uart_reset_detail_tb.sv';tb.write_text(text)
decode=['decode_pkg','seq_mul','mul_pipe','div_const_pipe','gemv_unit','act_quant','rmsnorm','vec_alu','attention','top_decode','qspi_reader'];board=['uart_rx','uart_tx','qspi_pad_regs','board_ctrl','boolean_core']
sources=[f'rtl/decode/{x}.sv' for x in decode]+[f'rtl/board/{x}.sv' for x in board]+[os.environ.get('UTPU_FLASH_MODEL_OVERRIDE','tools/decode/tb/qspi_flash_model.sv')]
rc,out=compile_and_run(sources,'uart_tb',tb,timeout=300)
Path(os.environ.get('UTPU_TRACE_OUT','sprint/data/uart_reset_detail_trace.log')).write_text(out)
print('SUMMARY diagnostic_exit='+str(rc))
