import subprocess,time,json
from pathlib import Path
decode=['decode_pkg','seq_mul','mul_pipe','div_const_pipe','gemv_unit','act_quant','rmsnorm','vec_alu','attention','top_decode','qspi_reader']
board=['uart_rx','uart_tx','qspi_pad_regs','board_ctrl','boolean_core']
src=[f'rtl/decode/{x}.sv' for x in decode]+[f'rtl/board/{x}.sv' for x in board]+['tools/decode/tb/qspi_flash_model.sv','build/decode/uart_robust_fast_tb.sv']
cmd=[r'C:\iverilog\bin\iverilog.exe','-g2012','-s','uart_tb','-o','build/decode/uart_reset_diagnostic.vvp',*src]
t=time.perf_counter()
with open('sprint/data/uart_reset_icarus_compile.log','w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
if r.returncode:raise SystemExit(r.returncode)
with open('sprint/data/uart_reset_icarus_trace.log','w') as f:r=subprocess.run([r'C:\iverilog\bin\vvp.exe','build/decode/uart_reset_diagnostic.vvp'],stdout=f,stderr=subprocess.STDOUT,timeout=600)
Path('sprint/data/uart_reset_icarus_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t,returncode=r.returncode,command=cmd),indent=2))
