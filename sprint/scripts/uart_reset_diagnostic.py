import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'tools/decode'));sys.path.insert(0,str(R))
sys.argv=['tools/board/run_uart_rtl.py','--sim','verilator','--gen','1','--fast','--replace','rtl/board/boolean_core.sv=rtl/board/boolean_core.sv','--output','sprint/data/uart_reset_diagnostic.json']
import importlib.util
spec=importlib.util.spec_from_file_location('uart_diagnostic',R/'tools/board/run_uart_rtl.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
trace=r'''
  always @(posedge rst) if(MODE==2) $display("TRACE RESET cycle=%%0d flash_phase=%%0d cs=%%b clk=%%b",cycles,flash.phase,pad_cs_n,pad_sclk);
  always @(posedge pad_cs_n) if(MODE==2) $display("TRACE CSRISE cycle=%%0d instr=%%h phase=%%0d bit=%%0d",cycles,flash.instr,flash.phase,flash.bitn);
  always @(posedge pad_sclk) if(MODE==2 && !pad_cs_n && flash.rises==7) begin #0.01; $display("TRACE OPCODE cycle=%%0d instr=%%h phase=%%0d pt=%%b",cycles,flash.instr,flash.phase,dut.pt_active); end
  always @(posedge clk) if(MODE==2 && dut.pt_active && flash.instr==8'h9f && dut.u_ctl.pstate==2 && dut.u_ctl.hc==1) $display("TRACE IDBIT cycle=%%0d bit=%%0d phase=%%0d oe=%%h raw=%%h registered=%%h ridx=%%0d rbit=%%0d",cycles,dut.u_ctl.bit_i,flash.phase,f_oe,bus,dut.io_i,flash.ridx,flash.rbit);
'''
m.TB=m.TB.replace('endmodule',trace+'endmodule')
base=m.compile_and_run
Path('sprint/data/uart_reset_trace.log').write_text('')
def traced(*args,**kwargs):
    rc,out=base(*args,**kwargs)
    with Path('sprint/data/uart_reset_trace.log').open('a') as f:f.write('\nRUN '+str(args[2])+'\n'+out)
    return rc,out
m.compile_and_run=traced
m.main()
