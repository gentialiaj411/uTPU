from pathlib import Path
source=Path('tools/decode/tb/qspi_flash_model.sv').read_text()
old="    always @(negedge sck) if (cs_n === 1'b0 && phase == 5) begin"
assert source.count(old)==1
source=source.replace(old,"    // One timed output process: register and array-data phases are mutually exclusive.\n    always @(negedge sck) if (cs_n === 1'b0) begin\n      if (phase == 5) begin")
old="    end\n    // data launch: after the last dummy rising edge, each falling edge starts the next nibble\n    always @(negedge sck) if (cs_n === 1'b0 && phase == 4) begin"
assert source.count(old)==1
source=source.replace(old,"    end else if (phase == 4) begin\n        // Array data launch, with the same tHO/tV and checks as before.")
assert source.endswith('    end\nendmodule\n')
source=source[:-len('    end\nendmodule\n')]+'    end\n    end\nendmodule\n'
Path('sprint/data/qspi_flash_model_single_process.sv').write_text(source)
p=Path('sprint/scripts/uart_reset_detail.py');s=p.read_text().replace("['tools/decode/tb/qspi_flash_model.sv']", "[os.environ.get('UTPU_FLASH_MODEL_OVERRIDE','tools/decode/tb/qspi_flash_model.sv')]")
s=s.replace("Path('sprint/data/uart_reset_detail_trace.log').write_text(out)", "Path(os.environ.get('UTPU_TRACE_OUT','sprint/data/uart_reset_detail_trace.log')).write_text(out)")
p.write_text(s)
