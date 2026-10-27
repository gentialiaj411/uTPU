# NOT RUN by the M5 work (no bitstream was generated). From the repo root:
#   C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat -mode batch -nojournal -nolog -source tools/board/make_bitstream.tcl
# Produces build/boolean/boolean_top.bit and build/boolean/boolean_flash.mcs (bitstream at 0x0, weight image at 0x280000).
# Config stays x1 on purpose: the board then boots whether or not the flash QUAD bit is set. The design itself needs
# QUAD = 1 for its 6Bh reads; set it with tools/board/boolean_host.py enable-quad (see docs/r3_progress.md, M5).
# Prerequisite: build/decode/decode_image_15m/image.bin (python tools/decode/pack_decode_image.py, stories15M).
file mkdir build/boolean
foreach f {rtl/decode/decode_pkg.sv rtl/decode/seq_mul.sv rtl/decode/mul_pipe.sv rtl/decode/div_const_pipe.sv rtl/decode/gemv_unit.sv
           rtl/decode/act_quant.sv rtl/decode/rmsnorm.sv rtl/decode/vec_alu.sv rtl/decode/attention.sv rtl/decode/top_decode.sv
           rtl/decode/qspi_reader.sv rtl/board/uart_rx.sv rtl/board/uart_tx.sv rtl/board/board_ctrl.sv rtl/board/boolean_core.sv
           rtl/board/qspi_pads.sv rtl/board/boolean_top.sv} {
    read_verilog -sv $f
}
read_xdc constraints/boolean_xc7s50.xdc
synth_design -top boolean_top -part xc7s50csga324-1
opt_design
place_design
route_design
report_timing_summary -file build/boolean/timing.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
if {$wns < 0} { error "timing not met (WNS $wns ns); not writing a bitstream" }
write_bitstream -force build/boolean/boolean_top.bit
write_cfgmem -format mcs -interface spix1 -size 16 \
    -loadbit {up 0x00000000 build/boolean/boolean_top.bit} \
    -loaddata {up 0x00280000 build/decode/decode_image_15m/image.bin} \
    -file build/boolean/boolean_flash.mcs -force
