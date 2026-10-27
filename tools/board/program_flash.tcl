# NOT RUN by the M5 work (no hardware access). With the Boolean connected over USB-JTAG:
#   C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat -mode batch -nojournal -nolog -source tools/board/program_flash.tcl
# Programs build/boolean/boolean_flash.mcs (bitstream + weight image) into the S25FL128S. The 16 MB erase/program/verify
# takes several minutes. Power-cycle (or boot_hw_device) afterwards.
# Part: s25fl128sxxxxxx0-spi-x1_x2_x4 assumes 256 kB uniform sectors (RDID byte 4 = 00h: run
# `boolean_host.py id` first; if it reads 01h use the ...xxxxxx1... part).
open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices xc7s50_0] 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev
create_hw_cfgmem -hw_device $dev [lindex [get_cfgmem_parts {s25fl128sxxxxxx0-spi-x1_x2_x4}] 0]
set cm [get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.ADDRESS_RANGE  {use_file} $cm
set_property PROGRAM.FILES          [list build/boolean/boolean_flash.mcs] $cm
set_property PROGRAM.PRM_FILE       {} $cm
set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} $cm
set_property PROGRAM.BLANK_CHECK    0 $cm
set_property PROGRAM.ERASE          1 $cm
set_property PROGRAM.CFG_PROGRAM    1 $cm
set_property PROGRAM.VERIFY         1 $cm
set_property PROGRAM.CHECKSUM       0 $cm
create_hw_bitstream -hw_device $dev [get_property PROGRAM.HW_CFGMEM_BITFILE $dev]
program_hw_devices $dev
program_hw_cfgmem -hw_cfgmem $cm
boot_hw_device $dev
