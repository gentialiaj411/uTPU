# Invoked by program_flash.py (which sets ::mcs_file, ::prm_file, ::bit_file and ::mode). Vivado Hardware Manager.
#   mode flash: indirect flash programming of the MCS (bitstream at 0x0 + weight image at 0x280000), then reboot from flash.
#   mode jtag : load the .bit straight into the FPGA (volatile; the weights must already be in flash at 0x280000).
# The flash flow is the one the Vivado GUI emits for "Add Configuration Memory Device" + "Program Configuration Memory
# Device" (UG908, "Programming Configuration Memory Devices"): Vivado first loads its own indirect-programming bitstream
# (PROGRAM.HW_CFGMEM_BITFILE) into the FPGA so the flash is reachable over JTAG, then erases/programs/verifies.
open_hw_manager
connect_hw_server
open_hw_target
set hw_device [lindex [get_hw_devices xc7s50*] 0]
if {$hw_device eq ""} { error "no xc7s50 on the JTAG chain: [get_hw_devices]" }
current_hw_device $hw_device
refresh_hw_device -update_hw_probes false $hw_device

if {$::mode eq "jtag"} {
    set_property PROGRAM.FILE $::bit_file $hw_device
    program_hw_devices $hw_device
    refresh_hw_device $hw_device
    puts "UTPU: FPGA configured over JTAG from $::bit_file (DONE=[get_property REGISTER.IR.BIT5_DONE $hw_device])"
} else {
    set cfgpart [lindex [get_cfgmem_parts {s25fl128sxxxxxx0-spi-x1_x2_x4}] 0]
    if {$cfgpart eq ""} { error "cfgmem part s25fl128sxxxxxx0-spi-x1_x2_x4 not in this Vivado's part list (VERIFY: get_cfgmem_parts *s25fl128*)" }
    create_hw_cfgmem -hw_device $hw_device -mem_dev $cfgpart
    set cfgmem [get_property PROGRAM.HW_CFGMEM $hw_device]
    set_property PROGRAM.ADDRESS_RANGE {use_file} $cfgmem
    set_property PROGRAM.FILES [list $::mcs_file] $cfgmem
    set_property PROGRAM.PRM_FILE $::prm_file $cfgmem
    set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} $cfgmem
    set_property PROGRAM.BLANK_CHECK 0 $cfgmem
    set_property PROGRAM.ERASE 1 $cfgmem
    set_property PROGRAM.CFG_PROGRAM 1 $cfgmem
    set_property PROGRAM.VERIFY 1 $cfgmem
    set_property PROGRAM.CHECKSUM 0 $cfgmem
    # load the indirect-programming core so the flash is reachable through the FPGA
    create_hw_bitstream -hw_device $hw_device [get_property PROGRAM.HW_CFGMEM_BITFILE $hw_device]
    program_hw_devices $hw_device
    refresh_hw_device $hw_device
    program_hw_cfgmem -hw_cfgmem $cfgmem
    puts "UTPU: flash erased, programmed and verified from $::mcs_file"
    # reconfigure from flash (same as pressing PROG): the board now runs the bitstream that is in flash
    boot_hw_device $hw_device
    refresh_hw_device $hw_device
    puts "UTPU: rebooted from flash (DONE=[get_property REGISTER.IR.BIT5_DONE $hw_device])"
}
close_hw_target
disconnect_hw_server
close_hw_manager
