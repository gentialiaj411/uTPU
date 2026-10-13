## Program an already-built bitstream over Vivado Hardware Manager.
## Usage: vivado -mode batch -source tools/program_bitstream.tcl -tclargs bitstream <file.bit>

if {[llength $argv] != 2 || [lindex $argv 0] ne "bitstream"} {
    error "usage: -tclargs bitstream <file.bit>"
}
set bitstream [file normalize [lindex $argv 1]]
if {![file exists $bitstream]} {
    error "bitstream not found: $bitstream"
}
open_hw_manager
connect_hw_server
open_hw_target
set device [lindex [get_hw_devices] 0]
if {$device eq ""} {
    error "no FPGA hardware device discovered"
}
current_hw_device $device
refresh_hw_device $device
set_property PROGRAM.FILE $bitstream $device
program_hw_devices $device
puts "PROGRAM_OK $bitstream"
close_hw_manager
