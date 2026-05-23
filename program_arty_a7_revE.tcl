## Vivado Tcl flow: build and program Arty A7-100T (Rev E) via USB-UART design.
## Usage (vivado-gui):
##   vivado-gui -source program_arty_a7_revE.tcl

set script_dir [file normalize [file dirname [info script]]]
set proj_dir   [file join $script_dir build vivado_arty_a7]
set proj_name  uTPU_arty_a7
set part_name  xc7a100tcsg324-1
set top_name   top

# Create project
create_project -force $proj_name $proj_dir -part $part_name

# Add RTL sources via explicit whitelist to avoid pulling in simulation/testbench files.
proc add_whitelist_dir {base rel} {
    set files [glob -nocomplain -directory [file join $base $rel] *.sv]
    set out {}
    foreach f $files {
        set name [string tolower [file tail $f]]
        if {[string match "*icarus*" $name]} { continue }
        if {[string match "tb_*" $name]} { continue }
        if {[string match "*_tb*" $name]} { continue }
        lappend out $f
    }
    return [lsort $out]
}

set rtl_files {}
set rtl_files [concat $rtl_files [add_whitelist_dir $script_dir "rtl/top"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $script_dir "rtl/PEArray"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $script_dir "rtl/UART"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $script_dir "rtl/fifo"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $script_dir "rtl/memory"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $script_dir "rtl/quantizer"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $script_dir "rtl/LeakyReLU"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $script_dir "rtl/unified_buffer"]]
lappend rtl_files [file join $script_dir generated generated_params.sv]

if {[llength $rtl_files] == 0} {
    error "No RTL source files found in whitelist"
}
read_verilog -sv $rtl_files

# Add constraints
set xdc_file [file join $script_dir arty_a7_revE_usb_uart.xdc]
read_xdc $xdc_file

# Set top
set_property top $top_name [current_fileset]

# Synthesize, implement, bitstream
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Program device
open_hw_manager
connect_hw_server
open_hw_target
set hw_device [lindex [get_hw_devices] 0]
current_hw_device $hw_device
refresh_hw_device $hw_device

set bitfile [file join $proj_dir $proj_name.runs impl_1 ${top_name}.bit]
set_property PROGRAM.FILE $bitfile $hw_device
program_hw_devices $hw_device

puts "Programmed $hw_device with $bitfile"
