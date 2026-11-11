## Write bitstream from an already-routed DCP (no re-synth).
## Usage (repo root):
##   vivado -mode batch -source tools/write_bitstream_from_dcp.tcl \
##     -tclargs dcp <path/to/top_routed.dcp> bit_out <path/to/out.bit>

if {[llength $argv] < 1} {
    puts "ERROR: pass -tclargs dcp <file> bit_out <file>"
    exit 1
}

set arg_opts [dict create]
for {set i 0} {$i < [llength $argv]} {incr i} {
    set k [lindex $argv $i]
    incr i
    dict set arg_opts $k [lindex $argv $i]
}

if {![dict exists $arg_opts dcp] || ![dict exists $arg_opts bit_out]} {
    puts "ERROR: required args: dcp <path> bit_out <path>"
    exit 1
}

set dcp [file normalize [dict get $arg_opts dcp]]
set bit_out [file normalize [dict get $arg_opts bit_out]]

if {![file exists $dcp]} {
    puts "ERROR: DCP not found: $dcp"
    exit 1
}

file mkdir [file dirname $bit_out]
open_checkpoint $dcp
write_bitstream -force $bit_out
puts "Wrote bitstream: $bit_out"
