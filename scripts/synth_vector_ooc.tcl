## Out-of-context timing/resource check of the Phase-3 vector sequencer.
set root [file normalize [file join [file dirname [info script]] ..]]
set part xc7s50csga324-1
set period 20.0
set name phase3_vector_ooc
create_project -force $name [file join $root build vivado_$name] -part $part
# Deliberately compile the entire vector implementation directory.  Keeping a
# hand-maintained subset here made the former OOC report blind to new vector
# sources (notably vector_alu.sv).
set files [lsort [glob -nocomplain [file join $root rtl vector *.sv]]]
if {[llength $files] == 0} { error "no rtl/vector/*.sv sources found" }
read_verilog -sv $files
synth_design -top vector_sequencer -part $part -flatten_hierarchy rebuilt
create_clock -name vector_clk -period $period [get_ports clk]
file mkdir [file join $root build reports]
report_timing_summary -file [file join $root build reports vector_ooc_timing.rpt]
report_utilization -file [file join $root build reports vector_ooc_utilization.rpt]
exit 0
