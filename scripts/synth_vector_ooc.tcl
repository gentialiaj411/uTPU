## Out-of-context timing/resource check of the Phase-3 vector sequencer.
set root [file normalize [file join [file dirname [info script]] ..]]
set part xc7s50csga324-1
set period 20.0
set name phase3_vector_ooc
create_project -force $name [file join $root build vivado_$name] -part $part
set files [list \
  [file join $root rtl vector vector_sequencer.sv] \
  [file join $root rtl vector vector_serial_alu.sv] \
  [file join $root rtl vector restoring_divider.sv]]
read_verilog -sv $files
synth_design -top vector_sequencer -part $part -flatten_hierarchy rebuilt
create_clock -name vector_clk -period $period [get_ports clk]
file mkdir [file join $root build reports]
report_timing_summary -file [file join $root build reports vector_ooc_timing.rpt]
report_utilization -file [file join $root build reports vector_ooc_utilization.rpt]
exit 0
