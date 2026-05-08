## Vivado batch flow for uTPU array-size comparison.
## Usage:
##   vivado -mode batch -source scripts/array_size_sweep.tcl
##
## Produces reports under:
##   build/reports/array_8/
##   build/reports/array_16/

set script_dir [file normalize [file dirname [info script]]]
set repo_dir   [file normalize [file join $script_dir ..]]
set part_name  xc7a100tcsg324-1
set top_name   top
set xdc_file   [file join $repo_dir arty_a7_revE_usb_uart.xdc]
set sizes      {8 16}

proc collect_sv_recursive {root} {
    set out {}
    if {![file exists $root]} {
        return $out
    }
    foreach p [glob -nocomplain -directory $root *] {
        if {[file isdirectory $p]} {
            set out [concat $out [collect_sv_recursive $p]]
        } elseif {[string equal [file extension $p] ".sv"]} {
            lappend out [file normalize $p]
        }
    }
    return $out
}

set rtl_files [collect_sv_recursive [file join $repo_dir rtl]]
set gen_files [collect_sv_recursive [file join $repo_dir generated]]
set all_sv    [concat $rtl_files $gen_files]

if {[llength $all_sv] == 0} {
    puts "ERROR: No SystemVerilog files found under rtl/ or generated/."
    exit 1
}

if {![file exists $xdc_file]} {
    puts "ERROR: Constraints file not found: $xdc_file"
    exit 1
}

foreach sz $sizes {
    set project_name "uTPU_arty_a7_a${sz}"
    set project_dir  [file join $repo_dir build "vivado_${project_name}"]
    set report_dir   [file join $repo_dir build reports "array_${sz}"]

    file mkdir $project_dir
    file mkdir $report_dir

    puts "============================================================"
    puts "Building ARRAY_SIZE=$sz"
    puts "Project: $project_dir"
    puts "Reports: $report_dir"
    puts "============================================================"

    create_project -force $project_name $project_dir -part $part_name
    read_verilog -sv $all_sv
    read_xdc $xdc_file

    set_property top $top_name [current_fileset]
    set_property generic "ARRAY_SIZE=$sz" [current_fileset]

    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    open_run synth_1

    report_utilization -file [file join $report_dir utilization_synth.rpt]
    report_utilization -hierarchical -file [file join $report_dir utilization_synth_hier.rpt]
    report_timing_summary -file [file join $report_dir timing_synth.rpt]

    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1
    open_run impl_1

    report_utilization -file [file join $report_dir utilization_impl.rpt]
    report_utilization -hierarchical -file [file join $report_dir utilization_impl_hier.rpt]
    report_timing_summary -file [file join $report_dir timing_impl.rpt]
    report_drc -file [file join $report_dir drc_impl.rpt]

    close_project
}

puts "Done. Compare report folders under build/reports/array_8 and array_16."
exit
