## OOC synth for bstore_wide_arm_ooc LUT estimate (no IOB/place).
## Usage:
##   vivado -mode batch -source scripts/synth_bstore_wide_ooc.tcl \
##     -tclargs WIDTH 8 report_prefix bstore_wide_ooc_w8

proc parse_kv_args {argv} {
    set out [dict create]
    set argc [llength $argv]
    set i 0
    while {$i < $argc} {
        set arg [lindex $argv $i]
        if {[regexp {^([^=]+)=(.*)$} $arg -> key value]} {
            dict set out $key $value
        } elseif {$i + 1 < $argc} {
            set next [lindex $argv [expr {$i + 1}]]
            if {[regexp {^([^=]+)=(.*)$} $next]} {
                dict set out $arg 1
            } else {
                dict set out $arg $next
                incr i
            }
        } else {
            dict set out $arg 1
        }
        incr i
    }
    return $out
}

proc maybe_get {opts key default} {
    if {[dict exists $opts $key]} {
        return [dict get $opts $key]
    }
    return $default
}

set script_dir [file normalize [file dirname [info script]]]
set repo_root  [file normalize [file join $script_dir ..]]
set arg_opts   [parse_kv_args $argv]
set part_name  xc7a100tcsg324-1
set width      [maybe_get $arg_opts WIDTH 1]
set report_prefix [string trim [maybe_get $arg_opts report_prefix "bstore_wide_ooc"]]
set reports_dir [file join $repo_root build reports]
file mkdir $reports_dir

create_project -in_memory -part $part_name
read_verilog -sv [file join $repo_root rtl top bstore_wide_arm_ooc.sv]
set_property top bstore_wide_arm_ooc [current_fileset]
set_property generic "WIDTH=$width ADDR_W=12 DATA_W=16" [get_filesets sources_1]

synth_design -mode out_of_context -top bstore_wide_arm_ooc -part $part_name
report_utilization -file [file join $reports_dir ${report_prefix}_utilization.rpt]
report_utilization -hierarchical -file [file join $reports_dir ${report_prefix}_utilization_hier.rpt]
puts "BSTORE_WIDE_OOC_DONE width=$width prefix=$report_prefix"
