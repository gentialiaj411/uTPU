## Full-top synth+impl for one design-space point (route only, no bitstream).
## Writes build/reports/${report_prefix}_{timing_summary,utilization,route_status}.rpt
##
## Usage:
##   vivado -mode batch -source scripts/synth_design_space_point.tcl -tclargs \
##     ARRAY_SIZE 8 COMPUTE_DATA_WIDTH 8 MAX_BATCH_COUNT 48 clock_period 20 \
##     PROG_DEPTH 65536 QUANTIZER_PIPE_DEPTH 3 report_prefix dss_n8_cdw8_mb48_clk20

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

proc period_tag {period} {
    # Avoid Windows path dots from "15.0" -> use 15 when integral.
    if {[regexp {^([0-9]+)\.0+$} $period -> whole]} {
        return $whole
    }
    # Replace remaining dots with p for non-integral periods.
    regsub -all {\.} $period p period
    return $period
}

set script_dir [file normalize [file dirname [info script]]]
set repo_root  [file normalize [file join $script_dir ..]]
set arg_opts   [parse_kv_args $argv]
set part_name  xc7a100tcsg324-1
set top_name   top

set array_size [maybe_get $arg_opts ARRAY_SIZE 8]
set cdw        [maybe_get $arg_opts COMPUTE_DATA_WIDTH 8]
set adw        [maybe_get $arg_opts ACCUMULATOR_DATA_WIDTH ""]
if {$adw eq ""} {
    if {$cdw >= 8} {
        set adw 32
    } else {
        set adw 16
    }
}
set max_batch  [maybe_get $arg_opts MAX_BATCH_COUNT 48]
set prog_depth [maybe_get $arg_opts PROG_DEPTH 65536]
set buf_size   [maybe_get $arg_opts BUFFER_SIZE 4096]
set ext_addr   [maybe_get $arg_opts EXT_ADDR_EN 1]
set q_lanes    [maybe_get $arg_opts QUANTIZER_LANES $array_size]
set r_lanes    [maybe_get $arg_opts RELU_LANES $array_size]
set pipe_depth [maybe_get $arg_opts QUANTIZER_PIPE_DEPTH 3]
set clock_period [maybe_get $arg_opts clock_period 20]
set report_prefix [string trim [maybe_get $arg_opts report_prefix ""]]
set jobs       [maybe_get $arg_opts jobs 4]
set clk_tag    [period_tag $clock_period]

if {$report_prefix eq ""} {
    set report_prefix "dss_n${array_size}_cdw${cdw}_mb${max_batch}_clk${clk_tag}_pd${pipe_depth}_prog${prog_depth}"
}

set proj_name "uTPU_dss_n${array_size}_cdw${cdw}_mb${max_batch}_clk${clk_tag}"
set proj_dir  [file join $repo_root build "vivado_${proj_name}"]
file mkdir $proj_dir

create_project -force $proj_name $proj_dir -part $part_name

set rtl_files {}
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/top"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/PEArray"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/UART"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/fifo"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/memory"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/quantizer"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/LeakyReLU"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/unified_buffer"]]
lappend rtl_files [file join $repo_root generated generated_params.sv]
read_verilog -sv $rtl_files

set xdc_file [file join $repo_root arty_a7_revE_usb_uart.xdc]
set xdc_override [file join $proj_dir "${proj_name}_clock_override.xdc"]
set in_fh  [open $xdc_file r]
set out_fh [open $xdc_override w]
while {[gets $in_fh line] >= 0} {
    if {[string match "*create_clock*" $line]} {
        continue
    }
    puts $out_fh $line
}
close $in_fh
puts $out_fh "create_clock -name sys_clk_pin -period $clock_period -waveform {0 [expr {$clock_period / 2.0}]} \[get_ports { clk }\];"
close $out_fh
read_xdc $xdc_override

set_property top $top_name [current_fileset]
set generic_opt [join [list \
    "PROG_DEPTH=$prog_depth" \
    "COMPUTE_DATA_WIDTH=$cdw" \
    "ACCUMULATOR_DATA_WIDTH=$adw" \
    "ARRAY_SIZE=$array_size" \
    "BUFFER_SIZE=$buf_size" \
    "EXT_ADDR_EN=$ext_addr" \
    "MAX_BATCH_COUNT=$max_batch" \
    "QUANTIZER_LANES=$q_lanes" \
    "RELU_LANES=$r_lanes" \
    "QUANTIZER_PIPE_DEPTH=$pipe_depth" \
] " "]
set_property generic $generic_opt [current_fileset]
puts "DSS generics: $generic_opt"
puts "DSS clock_period=$clock_period report_prefix=$report_prefix proj=$proj_name"

# Match program_arty_a7_revE.tcl: sequential synth then impl-to-route (no bitstream).
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
set synth_progress [get_property PROGRESS [get_runs synth_1]]
set synth_status [get_property STATUS [get_runs synth_1]]
puts "synth_1 PROGRESS=$synth_progress STATUS=$synth_status"
if {$synth_progress ne "100%"} {
    puts "ERROR: synth_1 did not complete (PROGRESS=$synth_progress STATUS=$synth_status)"
    exit 1
}
if {[string match -nocase "*fail*" $synth_status] || [string match -nocase "*error*" $synth_status]} {
    puts "ERROR: synth_1 failed STATUS=$synth_status"
    catch {
        open_run synth_1
        set reports_dir [file join $repo_root build reports]
        file mkdir $reports_dir
        report_utilization -file [file join $reports_dir "${report_prefix}_utilization_synth.rpt"]
    }
    exit 1
}

launch_runs impl_1 -to_step route_design -jobs $jobs
wait_on_run impl_1
set impl_progress [get_property PROGRESS [get_runs impl_1]]
set impl_status [get_property STATUS [get_runs impl_1]]
puts "impl_1 PROGRESS=$impl_progress STATUS=$impl_status"

set reports_dir [file join $repo_root build reports]
file mkdir $reports_dir
set util_rpt  [file join $reports_dir "${report_prefix}_utilization.rpt"]
set tim_rpt   [file join $reports_dir "${report_prefix}_timing_summary.rpt"]
set route_rpt [file join $reports_dir "${report_prefix}_route_status.rpt"]

set opened 0
if {![catch {open_run impl_1}]} {
    set opened 1
} elseif {![catch {open_run synth_1}]} {
    set opened 1
    puts "WARN: opened synth_1 for reports (impl incomplete)"
}
if {$opened} {
    report_utilization -file $util_rpt
    report_timing_summary -file $tim_rpt
    catch {report_route_status -file $route_rpt}
} else {
    puts "ERROR: could not open impl_1 or synth_1 for reports"
    exit 1
}
puts "DSS_DONE prefix=$report_prefix impl_status=$impl_status"
exit 0
