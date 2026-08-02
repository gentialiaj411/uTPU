## Vivado batch synth for packed-DSP A/B comparison (baseline top vs top_packed).
## Usage:
##   vivado -mode batch -source scripts/synth_packed_dsp.tcl \
##     -tclargs do_program 0 top_name top_packed report_prefix packed_array_8x8_int8 \
##     ARRAY_SIZE 8 COMPUTE_DATA_WIDTH 8 ACCUMULATOR_DATA_WIDTH 32

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
set proj_dir   [file normalize [maybe_get $arg_opts proj_dir [file join $repo_root build vivado_packed_dsp]]]
set proj_name  [maybe_get $arg_opts proj_name uTPU_packed_dsp]
set part_name  xc7a100tcsg324-1
set top_name   [maybe_get $arg_opts top_name top_packed]
set generic_opt [maybe_get $arg_opts generic ""]
if {$generic_opt eq ""} {
    set generic_pairs {}
    foreach key {PROG_DEPTH COMPUTE_DATA_WIDTH ACCUMULATOR_DATA_WIDTH ARRAY_SIZE BUFFER_SIZE EXT_ADDR_EN MAX_BATCH_COUNT QUANTIZER_LANES QUANTIZER_SIZE} {
        if {[dict exists $arg_opts $key]} {
            lappend generic_pairs "${key}=[dict get $arg_opts $key]"
        }
    }
    set generic_opt [join $generic_pairs " "]
}
regsub -all {,} $generic_opt { } generic_opt
regsub -all {:} $generic_opt {=} generic_opt
set generic_opt [string trim $generic_opt]
set report_prefix [string trim [maybe_get $arg_opts report_prefix ""]]

create_project -force $proj_name $proj_dir -part $part_name

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
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/top"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/PEArray"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/UART"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/fifo"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/memory"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/quantizer"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/LeakyReLU"]]
set rtl_files [concat $rtl_files [add_whitelist_dir $repo_root "rtl/unified_buffer"]]
set gen_params [file join $repo_root generated generated_params.sv]
if {[file exists $gen_params]} {
    lappend rtl_files $gen_params
}

if {[llength $rtl_files] == 0} {
    error "No RTL source files found in whitelist"
}
read_verilog -sv $rtl_files

set xdc_file [file join $repo_root arty_a7_revE_usb_uart.xdc]
if {[file exists $xdc_file]} {
    read_xdc $xdc_file
}

set_property top $top_name [current_fileset]
if {$generic_opt ne ""} {
    set_property generic $generic_opt [current_fileset]
    puts "Applied generics: $generic_opt"
}

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {$report_prefix ne ""} {
    set reports_dir [file join $repo_root build reports]
    file mkdir $reports_dir
    set timing_rpt      [file join $reports_dir "${report_prefix}_timing_summary.rpt"]
    set utilization_rpt [file join $reports_dir "${report_prefix}_utilization.rpt"]
    set synth_util_rpt  [file join $reports_dir "${report_prefix}_synth_utilization.rpt"]
    set route_rpt       [file join $reports_dir "${report_prefix}_route_status.rpt"]
    set dsp_rpt         [file join $reports_dir "${report_prefix}_dsp_utilization.rpt"]
    set dcp_rpt         [file join $reports_dir "${report_prefix}_post_route.dcp"]

    open_run synth_1
    report_utilization -file $synth_util_rpt
    if {[catch {report_dsp_utilization -file $dsp_rpt} dsp_report_err]} {
        set fh [open $dsp_rpt w]
        puts $fh "report_dsp_utilization unavailable: $dsp_report_err"
        close $fh
    }
}

launch_runs impl_1 -to_step write_bitstream -jobs 4
set impl_wait_rc [catch {wait_on_run impl_1} impl_wait_err]
set impl_status [get_property STATUS [get_runs impl_1]]
set impl_progress [get_property PROGRESS [get_runs impl_1]]
set routed_ok 0
if {!$impl_wait_rc} {
    if {![catch {open_run impl_1}]} {
        set routed_ok 1
    }
}

if {$report_prefix ne ""} {
    set reports_dir [file join $repo_root build reports]
    set timing_rpt      [file join $reports_dir "${report_prefix}_timing_summary.rpt"]
    set utilization_rpt [file join $reports_dir "${report_prefix}_utilization.rpt"]
    set route_rpt       [file join $reports_dir "${report_prefix}_route_status.rpt"]
    set dcp_rpt         [file join $reports_dir "${report_prefix}_post_route.dcp"]

    set fh [open $route_rpt w]
    puts $fh "impl_status: $impl_status"
    puts $fh "impl_progress: $impl_progress"
    if {$impl_wait_rc} {
        puts $fh "impl_wait_error: $impl_wait_err"
    }
    close $fh

    if {$routed_ok} {
        report_timing_summary -file $timing_rpt
        report_utilization -file $utilization_rpt
        report_route_status -append -file $route_rpt
        write_checkpoint -force $dcp_rpt

        set bitfile [file join $proj_dir $proj_name.runs impl_1 ${top_name}.bit]
        if {[file exists $bitfile]} {
            set bit_out [file join $reports_dir "${report_prefix}.bit"]
            file copy -force $bitfile $bit_out
            puts "Bitstream copied to $bit_out"
        }
    } else {
        file copy -force $synth_util_rpt $utilization_rpt
        puts "Implementation did not produce a routed design; copied synth utilization to $utilization_rpt"
    }
    puts "Reports written with prefix $report_prefix"
}

set do_program [expr {[dict exists $arg_opts do_program] ? ([dict get $arg_opts do_program] != 0) : 0}]
if {$do_program} {
    puts "Hardware programming not enabled for packed-DSP batch flow by default."
} else {
    puts "Skipping hardware programming (do_program=0)."
}
