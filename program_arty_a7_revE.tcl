## Vivado Tcl flow: build and program Arty A7-100T (Rev E) via USB-UART design.
## Usage (vivado-gui):
##   vivado-gui -source program_arty_a7_revE.tcl

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
set arg_opts   [parse_kv_args $argv]
set proj_dir   [file normalize [maybe_get $arg_opts proj_dir [file join $script_dir build vivado_arty_a7]]]
set proj_name  [maybe_get $arg_opts proj_name uTPU_arty_a7]
set part_name  xc7a100tcsg324-1
set top_name   top
set clock_period [string trim [maybe_get $arg_opts clock_period ""]]
set generic_opt [maybe_get $arg_opts generic ""]
if {$generic_opt eq ""} {
    set generic_pairs {}
    foreach key {PROG_DEPTH COMPUTE_DATA_WIDTH ACCUMULATOR_DATA_WIDTH ARRAY_SIZE BUFFER_SIZE EXT_ADDR_EN MAX_BATCH_COUNT QUANTIZER_LANES RELU_LANES QUANTIZER_PIPE_DEPTH WEIGHT_OVERLAP_EN} {
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
if {$clock_period ne ""} {
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
    puts "Applied clock period override: $clock_period ns"
} else {
    read_xdc $xdc_file
}

# Set top
set_property top $top_name [current_fileset]
if {$generic_opt ne ""} {
    set_property generic $generic_opt [current_fileset]
    puts "Applied generics: $generic_opt"
}

# Optional run tuning knobs for timing sweeps.
set synth_strategy [string trim [maybe_get $arg_opts synth_strategy ""]]
set impl_strategy  [string trim [maybe_get $arg_opts impl_strategy ""]]
set retiming_opt   [string trim [maybe_get $arg_opts retiming ""]]
set post_place_phys_opt [string trim [maybe_get $arg_opts post_place_phys_opt ""]]
set post_route_phys_opt [string trim [maybe_get $arg_opts post_route_phys_opt ""]]
if {$synth_strategy ne ""} {
    set_property strategy $synth_strategy [get_runs synth_1]
    puts "Applied synth strategy: $synth_strategy"
}
if {$impl_strategy ne ""} {
    set_property strategy $impl_strategy [get_runs impl_1]
    puts "Applied impl strategy: $impl_strategy"
}
if {$retiming_opt ne "" && $retiming_opt ne "0"} {
    set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
    puts "Enabled synth retiming"
}
if {$post_place_phys_opt ne "" && $post_place_phys_opt ne "0"} {
    set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    puts "Enabled post-place phys_opt_design"
}
if {$post_route_phys_opt ne "" && $post_route_phys_opt ne "0"} {
    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    puts "Enabled post-route phys_opt_design"
}

# Synthesize, implement, bitstream
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
open_run impl_1

if {$report_prefix ne ""} {
    set reports_dir [file join $script_dir build reports]
    file mkdir $reports_dir
    set timing_rpt     [file join $reports_dir "${report_prefix}_timing_summary.rpt"]
    set utilization_rpt [file join $reports_dir "${report_prefix}_utilization.rpt"]
    set route_rpt      [file join $reports_dir "${report_prefix}_route_status.rpt"]
    set dcp_rpt        [file join $reports_dir "${report_prefix}_post_route.dcp"]

    report_timing_summary -file $timing_rpt
    report_utilization -file $utilization_rpt
    report_route_status -file $route_rpt
    write_checkpoint -force $dcp_rpt

    set bitfile [file join $proj_dir $proj_name.runs impl_1 ${top_name}.bit]
    if {[file exists $bitfile]} {
        set bit_out [file join $reports_dir "${report_prefix}.bit"]
        file copy -force $bitfile $bit_out
        puts "Bitstream copied to $bit_out"
    }
    puts "Reports written with prefix $report_prefix"
}

# Program device
set do_program [expr {[dict exists $arg_opts do_program] ? ([dict get $arg_opts do_program] != 0) : 0}]
if {$do_program} {
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
} else {
    puts "Skipping hardware programming step (set do_program=1 to enable)."
}
