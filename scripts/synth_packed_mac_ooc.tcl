## Out-of-context synth for packed MAC DSP measurement (no IOB/place).
## Usage:
##   vivado -mode batch -source scripts/synth_packed_mac_ooc.tcl \
##     -tclargs top_name pe_array_packed ARRAY_SIZE 16 report_prefix packed_mac_ooc_16x16

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
set top_name   [maybe_get $arg_opts top_name pe_array_packed]
set array_size [maybe_get $arg_opts ARRAY_SIZE 16]
set cdw        [maybe_get $arg_opts COMPUTE_DATA_WIDTH 8]
set adw        [maybe_get $arg_opts ACCUMULATOR_DATA_WIDTH 32]
set pack_shift [maybe_get $arg_opts PACK_SHIFT ""]
set report_prefix [string trim [maybe_get $arg_opts report_prefix "packed_mac_ooc"]]
set reports_dir [file join $repo_root build reports]
file mkdir $reports_dir

create_project -in_memory -part $part_name
set rtl_files {}
lappend rtl_files [file join $repo_root rtl PEArray pe_packed_pair.sv]
if {$top_name eq "pe_array_packed"} {
    lappend rtl_files [file join $repo_root rtl PEArray pe_array_packed.sv]
} elseif {$top_name eq "pe_packed_skewed"} {
    # pe_packed_skewed lives in pe_array_packed.sv (INT8 skewed extract).
    # For PACK_SHIFT/CDW sweeps prefer top_name=pe_packed_pair.
    lappend rtl_files [file join $repo_root rtl PEArray pe_array_packed.sv]
}

add_files -norecurse $rtl_files
set_property top $top_name [current_fileset]

set generic_list {}
if {$top_name eq "pe_array_packed"} {
    lappend generic_list "ARRAY_SIZE=$array_size"
    lappend generic_list "COMPUTE_DATA_WIDTH=$cdw"
    lappend generic_list "ACCUMULATOR_DATA_WIDTH=$adw"
} elseif {$top_name eq "pe_packed_skewed" || $top_name eq "pe_packed_pair"} {
    lappend generic_list "COMPUTE_DATA_WIDTH=$cdw"
    lappend generic_list "ACCUMULATOR_DATA_WIDTH=$adw"
    if {$pack_shift ne ""} {
        lappend generic_list "PACK_SHIFT=$pack_shift"
    }
}
if {[llength $generic_list] > 0} {
    set_property generic $generic_list [current_fileset]
    puts "Applied generics: $generic_list"
}

# OOC: measure DSP inference without package pins / place.
synth_design -mode out_of_context -top $top_name -part $part_name

set util_rpt   [file join $reports_dir "${report_prefix}_utilization.rpt"]
set dsp_rpt    [file join $reports_dir "${report_prefix}_dsp.rpt"]
set cells_rpt  [file join $reports_dir "${report_prefix}_dsp_cells.rpt"]
report_utilization -file $util_rpt
report_utilization -hierarchical -file [file join $reports_dir "${report_prefix}_hier_utilization.rpt"]
# Cell-level DSP inventory for packing audit.
# Vivado 2025.2 OOC designs often leave REF_NAME=DSP48E1 while PRIMITIVE_TYPE
# matching is empty; prefer REF_NAME then fall back to utilization parse.
set dsp_cells [get_cells -quiet -hierarchical -filter {REF_NAME =~ DSP48*}]
if {[llength $dsp_cells] == 0} {
    set dsp_cells [get_cells -quiet -hierarchical -filter {PRIMITIVE_TYPE =~ DSP.*}]
}
if {[llength $dsp_cells] > 0} {
    report_property $dsp_cells -file $cells_rpt
    set dsp_count [llength $dsp_cells]
} else {
    set dsp_count 0
    set fh [open $cells_rpt w]
    puts $fh "No DSP cells found after OOC synth via get_cells; see utilization.rpt."
    close $fh
    # Fallback: parse "DSP48E1 only" row from utilization.
    set util_text [read [open $util_rpt r]]
    if {[regexp {DSP48E1 only\s*\|\s*(\d+)} $util_text -> n]} {
        set dsp_count $n
    } elseif {[regexp {\|\s*DSPs\s*\|\s*(\d+)} $util_text -> n]} {
        set dsp_count $n
    }
}
set fh [open $dsp_rpt w]
puts $fh "top_name: $top_name"
puts $fh "ARRAY_SIZE: $array_size"
puts $fh "dsp_cell_count: $dsp_count"
puts $fh "hypothesis_2mac_per_dsp_for_array: [expr {($array_size * $array_size) / 2}]"
puts $fh "mac_count: [expr {$array_size * $array_size}]"
close $fh
puts "OOC synth complete: dsp_cell_count=$dsp_count report_prefix=$report_prefix"
