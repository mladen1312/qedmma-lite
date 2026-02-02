# =============================================================================
# Zero-DSP Correlator - Vivado Synthesis Script
# =============================================================================
# Author:  Dr. Mladen Mešter / Nexellum d.o.o.
# License: AGPL-3.0-or-later
# Contact: mladen@nexellum.com | +385 99 737 5100
# =============================================================================
# Usage:
#   vivado -mode batch -source synth_zero_dsp.tcl
#   vivado -mode batch -source synth_zero_dsp.tcl -tclargs -part xc7z020clg400-1
# =============================================================================

# -----------------------------------------------------------------------------
# Parse Command Line Arguments
# -----------------------------------------------------------------------------
set part_name "xczu48dr-ffvg1517-2-e"  ;# Default: RFSoC 4x2
set top_module "zero_dsp_correlator"
set output_dir "./output"
set project_name "zero_dsp_project"

# Process arguments
for {set i 0} {$i < $argc} {incr i} {
    set arg [lindex $argv $i]
    switch -exact -- $arg {
        "-part" {
            incr i
            set part_name [lindex $argv $i]
        }
        "-top" {
            incr i
            set top_module [lindex $argv $i]
        }
        "-output" {
            incr i
            set output_dir [lindex $argv $i]
        }
    }
}

puts "============================================================"
puts " Zero-DSP Correlator Synthesis"
puts " Part:   $part_name"
puts " Top:    $top_module"
puts " Output: $output_dir"
puts "============================================================"

# -----------------------------------------------------------------------------
# Create Project
# -----------------------------------------------------------------------------
create_project -force $project_name $output_dir/$project_name -part $part_name

# Set project properties
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# -----------------------------------------------------------------------------
# Add Source Files
# -----------------------------------------------------------------------------
add_files -norecurse {
    ../rtl/zero_dsp_correlator.sv
}

# Set top module
set_property top $top_module [current_fileset]

# -----------------------------------------------------------------------------
# Add Constraints
# -----------------------------------------------------------------------------
add_files -fileset constrs_1 -norecurse ../constraints/zero_dsp_synth.xdc
add_files -fileset constrs_1 -norecurse ../constraints/zero_dsp_timing.xdc

# -----------------------------------------------------------------------------
# Synthesis Settings
# [REQ-ZD-001] CRITICAL: Force zero DSP inference
# -----------------------------------------------------------------------------
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]

# Key synthesis options for Zero-DSP
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {
    -mode out_of_context
    -no_lc
    -keep_equivalent_registers
    -resource_sharing off
} -objects [get_runs synth_1]

# Disable DSP inference globally (belt and suspenders with RTL attributes)
set_property STEPS.SYNTH_DESIGN.ARGS.MAX_DSP 0 [get_runs synth_1]

# -----------------------------------------------------------------------------
# Run Synthesis
# -----------------------------------------------------------------------------
puts "Starting synthesis..."
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1

# Check synthesis status
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

# Open synthesized design
open_run synth_1 -name synth_1

# -----------------------------------------------------------------------------
# Verify Zero-DSP Usage [REQ-ZD-001]
# -----------------------------------------------------------------------------
puts ""
puts "============================================================"
puts " RESOURCE UTILIZATION REPORT"
puts "============================================================"

# Get DSP utilization
set dsp_used [llength [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ DSP.*}]]
puts "DSP48 Slices Used: $dsp_used"

if {$dsp_used > 0} {
    puts ""
    puts "!!! WARNING: DSP48 SLICES DETECTED !!!"
    puts "This violates [REQ-ZD-001] Zero DSP requirement."
    puts ""
    puts "DSP cells found:"
    foreach cell [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ DSP.*}] {
        puts "  $cell"
    }
    puts ""
    puts "Check RTL for missing (* use_dsp = \"no\" *) attributes"
    # Don't exit - allow inspection
}

# Full utilization report
report_utilization -file $output_dir/utilization_synth.rpt
report_utilization -hierarchical -file $output_dir/utilization_hier.rpt

# Print summary
set lut_used [get_property USED [get_report_config -of [get_reports "SLICE LUTs"]]] 
set ff_used [get_property USED [get_report_config -of [get_reports "Slice Registers"]]]

puts ""
puts "Resource Summary:"
puts "  LUTs:     $lut_used"
puts "  FFs:      $ff_used"
puts "  DSPs:     $dsp_used"
puts "  BRAMs:    [llength [get_cells -hierarchical -filter {PRIMITIVE_TYPE =~ BMEM.*}]]"

# -----------------------------------------------------------------------------
# Timing Analysis
# -----------------------------------------------------------------------------
puts ""
puts "============================================================"
puts " TIMING ANALYSIS"
puts "============================================================"

# Create clock for out-of-context synthesis
create_clock -period 10.000 -name clk [get_ports clk]

report_timing_summary -file $output_dir/timing_synth.rpt

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1]]
puts "Worst Negative Slack (WNS): $wns ns"

if {$wns < 0} {
    puts "WARNING: Timing not met. WNS = $wns ns"
    puts "Consider adding pipeline stages or reducing CORR_LENGTH"
}

# -----------------------------------------------------------------------------
# Generate Reports
# -----------------------------------------------------------------------------
report_power -file $output_dir/power_synth.rpt
report_methodology -file $output_dir/methodology.rpt

# -----------------------------------------------------------------------------
# Write Checkpoint
# -----------------------------------------------------------------------------
write_checkpoint -force $output_dir/${top_module}_synth.dcp

puts ""
puts "============================================================"
puts " SYNTHESIS COMPLETE"
puts "============================================================"
puts "Output files in: $output_dir"
puts "  - ${top_module}_synth.dcp (checkpoint)"
puts "  - utilization_synth.rpt"
puts "  - timing_synth.rpt"
puts "  - power_synth.rpt"
puts "============================================================"

# Exit cleanly
close_project
exit 0
