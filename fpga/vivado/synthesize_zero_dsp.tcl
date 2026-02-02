# =============================================================================
# QEDMMA-Lite v3.0 - Zero-DSP Correlator Vivado Synthesis Script
# =============================================================================
# Copyright (C) 2026 Dr. Mladen Mešter / Nexellum
# License: AGPL-3.0-or-later
#
# Usage:
#   vivado -mode batch -source synthesize_zero_dsp.tcl
#   
# Or interactive:
#   vivado -mode tcl
#   source synthesize_zero_dsp.tcl
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
set PROJECT_NAME "qedmma_zero_dsp"
set PROJECT_DIR "./vivado_project"
set RTL_DIR "../rtl"
set CONSTRAINT_DIR "./constraints"

# Target device (modify for your board)
# Zynq-7020 (Red Pitaya, Cora Z7)
# set PART "xc7z020clg400-1"

# Zynq UltraScale+ (ZCU102, ZCU104)
set PART "xczu7ev-ffvc1156-2-e"

# Design parameters
set N_SAMPLES 1024
set CORR_WIDTH 12

# Clock constraint (MHz)
set CLK_PERIOD_NS 5.0  ;# 200 MHz target

# -----------------------------------------------------------------------------
# Create Project
# -----------------------------------------------------------------------------
puts "============================================================"
puts "QEDMMA Zero-DSP Correlator Synthesis"
puts "============================================================"
puts "Part: $PART"
puts "N_SAMPLES: $N_SAMPLES"
puts "Target Clock: [expr {1000.0/$CLK_PERIOD_NS}] MHz"
puts "============================================================"

# Create project
create_project $PROJECT_NAME $PROJECT_DIR -part $PART -force

# Set project properties
set_property target_language VHDL [current_project]
set_property default_lib work [current_project]

# -----------------------------------------------------------------------------
# Add RTL Sources
# -----------------------------------------------------------------------------
puts "Adding RTL sources..."

# Create RTL directory if it doesn't exist
file mkdir $RTL_DIR

# Add Zero-DSP correlator VHDL
add_files -norecurse [list \
    "$RTL_DIR/zero_dsp_correlator.vhd" \
]

# Update compile order
update_compile_order -fileset sources_1

# -----------------------------------------------------------------------------
# Create Constraints
# -----------------------------------------------------------------------------
puts "Creating timing constraints..."

file mkdir $CONSTRAINT_DIR

set XDC_FILE "$CONSTRAINT_DIR/timing.xdc"
set xdc_fp [open $XDC_FILE w]

puts $xdc_fp "# ==================================================================="
puts $xdc_fp "# QEDMMA Zero-DSP Correlator Timing Constraints"
puts $xdc_fp "# ==================================================================="
puts $xdc_fp ""
puts $xdc_fp "# Clock definition"
puts $xdc_fp "create_clock -period $CLK_PERIOD_NS -name clk \[get_ports clk\]"
puts $xdc_fp ""
puts $xdc_fp "# Input delays (adjust for your system)"
puts $xdc_fp "set_input_delay -clock clk -max 2.0 \[get_ports {x_ref* y_in y_valid rst}\]"
puts $xdc_fp "set_input_delay -clock clk -min 0.5 \[get_ports {x_ref* y_in y_valid rst}\]"
puts $xdc_fp ""
puts $xdc_fp "# Output delays"
puts $xdc_fp "set_output_delay -clock clk -max 2.0 \[get_ports {corr_out* corr_valid}\]"
puts $xdc_fp "set_output_delay -clock clk -min 0.5 \[get_ports {corr_out* corr_valid}\]"
puts $xdc_fp ""
puts $xdc_fp "# False paths for async reset"
puts $xdc_fp "set_false_path -from \[get_ports rst\]"

close $xdc_fp

add_files -fileset constrs_1 -norecurse $XDC_FILE

# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------
puts "Running Synthesis..."

# Set synthesis options
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]

# Add generic parameters
set_property generic "N_SAMPLES=$N_SAMPLES CORR_WIDTH=$CORR_WIDTH" [get_filesets sources_1]

# Launch synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis status
if {[get_property STATUS [get_runs synth_1]] != "synth_design Complete!"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

# Open synthesized design
open_run synth_1

# -----------------------------------------------------------------------------
# Resource Report
# -----------------------------------------------------------------------------
puts ""
puts "============================================================"
puts "SYNTHESIS RESOURCE REPORT"
puts "============================================================"

report_utilization -file "$PROJECT_DIR/utilization_synth.rpt"

# Print key metrics
set util_report [report_utilization -return_string]
puts $util_report

# Check DSP usage (should be 0!)
set dsp_line [regexp -inline {DSP\s+\|\s+(\d+)} $util_report]
if {[llength $dsp_line] > 1} {
    set dsp_used [lindex $dsp_line 1]
    if {$dsp_used == 0} {
        puts "\n✅ DSP USAGE: 0 (Zero-DSP verified!)"
    } else {
        puts "\n❌ WARNING: DSP blocks used: $dsp_used"
    }
}

# -----------------------------------------------------------------------------
# Implementation (Optional - uncomment to run)
# -----------------------------------------------------------------------------
puts ""
puts "Running Implementation..."

launch_runs impl_1 -jobs 4
wait_on_run impl_1

if {[get_property STATUS [get_runs impl_1]] != "route_design Complete!"} {
    puts "ERROR: Implementation failed!"
    exit 1
}

open_run impl_1

# Timing report
report_timing_summary -file "$PROJECT_DIR/timing_impl.rpt"

# Final utilization
report_utilization -file "$PROJECT_DIR/utilization_impl.rpt"

# Power report
report_power -file "$PROJECT_DIR/power.rpt"

# -----------------------------------------------------------------------------
# Generate Bitstream (Optional - uncomment if needed)
# -----------------------------------------------------------------------------
# puts "Generating Bitstream..."
# launch_runs impl_1 -to_step write_bitstream -jobs 4
# wait_on_run impl_1

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
puts ""
puts "============================================================"
puts "SYNTHESIS COMPLETE"
puts "============================================================"
puts "Reports saved to: $PROJECT_DIR/"
puts "  - utilization_synth.rpt"
puts "  - utilization_impl.rpt"  
puts "  - timing_impl.rpt"
puts "  - power.rpt"
puts ""
puts "To open project in GUI:"
puts "  vivado $PROJECT_DIR/$PROJECT_NAME.xpr"
puts "============================================================"

# Close project
close_project
