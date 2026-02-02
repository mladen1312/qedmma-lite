# =============================================================================
# QEDMMA-Lite v3.0 - Zero-DSP Correlator Vitis HLS Script
# =============================================================================
# Copyright (C) 2026 Dr. Mladen Mešter / Nexellum
# License: AGPL-3.0-or-later
#
# Usage:
#   vitis_hls -f run_hls.tcl
# =============================================================================

# Configuration
set PROJECT_NAME "zero_dsp_hls"
set SOLUTION_NAME "solution1"
set TOP_FUNCTION "zero_dsp_correlator_top"

# Target device
set PART "xczu7ev-ffvc1156-2-e"
set CLOCK_PERIOD "5"  ;# 200 MHz

# Create project
open_project -reset $PROJECT_NAME

# Set top function
set_top $TOP_FUNCTION

# Add source files
add_files "../hls/zero_dsp_correlator.cpp"
add_files -tb "../hls/zero_dsp_correlator_tb.cpp"

# Create solution
open_solution -reset $SOLUTION_NAME

# Set target device and clock
set_part $PART
create_clock -period $CLOCK_PERIOD -name default

# Configure solution
config_compile -pipeline_loops 0
config_schedule -effort high

# Run C simulation
puts "Running C Simulation..."
csim_design

# Run synthesis
puts "Running C Synthesis..."
csynth_design

# Run co-simulation
puts "Running Co-simulation..."
cosim_design -rtl vhdl

# Export RTL
puts "Exporting RTL..."
export_design -format ip_catalog -description "Zero-DSP Correlator IP" -vendor "Nexellum" -library "qedmma" -version "3.0"

# Generate reports
puts ""
puts "============================================================"
puts "HLS SYNTHESIS COMPLETE"
puts "============================================================"

# Close project
close_project

puts "IP exported to: $PROJECT_NAME/$SOLUTION_NAME/impl/ip"
