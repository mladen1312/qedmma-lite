# =============================================================================
# Zero-DSP Correlator - Synthesis Constraints
# =============================================================================
# Author:  Dr. Mladen Mešter / Nexellum d.o.o.
# License: AGPL-3.0-or-later
# Contact: mladen@nexellum.com | +385 99 737 5100
# =============================================================================
# [REQ-ZD-001] CRITICAL: Prevent DSP48 inference
# [REQ-ZD-003] Target: 100 MHz operation (10 ns period)
# =============================================================================

# -----------------------------------------------------------------------------
# CLOCK DEFINITION
# -----------------------------------------------------------------------------
# Primary clock - 100 MHz target for [REQ-ZD-003]
create_clock -period 10.000 -name clk [get_ports clk]

# Clock uncertainty (jitter + skew estimate)
set_clock_uncertainty 0.500 [get_clocks clk]

# -----------------------------------------------------------------------------
# ZERO-DSP CONSTRAINTS [REQ-ZD-001]
# Prevent ANY DSP48 inference in the entire module hierarchy
# -----------------------------------------------------------------------------

# Method 1: Global DSP disable (applied at synthesis level)
# set_property DSP_STYLE NONE [get_cells -hier -filter {IS_SEQUENTIAL==FALSE}]

# Method 2: Mark all multiplication operations to use LUTs
set_property USE_DSP NO [get_cells -hier -filter {NAME =~ *products*}]
set_property USE_DSP NO [get_cells -hier -filter {NAME =~ *mult*}]
set_property USE_DSP NO [get_cells -hier -filter {NAME =~ *accum*}]

# Method 3: Force fabric implementation for all arithmetic
# Applied to the entire correlator module
set_property USE_DSP NO [get_cells -hier -filter {PARENT =~ *zero_dsp_correlator*}]

# -----------------------------------------------------------------------------
# PIPELINE CONSTRAINTS
# Allow retiming for timing closure
# -----------------------------------------------------------------------------
set_property RETIMING TRUE [get_cells -hier -filter {NAME =~ *accum_stage*}]
set_property RETIMING TRUE [get_cells -hier -filter {NAME =~ *group_sums*}]

# -----------------------------------------------------------------------------
# INPUT/OUTPUT CONSTRAINTS
# Assume registered I/O at boundaries
# -----------------------------------------------------------------------------

# Input delay (relative to clock)
set_input_delay -clock clk -max 2.0 [get_ports s_axis_*]
set_input_delay -clock clk -min 0.5 [get_ports s_axis_*]
set_input_delay -clock clk -max 2.0 [get_ports cfg_*]
set_input_delay -clock clk -min 0.5 [get_ports cfg_*]
set_input_delay -clock clk -max 2.0 [get_ports m_axis_tready]

# Output delay (relative to clock)  
set_output_delay -clock clk -max 2.0 [get_ports m_axis_*]
set_output_delay -clock clk -min 0.5 [get_ports m_axis_*]
set_output_delay -clock clk -max 2.0 [get_ports status_*]

# -----------------------------------------------------------------------------
# RESET CONSTRAINTS
# Async reset - treat as false path for timing, ensure clean release
# -----------------------------------------------------------------------------
set_false_path -from [get_ports rst_n]

# -----------------------------------------------------------------------------
# CONFIGURATION PATHS
# Coefficients are loaded when disabled - multicycle path
# -----------------------------------------------------------------------------
set_multicycle_path 4 -setup -from [get_ports cfg_coefficients*]
set_multicycle_path 3 -hold -from [get_ports cfg_coefficients*]

# Mode and enable are quasi-static
set_false_path -from [get_ports cfg_mode*]

# -----------------------------------------------------------------------------
# CDC CONSTRAINTS (if crossing clock domains)
# Not needed for single-clock design, but template provided
# -----------------------------------------------------------------------------
# set_max_delay -datapath_only 5.0 -from [get_cells cdc_src_reg*] -to [get_cells cdc_dst_reg*]

# -----------------------------------------------------------------------------
# AREA CONSTRAINTS (optional - for guided placement)
# -----------------------------------------------------------------------------
# Uncomment to constrain correlator to specific SLR or region
# create_pblock pblock_correlator
# resize_pblock pblock_correlator -add {SLICE_X0Y0:SLICE_X50Y100}
# add_cells_to_pblock pblock_correlator [get_cells -hier -filter {NAME =~ *zero_dsp_correlator*}]

# -----------------------------------------------------------------------------
# DEBUG CONSTRAINTS
# Mark nets for ILA insertion if needed
# -----------------------------------------------------------------------------
# set_property MARK_DEBUG TRUE [get_nets -hier -filter {NAME =~ *products*}]
# set_property MARK_DEBUG TRUE [get_nets -hier -filter {NAME =~ *accum_stage*}]

# -----------------------------------------------------------------------------
# PHYSICAL CONSTRAINTS FOR CRITICAL PATHS
# -----------------------------------------------------------------------------
# Force carry chains to stay together (for adder tree)
set_property KEEP_HIERARCHY TRUE [get_cells -hier -filter {NAME =~ *group_sums*}]

# -----------------------------------------------------------------------------
# REPORTING DIRECTIVES
# -----------------------------------------------------------------------------
# Generate detailed reports during synthesis
set_property SEVERITY {Warning} [get_drc_checks TIMING-*]

# Report any DSP inference as error
set_property SEVERITY {Error} [get_drc_checks SYNTH-6]
