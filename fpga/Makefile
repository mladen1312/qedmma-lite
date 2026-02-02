# =============================================================================
# QEDMMA-Lite v3.0 - FPGA Build Makefile
# =============================================================================
# Copyright (C) 2026 Dr. Mladen Mešter / Nexellum
# License: AGPL-3.0-or-later
#
# Targets:
#   make sim        - Run VHDL simulation (GHDL)
#   make synth      - Run Vivado synthesis
#   make hls        - Run Vitis HLS
#   make all        - Run all targets
#   make clean      - Clean build artifacts
# =============================================================================

# Tool paths (adjust for your system)
GHDL ?= ghdl
VIVADO ?= vivado
VITIS_HLS ?= vitis_hls

# Configuration
TOP_MODULE = zero_dsp_correlator
N_SAMPLES ?= 64
CORR_WIDTH ?= 8

# Directories
RTL_DIR = rtl
HLS_DIR = hls
VIVADO_DIR = vivado
BUILD_DIR = build
WORK_DIR = $(BUILD_DIR)/work

# Source files
VHDL_SRCS = $(RTL_DIR)/$(TOP_MODULE).vhd
VHDL_TB = $(RTL_DIR)/$(TOP_MODULE)_tb.vhd
HLS_SRCS = $(HLS_DIR)/$(TOP_MODULE).cpp

# GHDL flags
GHDL_FLAGS = --std=08 --workdir=$(WORK_DIR)
GHDL_ELAB_FLAGS = $(GHDL_FLAGS) -Wl,-lm
GHDL_RUN_FLAGS = --vcd=$(BUILD_DIR)/$(TOP_MODULE).vcd --stop-time=10us

# =============================================================================
# Phony targets
# =============================================================================
.PHONY: all sim synth hls clean help

all: sim

help:
	@echo "QEDMMA Zero-DSP Correlator FPGA Build System"
	@echo "============================================="
	@echo ""
	@echo "Targets:"
	@echo "  make sim        - Run GHDL simulation"
	@echo "  make synth      - Run Vivado synthesis"
	@echo "  make hls        - Run Vitis HLS"
	@echo "  make clean      - Clean build artifacts"
	@echo ""
	@echo "Configuration:"
	@echo "  N_SAMPLES=$(N_SAMPLES)"
	@echo "  CORR_WIDTH=$(CORR_WIDTH)"

# =============================================================================
# Simulation (GHDL)
# =============================================================================
sim: $(BUILD_DIR)/$(TOP_MODULE)_tb
	@echo "Running simulation..."
	cd $(BUILD_DIR) && ./$(TOP_MODULE)_tb $(GHDL_RUN_FLAGS)
	@echo ""
	@echo "Waveform saved to: $(BUILD_DIR)/$(TOP_MODULE).vcd"
	@echo "View with: gtkwave $(BUILD_DIR)/$(TOP_MODULE).vcd"

$(BUILD_DIR)/$(TOP_MODULE)_tb: $(VHDL_SRCS) $(VHDL_TB) | $(WORK_DIR)
	@echo "Analyzing VHDL sources..."
	$(GHDL) -a $(GHDL_FLAGS) $(VHDL_SRCS)
	$(GHDL) -a $(GHDL_FLAGS) $(VHDL_TB)
	@echo "Elaborating testbench..."
	$(GHDL) -e $(GHDL_ELAB_FLAGS) -o $@ $(TOP_MODULE)_tb

$(WORK_DIR):
	mkdir -p $(WORK_DIR)

# =============================================================================
# Vivado Synthesis
# =============================================================================
synth: $(BUILD_DIR)/vivado_done

$(BUILD_DIR)/vivado_done: $(VHDL_SRCS) $(VIVADO_DIR)/synthesize_zero_dsp.tcl | $(BUILD_DIR)
	@echo "Running Vivado synthesis..."
	cd $(VIVADO_DIR) && $(VIVADO) -mode batch -source synthesize_zero_dsp.tcl
	touch $@

# =============================================================================
# Vitis HLS
# =============================================================================
hls: $(BUILD_DIR)/hls_done

$(BUILD_DIR)/hls_done: $(HLS_SRCS) $(HLS_DIR)/run_hls.tcl | $(BUILD_DIR)
	@echo "Running Vitis HLS..."
	cd $(HLS_DIR) && $(VITIS_HLS) -f run_hls.tcl
	touch $@

# =============================================================================
# Build directory
# =============================================================================
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# =============================================================================
# Clean
# =============================================================================
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(WORK_DIR)
	rm -rf $(VIVADO_DIR)/vivado_project
	rm -rf $(HLS_DIR)/zero_dsp_hls
	rm -f *.jou *.log
	@echo "Clean complete."

# =============================================================================
# CI/CD targets
# =============================================================================
.PHONY: ci lint

ci: sim
	@echo "CI build complete."

lint:
	@echo "Running VHDL lint..."
	$(GHDL) -s $(GHDL_FLAGS) $(VHDL_SRCS)
	@echo "Lint passed."
