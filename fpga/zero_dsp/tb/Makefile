# =============================================================================
# Zero-DSP Correlator - Cocotb Simulation Makefile
# =============================================================================
# Author:  Dr. Mladen Mešter / Nexellum d.o.o.
# License: AGPL-3.0-or-later
# Usage:   make sim          - Run all tests with Verilator
#          make sim SIM=icarus - Run with Icarus Verilog
#          make waves        - Open waveforms
#          make lint         - Run Verilator lint
#          make clean        - Clean build artifacts
# =============================================================================

# Simulator selection (verilator, icarus, questa, vcs)
SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# Design files
VERILOG_SOURCES = $(PWD)/../rtl/zero_dsp_correlator.sv

# Top-level module
TOPLEVEL = zero_dsp_correlator

# Python test module
MODULE = test_zero_dsp_correlator

# Cocotb configuration
export COCOTB_RESOLVE_X = ZEROS
export PYTHONDONTWRITEBYTECODE = 1

# Verilator-specific options
ifeq ($(SIM), verilator)
    EXTRA_ARGS += --trace --trace-structs
    EXTRA_ARGS += -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC
    EXTRA_ARGS += -Wno-UNSIGNED -Wno-CASEINCOMPLETE
    COMPILE_ARGS += -CFLAGS "-std=c++17"
endif

# Icarus-specific options
ifeq ($(SIM), icarus)
    COMPILE_ARGS += -g2012
endif

# QuestaSim/ModelSim options
ifeq ($(SIM), questa)
    COMPILE_ARGS += -sv
    SIM_ARGS += -voptargs=+acc
endif

# Include cocotb makefile
include $(shell cocotb-config --makefiles)/Makefile.sim

# =============================================================================
# Additional Targets
# =============================================================================

.PHONY: lint waves synth clean_all

# Verilator lint check
lint:
	@echo "Running Verilator lint..."
	verilator --lint-only -Wall \
		--timing \
		-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
		$(VERILOG_SOURCES)
	@echo "Lint PASSED"

# Open waveforms (GTKWave)
waves: sim
	@if [ -f dump.fst ]; then \
		gtkwave dump.fst zero_dsp.gtkw & \
	elif [ -f dump.vcd ]; then \
		gtkwave dump.vcd & \
	else \
		echo "No waveform file found. Run 'make sim' first."; \
	fi

# Quick synthesis check (requires Vivado)
synth:
	@echo "Running synthesis..."
	cd ../scripts && vivado -mode batch -source synth_zero_dsp.tcl

# Resource check (parse synthesis report)
resource_check:
	@echo "Checking DSP usage from synthesis report..."
	@grep -i "DSP" ../scripts/output/utilization_synth.rpt || echo "Run 'make synth' first"

# Generate Python model reference outputs
golden:
	@echo "Generating golden reference outputs..."
	python3 -c "from test_zero_dsp_correlator import ZeroDspGoldenModel; \
		m = ZeroDspGoldenModel(64, 16); \
		import numpy as np; \
		m.set_coefficients(np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1] + [0]*51)); \
		print('Barker-13 encoded:', hex(m.encode_coefficients()))"

# Full verification flow
verify: lint sim
	@echo "============================================"
	@echo " VERIFICATION COMPLETE"
	@echo "============================================"

# Extended clean
clean_all: clean
	rm -rf __pycache__ *.vcd *.fst *.gtkw
	rm -rf ../scripts/output
	rm -rf .pytest_cache

# Help
help:
	@echo "Zero-DSP Correlator Build System"
	@echo "================================"
	@echo "make sim          - Run Cocotb simulation (default: Verilator)"
	@echo "make sim SIM=icarus - Run with Icarus Verilog"
	@echo "make lint         - Verilator lint check"
	@echo "make waves        - Open waveform viewer"
	@echo "make synth        - Run Vivado synthesis"
	@echo "make verify       - Full lint + sim flow"
	@echo "make clean        - Clean simulation artifacts"
	@echo "make clean_all    - Deep clean all artifacts"
