# Zero-DSP Correlation Engine

**QEDMMA FPGA Optimization Module**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![RTL Verified](https://img.shields.io/badge/RTL-Cocotb%20Verified-green.svg)]()
[![FPGA: Zero DSP](https://img.shields.io/badge/FPGA-Zero%20DSP-red.svg)]()

> **Eliminate hardware multipliers from radar signal correlation through algebraic decomposition.**

## Overview

The Zero-DSP Correlation Engine achieves **real-time pulse compression without consuming a single DSP48 slice**, freeing these expensive resources for channelizers, beamformers, and other operations that truly require them.

### Key Features

| Feature | Specification |
|---------|---------------|
| **DSP Usage** | 0 slices (guaranteed by RTL attributes + XDC constraints) |
| **Throughput** | ≥100 MSPS @ 100 MHz |
| **Correlation Length** | Configurable 4-1024 taps |
| **Coefficient Types** | Binary (±1), Ternary (±1,0), CSD (arbitrary) |
| **Interface** | AXI-Stream (data) + AXI-Lite (control) |
| **Verification** | Cocotb testbench with bit-exact golden model |

## Mathematical Foundation

### The Zero-DSP Insight

For radar waveforms using binary/ternary codes (Barker, MLS, etc.), multiplication becomes conditional addition:

```
x · y = +x  if y = +1
      = -x  if y = -1
      =  0  if y = 0
```

For arbitrary coefficients, we use **Canonical Signed Digit (CSD)** encoding:

```
Example: 23 = 2⁵ - 2³ - 2⁰
Therefore: x × 23 = (x << 5) - (x << 3) - x
```

This converts multiplication to shifts and adds—operations that use only LUTs.

## Directory Structure

```
zero_dsp_forge/
├── docs/
│   └── ZERO_DSP_SPEC.md      # Full specification with requirements
├── rtl/
│   └── zero_dsp_correlator.sv # SystemVerilog implementation
├── tb/
│   ├── test_zero_dsp_correlator.py  # Cocotb testbench
│   └── Makefile                      # Simulation flow
├── constraints/
│   └── zero_dsp_synth.xdc    # Vivado constraints
├── scripts/
│   └── synth_zero_dsp.tcl    # Synthesis automation
└── python/
    └── zero_dsp.py           # Software model + HW interface
```

## Quick Start

### 1. Software Verification (Python)

```python
from zero_dsp import ZeroDspCorrelator, RadarCodes
import numpy as np

# Create correlator
corr = ZeroDspCorrelator(length=64, mode='software')
corr.set_coefficients(RadarCodes.BARKER_13)

# Process samples
signal = np.random.randn(1000).astype(np.int16) * 100
output = corr.process(signal)

print(f"Peak: {corr.peak_value} at index {corr.peak_index}")
```

### 2. RTL Simulation (Cocotb)

```bash
cd tb/
make sim                    # Run with Verilator (default)
make sim SIM=icarus         # Run with Icarus Verilog
make waves                  # View waveforms
make lint                   # Verilator lint check
```

### 3. FPGA Synthesis (Vivado)

```bash
cd scripts/
vivado -mode batch -source synth_zero_dsp.tcl

# For different targets:
vivado -mode batch -source synth_zero_dsp.tcl -tclargs -part xc7z020clg400-1
```

## CSD Coefficient Encoding

For arbitrary FIR coefficients:

```python
from zero_dsp import CSDEncoder

# Encode coefficient
coef = 127
csd = CSDEncoder.encode(coef)
print(CSDEncoder.to_operations(csd))
# Output: (x << 7) - x

# Generate Verilog
verilog = CSDEncoder.generate_verilog(127, data_width=16, out_width=24)
print(verilog)
```

## Resource Utilization

**Target: AMD ZU48DR (RFSoC 4x2)**

| Resource | 64-tap Binary | 64-tap CSD | Available |
|----------|--------------|------------|-----------|
| LUT | ~400 | ~2000 | 425,280 |
| FF | ~200 | ~800 | 850,560 |
| DSP | **0** | **0** | 4,272 |
| BRAM | 0 | 0 | 2,160 |

## Performance Comparison

| Method | DSP Slices | LUTs | Max Freq | Latency |
|--------|------------|------|----------|---------|
| Conventional (DSP) | 64 | ~100 | 500 MHz | 3 cycles |
| Zero-DSP Binary | 0 | ~400 | 200 MHz | 4 cycles |
| Zero-DSP CSD | 0 | ~2000 | 150 MHz | 6 cycles |

**Trade-off:** Slightly lower fmax and higher LUT usage, but **frees all DSP slices** for other processing.

## Integration with QEDMMA

This module integrates with the QEDMMA radar tracking system:

```
┌─────────────────────────────────────────────────────────────┐
│  QEDMMA Signal Processing Chain                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ADC → DDC → ┌─────────────────┐ → Detection → Tracker     │
│              │  Zero-DSP       │                            │
│              │  Correlator     │                            │
│              │  (This Module)  │                            │
│              └─────────────────┘                            │
│                                                             │
│  DSP Slices freed for: Channelizer, Beamformer, Doppler   │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

### Software
- Python 3.8+
- NumPy

### Simulation
- Cocotb 1.8+
- Verilator 5.x or Icarus Verilog

### Synthesis
- Vivado 2023.1+ (for RFSoC targets)

## License

**AGPL-3.0-or-later**

This module is part of the QEDMMA open-source ecosystem. Commercial licenses available for proprietary integration.

**Contact:**
- Email: mladen@nexellum.com
- Phone: +385 99 737 5100

## References

1. Proakis, J. G., & Manolakis, D. G. (2007). *Digital Signal Processing*
2. Parhi, K. K. (1999). *VLSI Digital Signal Processing Systems*
3. Barker, R. H. (1953). "Group Synchronization of Binary Digital Systems"

---

**© 2026 Dr. Mladen Mešter / Nexellum d.o.o. All rights reserved.**
