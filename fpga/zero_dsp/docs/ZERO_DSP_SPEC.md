# Zero-DSP Correlation Engine Specification
## QEDMMA FPGA Optimization Module

**Version:** 1.0.0  
**Author:** Dr. Mladen Mešter / Nexellum d.o.o.  
**License:** AGPL-3.0 (Open Source) / Commercial Available  
**Date:** 2026-02-02

---

## 1. Overview

The Zero-DSP Correlation Engine eliminates hardware multipliers from radar signal correlation by exploiting signal quantization and algebraic decomposition. This enables deployment on low-cost FPGAs without DSP slices while maintaining real-time performance.

## 2. Requirements Traceability

| REQ-ID | Description | Priority | Verification |
|--------|-------------|----------|--------------|
| [REQ-ZD-001] | Zero DSP48 slice usage | MUST | Resource report |
| [REQ-ZD-002] | Bit-exact vs reference | MUST | Cocotb testbench |
| [REQ-ZD-003] | Throughput ≥ 100 MSPS | MUST | Timing analysis |
| [REQ-ZD-004] | LUT < 5000, FF < 2000 | SHOULD | Utilization report |
| [REQ-ZD-005] | Latency ≤ 32 cycles | SHOULD | Simulation |
| [REQ-ZD-006] | Configurable correlation length | MUST | Parameter sweep |

## 3. Mathematical Foundation

### 3.1 Conventional Correlation

For received signal x[n] and reference y[n], the cross-correlation is:

```
R_xy[k] = Σ(n=0 to N-1) x[n] · y[n+k]
```

Each multiplication requires one DSP48 slice, consuming N DSPs for N-tap correlation.

### 3.2 Zero-DSP Decomposition Methods

#### Method A: Binary Reference (BPSK/Barker)

For binary reference codes y[n] ∈ {-1, +1}:

```
x[n] · y[n] = x[n]      if y[n] = +1
            = -x[n]     if y[n] = -1
```

**Implementation:** MUX + conditional negate (2's complement)

#### Method B: Ternary Reference (PRN with blanking)

For ternary codes y[n] ∈ {-1, 0, +1}:

```
x[n] · y[n] = x[n]      if y[n] = +1
            = 0         if y[n] = 0  
            = -x[n]     if y[n] = -1
```

**Implementation:** 2-bit control MUX

#### Method C: Multi-bit via Shift-Add (CSD)

For arbitrary coefficients, use Canonical Signed Digit (CSD) representation:

```
y = Σ(i) d_i · 2^i,  where d_i ∈ {-1, 0, +1}
```

Example: y = 23 = 32 - 8 - 1 = 2^5 - 2^3 - 2^0

```
x · 23 = (x << 5) - (x << 3) - x
```

**Benefit:** Any coefficient → shifts + adds only

#### Method D: BRAM LUT Multiplication

For fixed coefficients, precompute x·y for all x values:

```
BRAM[addr] = x · y_fixed,  addr = x
```

**Trade-off:** Uses BRAM instead of DSP (often more available)

## 4. Architecture Selection Matrix

| Method | Reference Type | DSP | LUT | BRAM | Throughput | Best For |
|--------|---------------|-----|-----|------|------------|----------|
| Binary | ±1 | 0 | ~100 | 0 | Highest | Barker, BPSK |
| Ternary | ±1, 0 | 0 | ~150 | 0 | High | PRN codes |
| CSD | Any fixed | 0 | ~500 | 0 | Medium | Matched filters |
| BRAM-LUT | Any fixed | 0 | ~50 | 1-2 | High | Complex waveforms |

## 5. Selected Architecture: Hybrid Zero-DSP

For QEDMMA radar application:
- **Primary:** Binary correlation for pulse compression
- **Secondary:** CSD for Doppler filter bank
- **Tertiary:** BRAM-LUT for arbitrary waveforms

## 6. Interface Specification

### 6.1 AXI-Stream Data Interface

```
Signal          Width   Direction   Description
─────────────────────────────────────────────────
s_axis_tdata    16      Input       I/Q sample (8-bit I, 8-bit Q)
s_axis_tvalid   1       Input       Data valid
s_axis_tready   1       Output      Backpressure
s_axis_tlast    1       Input       End of frame

m_axis_tdata    32      Output      Correlation magnitude
m_axis_tvalid   1       Output      Result valid
m_axis_tready   1       Input       Downstream ready
m_axis_tlast    1       Output      End of correlation
```

### 6.2 AXI-Lite Control Interface

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| 0x00 | CTRL | RW | [0] Enable, [1] Reset, [2] Mode |
| 0x04 | STATUS | RO | [0] Busy, [1] Done, [7:4] Error |
| 0x08 | CORR_LEN | RW | Correlation length (4-1024) |
| 0x0C | THRESHOLD | RW | Detection threshold |
| 0x10 | PEAK_VAL | RO | Peak correlation value |
| 0x14 | PEAK_IDX | RO | Peak index (range bin) |

## 7. Verification Plan

| Test ID | Description | Method |
|---------|-------------|--------|
| TB-ZD-001 | Bit-exact vs golden | Cocotb + NumPy reference |
| TB-ZD-002 | Throughput measurement | Cycle counting |
| TB-ZD-003 | Random stimulus | Constrained random |
| TB-ZD-004 | Corner cases | Directed tests |
| TB-ZD-005 | Resource verification | Post-synth report |

---

## Appendix A: CSD Encoding Algorithm

```python
def to_csd(n):
    """Convert integer to Canonical Signed Digit representation"""
    csd = []
    i = 0
    while n != 0:
        if n & 1:  # odd
            if (n & 3) == 3:  # ...11 pattern
                csd.append((i, -1))
                n += 1
            else:
                csd.append((i, 1))
                n -= 1
        n >>= 1
        i += 1
    return csd  # List of (position, sign) tuples
```

## Appendix B: Resource Estimation

For 64-tap correlation:
- Binary method: ~400 LUT, ~200 FF, 0 DSP, 0 BRAM
- CSD method: ~2000 LUT, ~800 FF, 0 DSP, 0 BRAM  
- BRAM-LUT: ~200 LUT, ~100 FF, 0 DSP, 2 BRAM

Target platform: AMD RFSoC 4x2 (ZU48DR)
- Available: 425K LUT, 930 DSP, 2520 BRAM
- Zero-DSP saves DSP for channelizer/beamformer
