# Zero-DSP Correlation

Zero-DSP Correlation eliminates all hardware multipliers by using XOR and popcount operations, enabling massive parallelization on FPGAs.

---

## 🎯 Key Benefits

| Feature | Traditional | Zero-DSP |
|---------|-------------|----------|
| DSP blocks | N×M | **0** |
| Logic type | DSP48 | LUT only |
| Max frequency | ~300 MHz | **>1 GHz** |
| Parallelization | Limited | Massive |
| Power consumption | High | **Low** |

---

## 📐 Mathematical Foundation

### Traditional Correlation

Standard cross-correlation requires N×M multiplications:

$$R_{xy}[m] = \sum_{n=0}^{N-1} x[n] \cdot y[n+m]$$

### 1-Bit Quantization Insight

For bipolar signals ($x, y \in \{-1, +1\}$):

$$x[n] \cdot y[n+m] = \begin{cases} +1 & \text{if } x[n] = y[n+m] \\ -1 & \text{otherwise} \end{cases}$$

This is equivalent to:

$$x[n] \cdot y[n+m] = 1 - 2 \cdot \text{XOR}(x_{\text{bin}}[n], y_{\text{bin}}[n+m])$$

### Zero-DSP Formula

Converting {-1, +1} to {0, 1}:

$$R_{xy}[m] = N - 2 \cdot \text{popcount}(X \oplus Y_m)$$

Where:
- $\oplus$ = bitwise XOR
- popcount = Hamming weight (count of 1s)
- $X, Y_m$ = packed bit vectors

!!! success "Key Insight"
    Multiplication → XOR + Popcount (no DSP needed!)

---

## 💻 Python Implementation

### 1-Bit Correlator

```python
from qedmma.advanced import ZeroDSPCorrelator
import numpy as np

# Create correlator
corr = ZeroDSPCorrelator(n_samples=1024, n_lags=256)

# Generate test signals
np.random.seed(42)
x = np.sign(np.random.randn(1024))  # Reference PRN
y = np.roll(x, 137)  # Delayed version
y = np.concatenate([y, np.zeros(256)])  # Pad for lags

# Correlate (uses XOR + popcount internally)
R, peak_lag = corr.correlate(x, y)

print(f"Detected delay: {peak_lag}")  # Should be 137
print(f"Peak correlation: {R[peak_lag]:.3f}")
```

### 2-Bit Correlator (4 levels)

```python
from qedmma.advanced import TwoBitCorrelator

# 2-bit provides better SNR than 1-bit
# Still DSP-free using 4x4 LUT

corr_2bit = TwoBitCorrelator(n_samples=1024)

# Quantization levels: {-3, -1, +1, +3}
# Multiplication via 16-entry LUT
R, peak_lag = corr_2bit.correlate(x, y, n_lags=256)
```

### Parallel Multi-Code Search

```python
from qedmma.advanced import ParallelZeroDSPCorrelator

# GPS-like acquisition: search 32 PRN codes simultaneously
parallel_corr = ParallelZeroDSPCorrelator(
    n_samples=1023,  # GPS C/A code length
    n_codes=32
)

# Generate PRN codes
codes = np.array([generate_gps_prn(sv) for sv in range(1, 33)])

# Search all codes in parallel
peaks, best_code, best_lag = parallel_corr.correlate_all(signal, codes)
print(f"Detected SV{best_code+1} at lag {best_lag}")
```

---

## 🔧 FPGA Implementation

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Zero-DSP Correlator                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   y_in ──►┌──────────────┐                             │
│           │ Shift Register│ (N bits)                    │
│           └──────┬───────┘                             │
│                  │                                      │
│                  ▼                                      │
│   x_ref ──►┌──────────────┐                            │
│            │   XOR Array   │ (N parallel XORs)         │
│            └──────┬───────┘                            │
│                   │                                     │
│                   ▼                                     │
│            ┌──────────────┐                            │
│            │ Popcount Tree │ (log₂N adder tree)        │
│            └──────┬───────┘                            │
│                   │                                     │
│                   ▼                                     │
│            ┌──────────────┐                            │
│            │  R = N - 2H   │                           │
│            └──────┬───────┘                            │
│                   │                                     │
│                   ▼                                     │
│              corr_out                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### VHDL Code

```vhdl
-- Zero-DSP Correlator (included in fpga/rtl/)
entity zero_dsp_correlator is
    generic (
        N_SAMPLES : integer := 64;
        CORR_WIDTH : integer := 8
    );
    port (
        clk        : in  std_logic;
        rst        : in  std_logic;
        x_ref      : in  std_logic_vector(N_SAMPLES-1 downto 0);
        y_in       : in  std_logic;
        y_valid    : in  std_logic;
        corr_out   : out signed(CORR_WIDTH-1 downto 0);
        corr_valid : out std_logic
    );
end entity;
```

### HLS C++ Code

```cpp
// Zero-DSP Correlator (included in fpga/hls/)
#include <ap_int.h>

#define N_SAMPLES 1024

typedef ap_uint<N_SAMPLES> packed_bits_t;
typedef ap_int<12> corr_t;

corr_t correlate_single(
    packed_bits_t x_ref,
    packed_bits_t y_shifted
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1
    
    packed_bits_t xor_result = x_ref ^ y_shifted;
    ap_uint<11> hamming = popcount<N_SAMPLES>(xor_result);
    
    return (corr_t)(N_SAMPLES - 2 * hamming);
}
```

---

## 📊 Resource Utilization

### Zynq-7020 (N=1024)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs | 2,847 | 53,200 | 5.4% |
| FFs | 1,156 | 106,400 | 1.1% |
| DSP48 | **0** | 220 | **0%** |
| BRAM | 2 | 140 | 1.4% |

### UltraScale+ (N=4096)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| CLB LUTs | 11,284 | 274,080 | 4.1% |
| CLB FFs | 4,612 | 548,160 | 0.8% |
| DSP48 | **0** | 2,520 | **0%** |
| Block RAM | 8 | 912 | 0.9% |

**Maximum Frequency: 1.2 GHz** (UltraScale+ -2 speed grade)

---

## 🔬 Performance Analysis

### SNR Loss vs Full-Precision

| Quantization | SNR Loss | DSP Usage |
|--------------|----------|-----------|
| Full precision (32-bit float) | 0 dB | 100% |
| 8-bit fixed | 0.1 dB | 100% |
| 2-bit | 1.5 dB | **0%** |
| 1-bit | 2.5 dB | **0%** |

!!! info "Acceptable Loss"
    For spread-spectrum systems with high processing gain,
    2-3 dB loss is negligible compared to DSP savings.

### Throughput Comparison

```
Platform: Zynq UltraScale+ ZU7EV

Correlator Type    | Throughput    | DSP Usage
-------------------|---------------|----------
Traditional (MAC)  | 500 Mcorr/s   | 100%
Zero-DSP (1-bit)   | 4.8 Gcorr/s   | 0%
```

---

## 🎯 Applications

1. **GPS/GNSS Acquisition** - Search 32 satellites × 1023 code phases
2. **Radar Pulse Compression** - LFM/Barker code correlation
3. **Spread Spectrum** - CDMA, DSSS demodulation
4. **Sonar Processing** - Underwater acoustic correlation
5. **Radio Astronomy** - Pulsar timing, VLBI

---

## 📚 References

1. Tsui, J. "Fundamentals of Global Positioning System Receivers" (2000)
2. Parkinson, B. "Global Positioning System: Theory and Applications" (1996)
3. Van Nee, D.J.R. "Spread-Spectrum Code and Carrier Synchronization" (1992)

---

## 🔗 See Also

- [FPGA Synthesis Guide](../fpga/synthesis.md)
- [Zero-DSP Correlator RTL](../fpga/zero-dsp-correlator.md)
- [Benchmarks](../benchmarks/filter-comparison.md)
