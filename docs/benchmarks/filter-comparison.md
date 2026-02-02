# Filter Benchmarks

Comprehensive benchmarks comparing EKF, UKF, CKF, and Zero-DSP correlation algorithms.

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Monte Carlo runs | 50-100 |
| Scenario length | 100 steps |
| Time step | 0.1 s |
| Target | Constant velocity + maneuver |
| Measurement | Range + bearing (radar) |

---

## Benchmark 1: Filter Accuracy (SNR=10dB)

| Filter | RMSE Position (m) | RMSE Velocity (m/s) | Time (μs) | Divergence |
|--------|-------------------|---------------------|-----------|------------|
| **EKF** | 3.89 | 4.94 | **36.0** | 0% |
| **UKF** | **3.74** | 10.80 | 240.8 | 0% |
| **CKF** | 3.89 | 4.94 | 184.9 | 0% |
| **SR-CKF** | 3.99 | 4.97 | 197.2 | 0% |

!!! tip "Recommendation"
    - **Speed critical:** Use EKF (7x faster)
    - **Accuracy critical:** Use UKF (best position RMSE)
    - **Balance:** Use CKF (good accuracy, moderate speed)

---

## Benchmark 2: SNR Sensitivity

| SNR (dB) | EKF RMSE (m) | UKF RMSE (m) | CKF RMSE (m) |
|----------|--------------|--------------|--------------|
| 20 | 2.47 | **1.76** | 2.47 |
| 10 | 3.91 | **3.83** | 3.91 |
| 5 | **5.83** | 34.64 | **5.83** |
| 0 | **10.00** | 32.38 | **10.00** |

!!! warning "UKF at Low SNR"
    UKF can diverge at very low SNR (< 5 dB) due to sigma point spread.
    Consider EKF or CKF for challenging environments.

---

## Benchmark 3: High-Dimensional Stability (n=9)

Testing 9-dimensional state (position, velocity, acceleration in 3D):

| Filter | Divergence Rate |
|--------|-----------------|
| UKF | 0.0% |
| CKF | 0.0% |

Both filters remain stable for 9D tracking in this test. CKF has theoretical advantages for n > 3 due to positive weights.

---

## Benchmark 4: Zero-DSP Correlation

| SNR (dB) | 1-bit Accuracy | 2-bit Accuracy | 1-bit Time (μs) | 2-bit Time (μs) |
|----------|----------------|----------------|-----------------|-----------------|
| 20 | 100% | 100% | 1880 | 177675 |
| 10 | 100% | 100% | 1821 | 174158 |
| 5 | 100% | 100% | 1828 | 173334 |
| 0 | 100% | 100% | 1829 | 177748 |
| -5 | 100% | 100% | 1836 | 174535 |

!!! success "Zero-DSP Performance"
    1-bit correlation achieves 100% accuracy even at -5 dB SNR
    while using **0 DSP blocks** on FPGA.

---

## Performance Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FILTER RECOMMENDATIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Scenario                          │ Recommended Filter                     │
├────────────────────────────────────┼────────────────────────────────────────┤
│  Linear/mildly nonlinear (n≤4)     │ EKF (fastest)                         │
│  Highly nonlinear (range-bearing)  │ UKF (best accuracy)                   │
│  High-dimensional (n>4)            │ CKF (stable weights)                  │
│  Numerical stability critical      │ SR-CKF (guaranteed PD)                │
│  Time-varying noise                │ UKF + Sage-Husa adaptive              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ZERO-DSP CORRELATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  1-bit: Best for high-SNR (>10dB), maximum FPGA throughput                 │
│  2-bit: Better low-SNR performance, still 0 DSP blocks                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Running Benchmarks

```bash
# Clone repository
git clone https://github.com/mladen1312/qedmma-lite.git
cd qedmma-lite

# Run benchmarks
python benchmark/benchmark_suite.py
```

---

## Reproducing Results

```python
from qedmma.advanced import UnscentedKalmanFilter, CubatureKalmanFilter
from benchmark.benchmark_suite import RadarScenario, run_filter_benchmark

# Create scenario
scenario = RadarScenario(n_steps=100, snr_db=10)

# Run benchmark
result = run_filter_benchmark("My Filter", my_filter, scenario, n_monte_carlo=100)
print(f"RMSE: {result.rmse_position:.2f}m")
```

---

## Hardware Benchmarks

For FPGA synthesis results, see [FPGA Synthesis Guide](../fpga/synthesis.md).

| IP Core | Target | LUTs | FFs | DSPs | Fmax |
|---------|--------|------|-----|------|------|
| Zero-DSP (N=1024) | ZU+ | 2,847 | 1,156 | **0** | 1.2 GHz |
| Zero-DSP (N=4096) | ZU+ | 11,284 | 4,612 | **0** | 1.0 GHz |

---

*Benchmarks performed on: AMD Ryzen 9 5900X, Python 3.12, NumPy 1.26*
