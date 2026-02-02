# QEDMMA-Lite Documentation

<div align="center">
  <h2>Quantum-Enhanced Data-Driven Multi-Model Algorithm</h2>
  <p><strong>Advanced Radar Tracking • FPGA Optimized • Open Source (AGPL-3.0)</strong></p>
</div>

---

## 🎯 What is QEDMMA-Lite?

QEDMMA-Lite is a state-of-the-art multi-target tracking library designed for radar systems. It provides:

- **Advanced Kalman Filters**: UKF, CKF, and adaptive variants
- **Multi-Target Tracking**: IMM, JPDA, MHT algorithms
- **FPGA Acceleration**: Zero-DSP correlation, HLS-ready code
- **Real-Time Performance**: Optimized for embedded deployment

```python
from qedmma.advanced import create_radar_ukf

# Create radar tracker with UKF
ukf, state = create_radar_ukf(dt=0.1)

# Track target
for measurement in radar_detections:
    state = ukf.predict(state, dt=0.1)
    state, innovation = ukf.update(state, measurement)
    print(f"Target position: ({state.x[0]:.1f}, {state.x[1]:.1f})")
```

---

## 🚀 Key Features

### v3.0.0 - Advanced Filters Release

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **UKF** | Unscented Kalman Filter | Nonlinear radar tracking |
| **CKF** | Cubature Kalman Filter | High-dimensional states (n>3) |
| **Adaptive Noise** | Sage-Husa estimator | Time-varying clutter |
| **Zero-DSP Correlation** | XOR + popcount | FPGA signal processing |

---

## 📊 Performance

```
┌─────────────────────────────────────────────────────────┐
│  Benchmark: 100 Monte Carlo runs, SNR=10dB             │
├─────────────────┬───────────┬───────────┬──────────────┤
│  Filter         │ RMSE (m)  │ Time (ms) │ FPGA Ready   │
├─────────────────┼───────────┼───────────┼──────────────┤
│  EKF            │ 12.4      │ 0.8       │ Yes          │
│  UKF            │ 8.7       │ 2.1       │ Moderate     │
│  CKF            │ 8.9       │ 1.9       │ Moderate     │
│  UKF + Adaptive │ 6.2       │ 2.4       │ Moderate     │
└─────────────────┴───────────┴───────────┴──────────────┘
```

---

## 🔐 License

QEDMMA-Lite is licensed under **AGPL-3.0-or-later**.

!!! warning "Commercial Use"
    If you use QEDMMA-Lite in proprietary software, you must either:
    
    1. Open-source your entire application under AGPL
    2. Purchase a commercial license from Nexellum

**Commercial Licensing:**

- 📧 Email: mladen@nexellum.com
- 🌐 Web: [www.nexellum.com](https://www.nexellum.com)
- 📱 Phone: +385 99 737 5100

---

## 📚 Quick Links

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Get started with pip install

    [:octicons-arrow-right-24: Install Guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Track your first target in 5 minutes

    [:octicons-arrow-right-24: Tutorial](getting-started/quickstart.md)

-   :material-chip:{ .lg .middle } **FPGA Guide**

    ---

    Deploy on Zynq/UltraScale+

    [:octicons-arrow-right-24: FPGA Docs](fpga/overview.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete API documentation

    [:octicons-arrow-right-24: API Docs](api/advanced.md)

</div>

---

## 🏢 About

QEDMMA is developed by **Dr. Mladen Mešter** at **Nexellum**.

The project bridges academic research with production-ready implementations, providing defense-grade tracking algorithms with open-source accessibility.

---

<div align="center">
  <sub>Built with ❤️ in Croatia | © 2026 Nexellum</sub>
</div>
