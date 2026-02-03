# QEDMMA-Lite

[![PyPI version](https://badge.fury.io/py/qedmma.svg)](https://badge.fury.io/py/qedmma)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/mladen1312/qedmma-lite/actions/workflows/tests.yml/badge.svg)](https://github.com/mladen1312/qedmma-lite/actions)

**Production-ready multi-model tracking library with clean API and competitive performance.**

> *"QEDMMA is to Stone Soup what Flask is to Django — 80% of use cases, 5 minutes setup."*

---

## 🎯 Why QEDMMA?

Standard Kalman filters assume constant motion models. Real targets **maneuver**. When motion model assumptions are violated, tracking accuracy degrades significantly.

QEDMMA solves this with Interacting Multiple Model (IMM) tracking that automatically switches between motion models based on measurement likelihood.

### Benchmarks (10-run Monte Carlo average)

| Scenario | QEDMMA | FilterPy EKF | FilterPy IMM | Notes |
|:---------|:------:|:------------:|:------------:|:------|
| Linear (CV) | 17m | **15m** | 16m | Simple scenario - EKF optimal |
| **Maneuvering (3g)** | **28m** | 31m | 28m | IMM shines here |
| **High Noise (σ=200m)** | **81m** | 86m | 86m | +6% improvement |
| **Evasive (Random)** | **32m** | 41m | 32m | +22% vs EKF |

*Position RMSE in meters. Lower is better. Bold = best in category.*

**Key insight:** QEDMMA excels in maneuvering, high-noise, and unpredictable scenarios where single-model filters struggle. For simple constant-velocity tracking, a well-tuned EKF is hard to beat.

---

## 🚀 Quick Start

```bash
pip install qedmma
```

```python
from qedmma import IMM, IMMConfig
import numpy as np

# Configure: 4D state [x, vx, y, vy], 2D measurements [x, y]
config = IMMConfig(
    dim_state=4,
    dim_meas=2,
    models=['CV', 'CA', 'CT']  # Constant Velocity, Acceleration, Turn
)

imm = IMM(config)

# Initialize
x0 = np.array([0, 0, 100, 50])        # Starting state
P0 = np.diag([100, 100, 10, 10])**2   # Initial covariance
Q = np.diag([1, 1, 10, 10])           # Process noise
R = np.diag([50, 50])**2              # Measurement noise

state = imm.init_state(x0, P0, Q, R)

# Track
for z in measurements:
    state = imm.predict(state, dt=0.1)
    state, likelihood = imm.update(state, z)
    
    print(f"Position: {state.x[:2]}")
    print(f"Model probs: CV={state.mu[0]:.0%} CA={state.mu[1]:.0%} CT={state.mu[2]:.0%}")
```

**That's 6 lines of setup vs 50+ for FilterPy IMM.**

---

## 📊 QEDMMA-Lite vs QEDMMA-PRO

| Feature | Lite (MIT) | PRO (Commercial) |
|:--------|:----------:|:----------------:|
| IMM Filter (CV/CA/CT) | ✅ | ✅ |
| UKF, CKF filters | ✅ | ✅ |
| Adaptive Noise Estimation | ✅ | ✅ |
| Zero-DSP Correlation | Basic | **Advanced** |
| **FPGA IP Cores** | ❌ | ✅ 22 cores |
| Multi-target (1000+) | ❌ | ✅ |
| ML-CFAR, Micro-Doppler | ❌ | ✅ |
| Anomaly Hunter™ Layer 2B | ❌ | ✅ **EXCLUSIVE** |
| Link-16, ASTERIX output | ❌ | ✅ |
| DO-254 / ISO 26262 docs | ❌ | ✅ |
| **License** | MIT | Commercial |
| **Price** | Free | $25K-$350K |

---

## 🔧 Features

### IMM (Interacting Multiple Model)
Automatically switches between motion models based on measurement likelihood:
- **CV**: Constant Velocity — straight-line motion
- **CA**: Constant Acceleration — speeding up/slowing down  
- **CT**: Coordinated Turn — maneuvering targets

### Advanced Filters
```python
from qedmma.advanced import UKF, CKF, AdaptiveNoiseEstimator

# Unscented Kalman Filter for nonlinear systems
ukf = UKF(f, h, n_states=6, n_meas=3)

# Cubature Kalman Filter (better for high dimensions)
ckf = CKF(f, h, n_states=9, n_meas=3)

# Adaptive noise estimation
estimator = AdaptiveNoiseEstimator(window=20)
R_estimated = estimator.estimate(innovations)
```

### Zero-DSP Correlation (Lite)
Low-complexity target association without expensive FFT operations:
```python
from qedmma.zero_dsp import ZeroDSPCorrelator

correlator = ZeroDSPCorrelator(threshold=0.7)
associations = correlator.associate(tracks, detections)
```

---

## 📚 Use Cases

- 🚗 **Autonomous Vehicles** — Sensor fusion, pedestrian/vehicle tracking
- 🚁 **Drones** — Target tracking, collision avoidance
- 🤖 **Robotics** — SLAM, object manipulation
- 📊 **Sports Analytics** — Player/ball tracking
- 📡 **Radar/Sonar** — Maneuvering target tracking
- 🛰️ **Aerospace** — Satellite tracking, debris monitoring

---

## ⚖️ Licensing

### MIT License (Free)

QEDMMA-Lite is released under the **MIT License** — use it freely in any project, commercial or open-source.

### Commercial License (QEDMMA-PRO)

For organizations requiring:
- 🔓 FPGA IP Cores (SystemVerilog/VHDL)
- 🚀 Physics-Agnostic Anomaly Hunter™
- 🛡️ DO-254 / ISO 26262 certification support
- 🆘 Priority engineering support

**Contact:** [mladen@nexellum.com](mailto:mladen@nexellum.com)

| Use Case | MIT (Lite) | Commercial (PRO) |
|:---------|:----------:|:----------------:|
| Academic research | ✅ Free | Optional |
| Internal R&D | ✅ Free | Optional |
| Open-source project | ✅ Free | Optional |
| Closed-source product | ✅ Free | Recommended |
| FPGA deployment | ❌ N/A | ✅ Required |
| Defense/aerospace | ✅ Free | ✅ Recommended |

---

## 📖 Documentation

- [API Reference](docs/api.md)
- [Tutorials](docs/tutorials/)
- [Examples](examples/)
- [Benchmarks](benchmarks/)

---

## 🏆 Why Choose QEDMMA?

| | QEDMMA | FilterPy | Stone Soup |
|:--|:------:|:--------:|:----------:|
| **IMM setup** | 6 lines | 50+ lines | 100+ lines |
| **Learning curve** | Minutes | Hours | Days |
| **Code size** | ~5K LOC | ~15K LOC | ~100K LOC |
| **Dependencies** | NumPy only | NumPy, SciPy | Heavy |
| **FPGA ready** | PRO ✅ | ❌ | ❌ |
| **Active 2026** | ✅ | ⚠️ 2023 | ✅ |

---

## 🔬 Running Benchmarks

```bash
# Clone the repo
git clone https://github.com/mladen1312/qedmma-lite.git
cd qedmma-lite

# Install dependencies
pip install -e ".[benchmark]"

# Run benchmark suite
python benchmarks/benchmark_suite.py
```

The benchmark compares QEDMMA against FilterPy across 5 scenarios with 10 Monte Carlo runs each.

---

## 📬 Contact

**Dr. Mladen Mešter**  
Nexellum d.o.o.  
📧 [mladen@nexellum.com](mailto:mladen@nexellum.com)  
🌐 [nexellum.com](https://nexellum.com)  
📱 +385 99 737 5100

---

## 📜 Citation

```bibtex
@software{qedmma2026,
  author = {Mešter, Mladen},
  title = {QEDMMA: Multi-Model Tracking Library},
  year = {2026},
  url = {https://github.com/mladen1312/qedmma-lite}
}
```

---

## ⭐ Star History

If QEDMMA helps your project, please star the repo!

---

*Built with 🔬 by [Dr. Mladen Mešter](mailto:mladen@nexellum.com) | [Nexellum d.o.o.](https://nexellum.com)*
