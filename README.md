# QEDMMA-Lite

[![PyPI version](https://badge.fury.io/py/qedmma.svg)](https://badge.fury.io/py/qedmma)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Production-ready IMM tracking with the simplest API.**

> *"Same algorithm as FilterPy IMM, but setup in 6 lines instead of 50+"*

---

## 🎯 Why QEDMMA?

Standard Kalman filters assume constant motion. Real targets **maneuver**. When this assumption breaks, tracking accuracy degrades significantly.

**IMM (Interacting Multiple Model)** solves this by automatically switching between motion models. QEDMMA makes IMM accessible without the complexity.

### Benchmark Results (20-run Monte Carlo)

| Scenario | IMM (QEDMMA/FilterPy) | Single-Model EKF | IMM Advantage |
|:---------|:---------------------:|:----------------:|:-------------:|
| Linear (CV) | **12m** | 11m | -9% (EKF optimal here) |
| **Maneuvering (3g)** | **40m** | 124m | **+68%** |
| **Evasive (random)** | **42m** | 115m | **+63%** |

*Position RMSE in meters. Lower is better.*

**Key insight:** IMM provides 60-70% improvement on maneuvering targets compared to single-model EKF. For simple linear motion, a well-tuned EKF is marginally better.

---

## 🚀 Quick Start

```bash
pip install qedmma
```

```python
from qedmma import QEDMMALite

# Create tracker - that's it!
tracker = QEDMMALite(dt=0.1, meas_noise=50.0)

# Initialize with first measurement
tracker.initialize(pos=[x0, y0], vel=[0, 0])

# Track
for measurement in measurements:
    state = tracker.update(measurement)
    print(f"Position: {state.pos}, Model probs: {state.model_probs}")
```

**6 lines.** Compare to [FilterPy IMM setup](https://filterpy.readthedocs.io/en/latest/kalman/IMMEstimator.html) which requires 50+ lines.

---

## 📊 QEDMMA vs FilterPy

| Aspect | QEDMMA | FilterPy IMM |
|:-------|:------:|:------------:|
| **Algorithm** | IMM (3 models) | IMM (configurable) |
| **Performance** | Identical¹ | Identical¹ |
| **Setup code** | **6 lines** | 50+ lines |
| **Learning curve** | Minutes | Hours |
| **Pre-tuned for radar** | ✅ | ❌ |
| **FPGA IP cores** | ✅ (PRO) | ❌ |

¹*With identical parameters, QEDMMA and FilterPy IMM produce the same results (<1% difference)*

### Why not just use FilterPy?

FilterPy is excellent, but requires you to:
- Create separate KalmanFilter objects for each model
- Configure transition matrices, process noise, measurement noise manually
- Set up the IMMEstimator with mode transition probabilities
- Handle the column-vector format

QEDMMA does all this for you with radar-optimized defaults.

---

## 🔧 Features

### Three Motion Models (CV/CA/CT)
- **CV (Constant Velocity)**: For straight-line motion
- **CA (Constant Acceleration)**: For speeding up/slowing down
- **CT (Coordinated Turn)**: For maneuvering targets

### Automatic Model Switching
The IMM algorithm automatically adjusts model probabilities based on measurement likelihood. No manual intervention needed.

### Radar-Optimized Defaults
Pre-tuned process noise (Q) values for typical radar tracking scenarios:
- `q_cv = 0.1` (low noise for straight flight)
- `q_ca = 5.0` (medium for acceleration)
- `q_ct = 30.0` (high for maneuvers)

---

## 📦 LITE vs PRO

| Feature | Lite (MIT) | PRO (Commercial) |
|:--------|:----------:|:----------------:|
| IMM Filter (CV/CA/CT) | ✅ | ✅ |
| Python implementation | ✅ | ✅ |
| **FPGA IP Cores** | ❌ | ✅ 22 cores |
| Multi-target (1000+) | ❌ | ✅ |
| Anomaly Hunter™ | ❌ | ✅ |
| DO-254 / ISO 26262 | ❌ | ✅ |
| **Price** | Free | $25K-$350K |

---

## 📚 Use Cases

- 🚗 **Autonomous Vehicles** — Pedestrian/vehicle tracking
- 🚁 **Drones** — Target tracking, collision avoidance
- 🤖 **Robotics** — SLAM, object manipulation
- 📡 **Radar/Sonar** — Maneuvering target tracking
- 🛰️ **Aerospace** — Satellite, debris monitoring

---

## ⚖️ License

**MIT License** — Use freely in any project.

For FPGA IP cores, multi-target tracking, or commercial support:  
📧 [mladen@nexellum.com](mailto:mladen@nexellum.com)

---

## 📬 Contact

**Dr. Mladen Mešter**  
Nexellum d.o.o.  
📧 [mladen@nexellum.com](mailto:mladen@nexellum.com)  
🌐 [nexellum.com](https://nexellum.com)

---

## 📜 Citation

```bibtex
@software{qedmma2026,
  author = {Mešter, Mladen},
  title = {QEDMMA: Production-Ready IMM Tracking Library},
  year = {2026},
  url = {https://github.com/mladen1312/qedmma-lite}
}
```

---

*Built with 🔬 by [Dr. Mladen Mešter](mailto:mladen@nexellum.com) | [Nexellum d.o.o.](https://nexellum.com)*
