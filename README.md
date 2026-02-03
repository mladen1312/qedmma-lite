# QEDMMA-Lite

[![PyPI version](https://badge.fury.io/py/qedmma.svg)](https://badge.fury.io/py/qedmma)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/mladen1312/qedmma-lite/actions/workflows/tests.yml/badge.svg)](https://github.com/mladen1312/qedmma-lite/actions)

**High-performance 4-model IMM tracking library optimized for maneuvering targets.**

> *"QEDMMA is to Stone Soup what Flask is to Django — production-ready tracking in minutes."*

---

## 🎯 Why QEDMMA?

Standard Kalman filters assume **constant velocity**. Real targets **maneuver**. When they do, your filter's error grows rapidly.

QEDMMA solves this with a 4-model Interacting Multiple Model (IMM) filter that automatically switches between:
- **CV**: Constant Velocity — straight-line motion
- **CA**: Constant Acceleration — speeding up/slowing down
- **CT**: Coordinated Turn — banking maneuvers
- **Jerk**: Constant Jerk — rapid acceleration changes

### ✅ Verified Benchmarks

*Monte Carlo simulation, n=10 runs, position RMSE in meters*

| Scenario | FilterPy EKF | FilterPy IMM | QEDMMA IMM | Improvement |
|:---------|:------------:|:------------:|:----------:|:-----------:|
| Linear (CV) | **18m** | **17m** | 25m | — |
| Maneuvering (3g) | 128m | 120m | **71m** | **+44%** ✅ |
| Aggressive (5g+) | 172m | 161m | **139m** | **+19%** ✅ |

**Key insight**: QEDMMA excels on **maneuvering targets**. For purely linear motion, single-model filters have lower computational overhead.

> 💡 **When to use QEDMMA**: Target tracking where maneuvers are expected (aircraft, missiles, evasive vehicles, drones).

---

## 🚀 Quick Start

```bash
pip install qedmma
```

```python
from qedmma import QEDMMATracker, Measurement
import numpy as np

# Create tracker (16 Hz update rate)
tracker = QEDMMATracker(dt=0.0625)

# Initialize with first measurement
tracker.initialize(
    initial_pos=np.array([1000, 2000, 5000]),  # x, y, z in meters
    initial_vel=np.array([200, 0, 0])          # vx, vy, vz in m/s
)

# Track incoming measurements
for pos, t in sensor_data:
    measurement = Measurement(
        pos=pos,
        noise_pos=50.0,  # Position uncertainty (m)
        noise_vel=10.0,  # Velocity uncertainty (m/s)
        time=t
    )
    
    state = tracker.update(measurement)
    
    print(f"Position: {state.pos}")
    print(f"Velocity: {state.vel} ({state.mach():.1f} Mach)")
    print(f"G-load: {state.g_load():.1f}g")
    print(f"Models: {tracker.imm.get_model_probabilities()}")
```

---

## 📊 QEDMMA-Lite vs QEDMMA-PRO

| Feature | Lite (Free) | PRO (Commercial) |
|:--------|:-----------:|:----------------:|
| 4-model IMM (CV/CA/CT/Jerk) | ✅ | ✅ |
| Automatic model adaptation | ✅ | ✅ |
| Python API | ✅ | ✅ |
| FPGA IP Cores | ❌ | ✅ 22 cores |
| Multi-target (1024+) | ❌ | ✅ |
| ML-CFAR, Micro-Doppler | ❌ | ✅ |
| Anomaly Hunter™ Layer 2B | ❌ | ✅ **EXCLUSIVE** |
| Link-16, ASTERIX output | ❌ | ✅ |
| DO-254 / ISO 26262 docs | ❌ | ✅ |
| **License** | MIT | Commercial |
| **Price** | Free | $25K-$350K |

---

## 🔧 How IMM Works

```
                     ┌─────────────┐
    Measurement ────►│   CV Model  │────┐
         │           └─────────────┘    │
         │           ┌─────────────┐    │     ┌──────────────┐
         ├──────────►│   CA Model  │────┼────►│   Weighted   │────► State
         │           └─────────────┘    │     │   Estimate   │      Estimate
         │           ┌─────────────┐    │     └──────────────┘
         ├──────────►│   CT Model  │────┤
         │           └─────────────┘    │
         │           ┌─────────────┐    │
         └──────────►│  Jerk Model │────┘
                     └─────────────┘
                     
Each model computes likelihood. Models with better predictions get higher weights.
```

---

## 📚 Use Cases

- 🛩️ **Air Traffic Control** — Commercial and military aircraft
- 🚀 **Missile Defense** — Ballistic and cruise missiles
- 🚁 **Drone Tracking** — Counter-UAS systems
- 🚗 **Autonomous Vehicles** — Sensor fusion, pedestrian tracking
- 🤖 **Robotics** — Dynamic obstacle avoidance
- 📊 **Sports Analytics** — Player and ball tracking

---

## ⚖️ Licensing

### MIT License (Free)

QEDMMA-Lite is released under the **MIT License** — use it freely in commercial or open-source projects.

### Commercial License (QEDMMA-PRO)

For FPGA deployment, defense applications, or certification support:

**Contact:** [mladen@nexellum.com](mailto:mladen@nexellum.com)

---

## 🏆 Why Not FilterPy / Stone Soup?

| | QEDMMA | FilterPy | Stone Soup |
|:--|:------:|:--------:|:----------:|
| **4-model IMM** | ✅ Built-in | ❌ Manual | ✅ Available |
| **Setup time** | 5 minutes | 1+ hours | 1+ days |
| **Maneuvering performance** | ✅ +44% | Baseline | ✅ Good |
| **FPGA ready** | ✅ PRO | ❌ No | ❌ No |
| **Code complexity** | ~900 lines | ~2000 lines | ~100K lines |

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
  title = {QEDMMA: Multi-Model IMM Tracking Library},
  year = {2026},
  url = {https://github.com/mladen1312/qedmma-lite}
}
```

---

## ⚠️ Limitations

- **Linear motion**: Single-model EKF may have lower overhead for purely constant-velocity targets
- **High noise**: Very high measurement noise (>150m σ) may require tuning
- **Real-time**: Python implementation is ~10x slower than FPGA; for real-time use consider QEDMMA-PRO

---

*Built with 🔬 by [Dr. Mladen Mešter](mailto:mladen@nexellum.com) | [Nexellum d.o.o.](https://nexellum.com)*
