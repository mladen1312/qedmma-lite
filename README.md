# QEDMMA Lite — Open Source IMM Tracker with True Smoother

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)]()
[![PyPI](https://img.shields.io/badge/PyPI-qedmma--lite-orange.svg)]()

**High-performance Interacting Multiple Model (IMM) tracker with the novel True IMM Smoother algorithm.**

---

## 🎯 What is QEDMMA?

QEDMMA (Quantum-Enhanced Dynamically-Managed Multi-Model Algorithm) is a state-of-the-art radar/sensor tracking algorithm that combines:

- **IMM (Interacting Multiple Model)** filter for maneuvering targets
- **True IMM Smoother** — per-model RTS smoothing for superior accuracy
- **Adaptive mode switching** between CV (constant velocity) and CT (coordinated turn) models

### The Innovation

Standard IMM smoothers fail because they smooth the combined state, mixing incompatible dynamics. Our **per-model RTS** approach smooths each model independently:

```python
# Per-model backward pass (the key insight!)
for model_j in models:
    G[j] = P_filt[j] @ F[j].T @ inv(P_pred[j])
    x_smooth[j] = x_filt[j] + G[j] @ (x_smooth[k+1] - x_pred[k+1])  # Uses stored predictions!

# Combine with forward mode probabilities
x_smooth = sum(mu[j] * x_smooth[j] for j in models)
```

---

## 📊 Performance

| Scenario | QEDMMA | Standard IMM | Improvement |
|----------|--------|--------------|-------------|
| High-g Maneuver (7g) | 1.44m | 3.24m | **+55%** |
| Jinking Target (6g) | 2.39m | 4.97m | **+52%** |
| Dogfight (8g) | 1.13m | 2.25m | **+50%** |

**Average improvement: +48% RMSE reduction** over standard IMM.

---

## 🚀 Installation

```bash
pip install qedmma-lite
```

Or from source:

```bash
git clone https://github.com/mladen1312/qedmma-lite.git
cd qedmma-lite
pip install -e .
```

---

## 📖 Quick Start

```python
import numpy as np
from qedmma import QEDMMATracker, SmoothingMode

# Create tracker
tracker = QEDMMATracker(
    dt=0.1,              # 10 Hz sample rate
    omega=0.196,         # Turn rate for 6g @ 300 m/s
    r=5.0,               # Measurement noise std (meters)
    smoothing_mode=SmoothingMode.FIXED_LAG
)

# Generate some measurements (replace with your data)
measurements = np.array([
    [100.0, 200.0],  # [x, y] at t=0
    [110.5, 198.2],  # [x, y] at t=0.1s
    [121.3, 195.8],  # ...
    # ... more measurements
])

# Run tracker
x_filtered, x_smoothed = tracker.run(measurements)

# x_smoothed has shape (N, 4): [x, y, vx, vy] for each timestep
print(f"Position: ({x_smoothed[-1, 0]:.1f}, {x_smoothed[-1, 1]:.1f})")
print(f"Velocity: ({x_smoothed[-1, 2]:.1f}, {x_smoothed[-1, 3]:.1f})")
```

---

## 🔧 API Reference

### QEDMMATracker

```python
QEDMMATracker(
    dt: float = 0.1,           # Sample period (seconds)
    omega: float = 0.196,      # Turn rate (rad/s)
    q_cv: float = 0.5,         # CV process noise
    q_ct: float = 1.0,         # CT process noise  
    r: float = 5.0,            # Measurement noise std
    p_stay: float = 0.88,      # Mode stay probability
    smoothing_mode: SmoothingMode = NONE
)
```

### SmoothingMode

| Mode | Description |
|------|-------------|
| `NONE` | Filter only (real-time) |
| `FULL` | Full RTS smoother (batch) |
| `FIXED_LAG` | Fixed-lag smoother (50 samples) |

### Methods

| Method | Description |
|--------|-------------|
| `run(measurements)` | Process all measurements, return (filtered, smoothed) |
| `predict()` | Predict next state |
| `update(z)` | Update with measurement |
| `get_state()` | Get current state estimate |
| `get_mode_probabilities()` | Get [μ_CV, μ_CT+, μ_CT-] |

---

## 🧮 Mathematical Background

### State Vector

```
x = [x, y, vx, vy]ᵀ
```

### Motion Models

**Constant Velocity (CV):**
```
F_cv = [1  0  dt  0 ]
       [0  1  0   dt]
       [0  0  1   0 ]
       [0  0  0   1 ]
```

**Coordinated Turn (CT±):**
```
F_ct = [1  0  sin(ωdt)/ω    -(1-cos(ωdt))/ω]
       [0  1  (1-cos(ωdt))/ω  sin(ωdt)/ω   ]
       [0  0  cos(ωdt)        -sin(ωdt)     ]
       [0  0  sin(ωdt)         cos(ωdt)     ]
```

### True IMM Smoother

The key insight is storing predictions from the forward pass and using them in the backward pass:

```
Forward: x_pred[k] = F @ x_mixed[k-1]  # Store this!

Backward (per model):
  G = P_filt @ F.T @ inv(P_pred)
  x_smooth[k] = x_filt[k] + G @ (x_smooth[k+1] - x_pred[k+1])
                                              ↑
                                    Use stored prediction, NOT F @ x_filt!
```

---

## 📁 Project Structure

```
qedmma-lite/
├── qedmma/
│   ├── __init__.py      # Package exports
│   ├── trackers.py      # QEDMMATracker implementation
│   ├── smoother.py      # IMMSmoother class
│   └── models.py        # Motion model definitions
├── tests/
│   └── test_tracker.py  # pytest test suite
├── examples/
│   └── basic_tracking.py
├── pyproject.toml
├── LICENSE              # AGPL v3
└── README.md
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📜 License

**AGPL v3** — You may use this software freely, but any modifications or derivative works must also be released under AGPL v3.

For commercial licensing without AGPL obligations, see [QEDMMA Pro](https://github.com/mladen1312/qedmma-pro).

---

## 🏢 Commercial Version

**QEDMMA Pro** includes:

| Feature | Lite (AGPL) | Pro (Commercial) |
|---------|-------------|------------------|
| Python Implementation | ✅ | ✅ |
| True IMM Smoother | ✅ | ✅ |
| FPGA RTL (SystemVerilog) | ❌ | ✅ |
| Multi-target tracking | ❌ | ✅ |
| DO-254 certification package | ❌ | ✅ |
| Technical support | Community | Dedicated |
| Commercial deployment | AGPL terms | Unrestricted |

**Pricing:** $15K (Dev) / $50K (Production) / $150K (Enterprise)

→ [Learn more about QEDMMA Pro](https://github.com/mladen1312/qedmma-pro)

---

## 📞 Contact

**Nexellum d.o.o.**

| | |
|---|---|
| **Author** | Dr. Mladen Mešter |
| **Email** | mladen@nexellum.com |
| **Phone** | +385 99 737 5100 |

---

## 📚 Citation

If you use QEDMMA in academic work, please cite:

```bibtex
@software{qedmma2026,
  author = {Mešter, Mladen},
  title = {QEDMMA: True IMM Smoother for Maneuvering Target Tracking},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/mladen1312/qedmma-lite}
}
```

---

<p align="center">
  <b>QEDMMA Lite — Open Source Precision Tracking</b>
  <br>
  © 2026 Nexellum d.o.o. | AGPL v3
</p>
