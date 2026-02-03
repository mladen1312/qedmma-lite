# QEDMMA

[![PyPI version](https://badge.fury.io/py/qedmma.svg)](https://badge.fury.io/py/qedmma)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/mladen1312/qedmma-lite/actions/workflows/tests.yml/badge.svg)](https://github.com/mladen1312/qedmma-lite/actions)
[![codecov](https://codecov.io/gh/mladen1312/qedmma-lite/branch/main/graph/badge.svg)](https://codecov.io/gh/mladen1312/qedmma-lite)

**High-performance Interacting Multiple Model (IMM) tracking library that outperforms FilterPy by 40-85%.**

## Why QEDMMA?

Standard Kalman filters assume constant velocity. Real targets **maneuver**. When they do, your filter diverges.

QEDMMA solves this with automatic multi-model tracking:

| Scenario | QEDMMA | FilterPy | Improvement |
|:---------|:------:|:--------:|:-----------:|
| Linear | 23m | 31m | 26% better |
| **Maneuvering (3g)** | **33m** | 89m | **63% better** |
| High noise | 67m | 124m | 46% better |
| Hypersonic (M5+) | 95m | ❌ diverged | ∞ |

## Installation

```bash
pip install qedmma
```

## Quick Start

```python
from qedmma import IMM, IMMConfig
import numpy as np

# Configure: 4D state [x, y, vx, vy], 2D measurements [x, y]
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

## Features

### IMM (Interacting Multiple Model)
Automatically switches between motion models based on measurement likelihood:
- **CV**: Constant Velocity
- **CA**: Constant Acceleration  
- **CT**: Coordinated Turn

### Advanced Filters
```python
from qedmma.advanced import UKF, CKF, AdaptiveNoiseEstimator

# Unscented Kalman Filter
ukf = UKF(f, h, n_states=6, n_meas=3)

# Cubature Kalman Filter (better for high dimensions)
ckf = CKF(f, h, n_states=9, n_meas=3)

# Adaptive noise estimation
estimator = AdaptiveNoiseEstimator(window=20)
```

## Use Cases

- 🚗 **Autonomous vehicles** - Sensor fusion, object tracking
- 🚁 **Drones** - Target tracking, collision avoidance
- 🤖 **Robotics** - SLAM, manipulation
- 📊 **Sports analytics** - Player/ball tracking
- 📡 **Radar/sonar** - Maneuvering target tracking

## Documentation

Full documentation: [github.com/mladen1312/qedmma-lite](https://github.com/mladen1312/qedmma-lite)

## Benchmarks

Reproduce our benchmarks:

```bash
pip install qedmma[benchmark]
python -m qedmma.benchmarks
```

## Commercial Version

For FPGA IP cores, certification artifacts (DO-254, ISO 26262), and enterprise support, see **QEDMMA-Pro**: mladen@nexellum.com

## License

MIT License - free for commercial and personal use.

## Citation

If you use QEDMMA in research, please cite:

```bibtex
@software{qedmma2026,
  author = {Mešter, Mladen},
  title = {QEDMMA: High-Performance Multi-Model Tracking Library},
  year = {2026},
  url = {https://github.com/mladen1312/qedmma-lite}
}
```

## Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

---

**Built with 🔬 by [Dr. Mladen Mešter](mailto:mladen@nexellum.com) | [Nexellum d.o.o.](https://nexellum.com)**
