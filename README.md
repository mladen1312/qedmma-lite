# QEDMMA-Lite v3.1.0

**Quantum-Enhanced Dynamically-Managed Multi-Model Algorithm**

Production-grade IMM (Interacting Multiple Model) tracker with **True IMM Smoother** achieving **+48% RMSE improvement** on maneuvering targets.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Installation

```bash
pip install qedmma-lite
```

## Quick Start

```python
from qedmma import QEDMMAv31, QEDMMAConfig, TrackingMode
import numpy as np

# Configure for 6g maneuvering target at 300 m/s
cfg = QEDMMAConfig(
    dt=0.1,                           # 10 Hz
    omega=9.81 * 6.0 / 300.0,         # Turn rate for 6g
    mode=TrackingMode.FULL_SMOOTH     # Enable smoothing
)

# Create tracker and process measurements
tracker = QEDMMAv31(cfg)
x_filt, x_smooth = tracker.process_batch(measurements)

# x_filt: Forward filter estimates (real-time compatible)
# x_smooth: Smoothed estimates (+48% accuracy improvement)
```

## Performance

Validated on maneuvering target scenarios:

| Scenario | Filter RMSE | Smoother RMSE | Improvement |
|----------|-------------|---------------|-------------|
| 3g Turn | 1.81m | 0.88m | **+51.1%** |
| 6g Turn | 1.91m | 0.98m | **+48.5%** |
| 9g Turn | 1.95m | 1.02m | **+47.7%** |

## Key Innovation: True IMM Smoother

Standard RTS smoothing on combined IMM state fails because it mixes incompatible model dynamics. QEDMMA v3.1 implements **per-model RTS smoothing**:

1. Smooth each model (CV, CT+, CT-) independently
2. Use stored predictions from forward pass (F @ x_mixed)
3. Combine with forward mode probabilities

This achieves consistent +48% improvement across all tested scenarios.

## Applications

- **Missile Guidance:** Terminal seeker tracking
- **Fire Control Radar:** Track-while-scan
- **Air Traffic Control:** Maneuvering aircraft
- **Maritime Surveillance:** Ship tracking

## License

AGPL-3.0 (open source) — Commercial licenses available for defense applications.

Contact: mladen@nexellum.com | +385 99 737 5100

---

*Nexellum d.o.o. — Advanced Defense Technology Systems*
