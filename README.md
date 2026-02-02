# QEDMMA-Lite

<div align="center">

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://github.com/mladen1312/qedmma-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/mladen1312/qedmma-lite/actions/workflows/ci.yml)
[![PRO Version](https://img.shields.io/badge/PRO-Enterprise-gold)](mailto:mladen@nexellum.com)

**Stop using single-model Kalman Filters for maneuvering targets.**

*Open-source multi-model tracking that outperforms standard EKF by 70%+*

[Quick Start](#-quick-start) • [Benchmark](#-benchmark-results) • [Why Upgrade?](#-why-upgrade-to-pro) • [Contact](#-contact)

</div>

---

## 🎯 The Problem

Standard Extended Kalman Filters (EKF) assume **constant velocity**. When targets maneuver:
- Track lag increases exponentially
- Error spikes during turns
- Complete track loss on aggressive maneuvers

**QEDMMA-Lite uses Interacting Multiple Model (IMM)** with adaptive mode switching between:
- Constant Velocity (CV)
- Constant Acceleration (CA)  
- Coordinated Turn (CT)

---

## 📊 Benchmark Results

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                    QEDMMA-Lite vs Standard EKF                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║  SCENARIO          │ Standard EKF  │ QEDMMA-Lite  │ Improvement            ║
╠════════════════════╪═══════════════╪══════════════╪════════════════════════╣
║  Fighter Aircraft  │    123.0 m    │    32.8 m    │  ▼ 73.3%               ║
║  Cruise Missile    │    150.1 m    │    40.7 m    │  ▼ 72.9%               ║
║  Hypersonic (M5+)  │    654.3 m    │    94.7 m    │  ▼ 85.5%               ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

Run the benchmark yourself:
```bash
python benchmark.py fighter
python benchmark.py cruise_missile
python benchmark.py hypersonic
```

---

## 🚀 Quick Start

### Installation
```bash
# Clone
git clone https://github.com/mladen1312/qedmma-lite.git
cd qedmma-lite

# Install
pip install -r requirements.txt

# Run demo
python benchmark.py fighter --plot
```

### Basic Usage
```python
from qedmma.trackers import IMMTracker

# Initialize
tracker = IMMTracker(dt=0.1)

# Track loop
for measurement in measurements:
    tracker.predict()
    estimate = tracker.update(measurement)
    print(f"Position: {estimate[:2]}")
```

---

## 📦 What's Included

| Component | Description | Location |
|-----------|-------------|----------|
| **IMM Tracker** | Interacting Multiple Model with CV/CA/CT | `python/qedmma/trackers/` |
| **UKF** | Unscented Kalman Filter for nonlinear | `python/qedmma/advanced/ukf.py` |
| **CKF** | Cubature Kalman Filter for high dimensions | `python/qedmma/advanced/ckf.py` |
| **Adaptive Noise** | Real-time Q/R estimation | `python/qedmma/advanced/adaptive_noise.py` |
| **Zero-DSP Correlation** | FPGA-optimized (0 DSP blocks) | `fpga/rtl/`, `fpga/hls/` |
| **Benchmark Suite** | Compare against EKF | `benchmark.py` |

---

## ⚖️ Lite vs PRO Comparison

| Feature | QEDMMA-Lite | QEDMMA-PRO |
|---------|:-----------:|:----------:|
| **Tracking Algorithm** | IMM (CV/CA/CT) | **Quantum-Evolutionary v6.1** |
| **Position RMSE** | | |
| └─ Fighter | 32.8 m | **< 15 m** |
| └─ Cruise Missile | 40.7 m | **< 20 m** |
| └─ Hypersonic (M5+) | 94.7 m | **< 50 m** |
| **Anomaly Detection** | ❌ | ✅ **Physics-Agnostic Layer 2B** |
| **FPGA Support** | Zero-DSP Correlator only | **Full Bitstream (RFSoC 4x2)** |
| **Multi-Static Fusion** | Requires sync | **Asynchronous (Bias-Compensated)** |
| **Real-time Clutter** | Basic CFAR | **AI-Adaptive (Neural CFAR)** |
| **TDOA Localization** | 2 nodes | **6+ nodes (Hyperbolic)** |
| **License** | **AGPL-3.0** | **Commercial** |
| **Support** | Community (GitHub Issues) | **Priority + SLA** |
| **Price** | Free | Contact for quote |

---

## ⚠️ Why AGPL License?

QEDMMA-Lite uses **AGPL-3.0** (GNU Affero General Public License).

**What this means:**
- ✅ **Free** for personal, academic, and open-source use
- ✅ **Modify** the code freely
- ⚠️ **If you deploy** this in a product/service, you **must open-source your entire codebase**

**For commercial use without open-sourcing:**
→ Contact us for a **Commercial License** (included with QEDMMA-PRO)

---

## 🚀 Why Upgrade to PRO?

<table>
<tr>
<td width="50%">

### QEDMMA-Lite Limitations

- ❌ No physics-agnostic mode (Layer 2B)
- ❌ RMSE floor ~30m (can't go lower)
- ❌ Manual noise tuning required
- ❌ No FPGA bitstreams
- ❌ AGPL restrictions for commercial use

</td>
<td width="50%">

### QEDMMA-PRO Advantages

- ✅ **<50m RMSE on hypersonic** targets
- ✅ Physics-agnostic anomaly detection
- ✅ Plug-and-play FPGA (RFSoC 4x2)
- ✅ Commercial license included
- ✅ Priority support + SLA
- ✅ Source code escrow

</td>
</tr>
</table>

**Ideal for:**
- Defense contractors (Raytheon, Thales, BAE)
- Aerospace (Airbus, Boeing, Lockheed)
- Academic research with commercialization path
- Startups building radar/tracking products

---

## 📧 Contact

**For QEDMMA-PRO licensing and inquiries:**

| | |
|---|---|
| 📧 Email | [mladen@nexellum.com](mailto:mladen@nexellum.com) |
| 🌐 Web | [www.nexellum.com](https://www.nexellum.com) |
| 📱 Phone | +385 99 737 5100 |
| 💼 LinkedIn | [Dr. Mladen Mešter](https://www.linkedin.com/in/mladen-mester/) |

---

## 📚 Citation

If you use QEDMMA-Lite in academic work:

```bibtex
@software{qedmma_lite,
  author = {Mešter, Mladen},
  title = {QEDMMA-Lite: Open-Source Multi-Model Radar Tracking},
  year = {2026},
  url = {https://github.com/mladen1312/qedmma-lite},
  license = {AGPL-3.0}
}
```

---

## 🤝 Contributing

Contributions welcome under AGPL-3.0 terms:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

<div align="center">

**Built with 🔬 by [Nexellum](https://www.nexellum.com)**

*Advancing radar technology through open innovation*

</div>
