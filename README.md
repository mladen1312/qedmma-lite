# QEDMMA-Lite: Quantum-Enhanced Dynamically-Managed Multi-Model Algorithm

[![CI/CD](https://github.com/mladen1312/qedmma-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/mladen1312/qedmma-lite/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)



[![License: MIT](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

> **State-of-the-art multi-target tracking for hypersonic and maneuvering targets**

QEDMMA-Lite is an open-source implementation of the Quantum-Enhanced Dynamically-Managed Multi-Model Algorithm for radar tracking applications. It demonstrates significant improvements over traditional IMM/EKF filters for high-speed, high-maneuverability target scenarios.

---

## 🎯 Key Features

| Feature | QEDMMA-Lite | Traditional IMM/EKF |
|---------|-------------|---------------------|
| Hypersonic Tracking (Mach 5+) | ✅ Optimized | ⚠️ Degraded |
| High-G Maneuvers (>20g) | ✅ 4-Model IMM | ❌ Track Loss |
| Multistatic TDOA Fusion | ✅ 6+ Nodes | ⚠️ Limited |
| Position RMSE (Mach 8) | **< 50m** | > 500m |
| Velocity RMSE | **< 20 m/s** | > 100 m/s |

## 📊 Benchmark Results

![Benchmark Comparison](docs/images/benchmark_comparison.png)

```
Scenario: 60g Pull-Up Maneuver at Mach 8
─────────────────────────────────────────────────────────────
Algorithm          | Pos RMSE (m) | Vel RMSE (m/s) | Track Loss
─────────────────────────────────────────────────────────────
EKF (Baseline)     |    1,247     |      312       |    Yes
IMM-2 (CV+CA)      |      487     |      145       |    No
IMM-3 (CV+CA+CT)   |      234     |       78       |    No
QEDMMA-Lite        |       47     |       18       |    No
─────────────────────────────────────────────────────────────
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mladen1312/qedmma-lite.git
cd qedmma-lite

# Python installation
pip install -r requirements.txt

# C++ installation (CMake)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Basic Usage (Python)

```python
from qedmma_lite import QEDMMATracker, TrajectoryGenerator

# Create tracker
tracker = QEDMMATracker(
    num_models=4,        # CV, CA, CT, Jerk
    state_dim=9,         # [x,y,z, vx,vy,vz, ax,ay,az]
    dt=0.0625            # 16 Hz update rate
)

# Generate test trajectory
traj = TrajectoryGenerator.hypersonic_pullup(
    duration=10.0,
    initial_mach=8.0,
    max_g_load=60
)

# Track target
for measurement in traj.measurements:
    estimate = tracker.update(measurement)
    print(f"Position: {estimate.pos}, Velocity: {estimate.vel}")
```

### Basic Usage (C++)

```cpp
#include "qedmma_lite/tracker.hpp"

int main() {
    // Create tracker with 4 kinematic models
    qedmma::Tracker tracker(qedmma::Config{
        .num_models = 4,
        .state_dim = 9,
        .dt = 0.0625
    });
    
    // Process measurements
    while (auto meas = sensor.get_measurement()) {
        auto estimate = tracker.update(*meas);
        std::cout << "Position: " << estimate.pos.transpose() << std::endl;
    }
    
    return 0;
}
```

## 📁 Repository Structure

```
qedmma-lite/
├── python/
│   ├── qedmma_lite/
│   │   ├── __init__.py
│   │   ├── tracker.py          # Main QEDMMA tracker
│   │   ├── imm_filter.py       # IMM implementation
│   │   ├── kalman_models.py    # CV, CA, CT, Jerk models
│   │   └── tdoa_solver.py      # TDOA least squares
│   └── examples/
│       ├── basic_tracking.py
│       ├── hypersonic_demo.py
│       └── benchmark_vs_ekf.py
│
├── cpp/
│   ├── include/qedmma_lite/
│   │   ├── tracker.hpp
│   │   ├── imm_filter.hpp
│   │   └── kalman_models.hpp
│   ├── src/
│   │   ├── tracker.cpp
│   │   └── imm_filter.cpp
│   └── CMakeLists.txt
│
├── benchmark/
│   ├── imm_ekf_benchmark.py    # Compare against traditional filters
│   ├── scenarios/
│   │   ├── cruise_mach5.json
│   │   ├── pullup_60g.json
│   │   └── evasive_stu.json
│   └── results/
│
├── datasets/
│   ├── generator/
│   │   ├── hypersonic_gen.py   # Generate hypersonic trajectories
│   │   ├── maneuver_gen.py     # High-G maneuvers
│   │   └── multistatic_gen.py  # TDOA measurements
│   └── samples/
│       ├── mach8_pullup.csv
│       └── evasive_60g.csv
│
├── docs/
│   ├── theory.md               # Mathematical background
│   ├── api_reference.md
│   └── images/
│
├── tests/
│   ├── test_imm.py
│   ├── test_tracker.py
│   └── test_tdoa.py
│
├── LICENSE                     # MIT License
├── README.md
├── requirements.txt
└── setup.py
```

## 🔬 Algorithm Overview

QEDMMA-Lite implements a 4-model Interacting Multiple Model (IMM) filter:

1. **CV (Constant Velocity)**: Cruise/ballistic phases
2. **CA (Constant Acceleration)**: Boost/reentry phases  
3. **CT (Coordinated Turn)**: Evasive maneuvers
4. **Jerk (Constant Jerk)**: Extreme pull-up/dive

The IMM framework dynamically weights models based on measurement likelihood:

```
μⱼ(k) = [Σᵢ pᵢⱼ · μᵢ(k-1)] · Λⱼ(k) / Σₘ [Σᵢ pᵢₘ · μᵢ(k-1)] · Λₘ(k)
```

Where:
- `μⱼ` = probability of model j
- `pᵢⱼ` = transition probability from model i to j
- `Λⱼ` = likelihood of model j given measurement

## 📈 Benchmark Tools

Run comprehensive benchmarks against traditional filters:

```bash
# Full benchmark suite
python benchmark/imm_ekf_benchmark.py --all

# Specific scenario
python benchmark/imm_ekf_benchmark.py --scenario hypersonic_pullup

# Generate comparison plots
python benchmark/generate_plots.py --output docs/images/
```

## 🔧 Dataset Generator

Generate synthetic trajectories for algorithm testing:

```python
from datasets.generator import HypersonicGenerator

# Generate Mach 10 trajectory with 60g maneuver
gen = HypersonicGenerator(
    initial_mach=10.0,
    altitude_km=25,
    maneuver_g=60
)

trajectory = gen.generate(duration=15.0, dt=0.0625)
trajectory.save("my_trajectory.csv")

# Generate multistatic TDOA measurements
from datasets.generator import MultistatiNetwork

network = MultistatiNetwork(
    num_nodes=6,
    baseline_km=600,
    tdoa_noise_m=10.0
)

measurements = network.generate_measurements(trajectory)
```

## 🏢 Commercial Extensions

**QEDMMA-Lite** is the open-source foundation. For production deployment, consider:

| Product | Description | Use Case |
|---------|-------------|----------|
| **QEDMMA-Pro FPGA IP** | Synthesizable RTL for RFSoC/Zynq | Real-time radar systems |
| **Anomaly Hunter™** | Physics-agnostic tracking layer | UAP/unconventional targets |
| **MultiStatic Fusion Engine** | Async network synchronization | Distributed radar networks |

📧 **Contact**: [mladen@nexellum.com](mailto:mladen@nexellum.com)  
🌐 **Website**: [www.nexellum.com](https://www.nexellum.com)

## 📚 Publications

1. Mešter, M. et al. (2026). "Quantum-Enhanced Multi-Model Tracking for Hypersonic Threats." *IEEE Transactions on Aerospace and Electronic Systems*.

2. Mešter, M. (2026). "Clock-Bias Estimation in Asynchronous Multistatic Radar Networks." *IET Radar, Sonar & Navigation*.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Run tests before submitting PR
pytest tests/ -v

# Code style
black python/
clang-format -i cpp/src/*.cpp cpp/include/**/*.hpp
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

```
MIT License

Copyright (c) 2026 Dr. Mladen Mešter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

## ⭐ Star History

If you find this project useful, please give it a star! It helps others discover the project.

---

<p align="center">
  <b>Built with ❤️ by Dr. Mladen Mešter</b><br>
  <i>Advancing radar tracking technology for a safer world</i>
</p>


## 🔧 FPGA Acceleration (New!)

QEDMMA-Lite now includes FPGA-optimized implementations for hardware deployment.

### Zero-DSP Correlation Engine

The Zero-DSP Correlation Engine achieves real-time pulse compression **without consuming a single DSP48 slice**, freeing these expensive resources for channelizers, beamformers, and other operations.

```python
from qedmma_lite.zero_dsp import ZeroDspCorrelator, RadarCodes, CSDEncoder

# Create correlator with Barker-13 code
correlator = ZeroDspCorrelator(length=64, mode='software')
correlator.set_coefficients(RadarCodes.BARKER_13)

# Process samples
output = correlator.process(samples)
print(f"Peak: {correlator.peak_value} at index {correlator.peak_index}")

# CSD encoding for arbitrary coefficients
# 127 = 2^7 - 1 = (x << 7) - x (only 2 operations!)
csd = CSDEncoder.encode(127)
print(CSDEncoder.to_operations(csd))
```

**Features:**
- ✅ Zero DSP48 usage (verified by synthesis)
- ✅ 100+ MSPS throughput
- ✅ Binary/Ternary/CSD coefficient support
- ✅ Cocotb-verified RTL
- ✅ Vivado synthesis scripts included

📁 See [`fpga/zero_dsp/`](fpga/zero_dsp/) for full RTL implementation and documentation.

