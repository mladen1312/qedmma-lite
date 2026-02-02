# Changelog

All notable changes to QEDMMA-Lite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation improvements
- Additional test coverage

## [1.0.0] - 2026-02-02

### Added

#### Core Tracking Algorithms
- **QEDMMA Tracker v1.0**: 4-model IMM Filter (CV, CA, CT, Jerk) for maneuvering targets
- **QEDMMA Tracker v2.0**: Enhanced with UKF/CKF sigma-point filters
- **Advanced Filters**: Unscented Kalman Filter (UKF), Cubature Kalman Filter (CKF)

#### Adaptive Noise Estimation
- **Mehra Estimator**: Innovation-based noise estimation
- **Sage-Husa Filter**: Exponential forgetting adaptive filter
- **Variational Bayesian**: Probabilistic noise estimation
- **Covariance Matching**: Windowed sample statistics
- **IMM-Adaptive**: Per-model noise adaptation

#### Zero-DSP FPGA Module
- **Zero-DSP Correlator**: DSP-free pulse compression (SystemVerilog RTL)
- **CSD Encoder**: Canonical Signed Digit coefficient encoding
- **Ternary Encoder**: Hardware coefficient format conversion
- **Radar Codes**: Barker, MLS sequence generators
- **Cocotb Testbench**: Complete verification suite
- **Vivado Scripts**: Synthesis automation for AMD/Xilinx FPGAs

#### CI/CD Pipeline
- GitHub Actions workflow with matrix testing (Python 3.9-3.12)
- Verilator RTL linting
- Cocotb simulation integration
- Pre-commit hooks (Ruff, Black, MyPy)
- Dependabot for dependency updates
- Issue and PR templates

### Documentation
- Comprehensive README with badges
- API documentation
- Usage examples
- FPGA integration guide

## [0.1.0] - 2026-01-15

### Added
- Initial release
- Basic IMM-EKF tracker
- Dataset generator for benchmarking

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2026-02-02 | Production release with adaptive noise, Zero-DSP, CI/CD |
| 0.1.0 | 2026-01-15 | Initial open-source release |

## Migration Guide

### From 0.1.x to 1.0.0

```python
# Old import
from qedmma_lite.tracker import QEDMMATracker

# New import (still works)
from qedmma_lite.tracker import QEDMMATracker

# New features available
from qedmma_lite import (
    ZeroDspCorrelator,        # FPGA correlation
    AdaptiveNoiseEstimator,   # Adaptive filtering
    AdaptiveKalmanFilter,     # Integrated adaptive KF
)
```

## Links

- [Repository](https://github.com/mladen1312/qedmma-lite)
- [PyPI Package](https://pypi.org/project/qedmma-lite/)
- [Commercial License](https://www.nexellum.com)
- [Issue Tracker](https://github.com/mladen1312/qedmma-lite/issues)

---

**© 2026 Dr. Mladen Mešter / Nexellum d.o.o.**
