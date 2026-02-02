# Changelog

All notable changes to QEDMMA-Lite will be documented in this file.

## [3.0.0] - 2026-02-02

### 🔐 License Change
- **MIT → AGPL-3.0-or-later**: Enterprise users must open-source derivatives or purchase commercial license
- Commercial licensing available via Nexellum (mladen@nexellum.com)

### 🚀 Added - Advanced Filters

#### Unscented Kalman Filter (UKF)
- Sigma-point based nonlinear state estimation
- Van der Merwe scaled unscented transform
- Configurable α, β, κ parameters
- Radar tracking preset (`create_radar_ukf()`)

#### Cubature Kalman Filter (CKF)
- Spherical-radial cubature rule
- No tuning parameters required
- Superior high-dimensional performance (n > 3)
- Square-root variant (SR-CKF) for numerical stability

#### Adaptive Noise Estimation
- Innovation-based R estimation
- Covariance matching for Q and R
- Sage-Husa recursive estimator with forgetting factor
- Composite estimator with voting

#### Zero-DSP Correlation
- 1-bit XOR + popcount correlation
- 2-bit LUT-based correlation
- HLS C++ implementation for Vitis
- VHDL implementation for direct synthesis
- 0 DSP blocks, pure LUT logic
- >1 GHz achievable on UltraScale+

### 📁 New Files
- `python/qedmma/advanced/ukf.py`
- `python/qedmma/advanced/ckf.py`
- `python/qedmma/advanced/adaptive_noise.py`
- `python/qedmma/advanced/zero_dsp_correlation.py`
- `fpga/hls/zero_dsp_correlator.cpp`
- `fpga/rtl/zero_dsp_correlator.vhd`
- `tests/test_advanced_filters.py`

### ✅ Tests
- 12 unit tests covering all new modules
- Integration test for complete tracking pipeline

## [2.0.0] - 2026-01-15
- Initial public release with EKF, IMM, JPDA
- Multi-target tracking framework
- Basic FPGA support

---

For commercial licensing inquiries:
- Email: mladen@nexellum.com
- Web: www.nexellum.com
- Phone: +385 99 737 5100
