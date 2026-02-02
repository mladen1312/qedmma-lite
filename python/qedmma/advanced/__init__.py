"""
QEDMMA-Lite v3.0 - Advanced Filters
====================================
Copyright (C) 2026 Dr. Mladen Mešter / Nexellum
License: AGPL-3.0-or-later

For commercial licensing: mladen@nexellum.com

Advanced filter algorithms for radar tracking:
- UKF: Unscented Kalman Filter
- CKF: Cubature Kalman Filter  
- Adaptive Noise: Real-time Q/R estimation
- Zero-DSP: FPGA-optimized correlation
"""

from .ukf import (
    UnscentedKalmanFilter,
    UKFState,
    UKFParams,
    create_radar_ukf
)

from .ckf import (
    CubatureKalmanFilter,
    CKFState,
    SquareRootCKF,
    create_high_dim_ckf
)

from .adaptive_noise import (
    AdaptiveNoiseState,
    InnovationBasedEstimator,
    CovarianceMatchingEstimator,
    SageHusaEstimator,
    MLENoiseEstimator,
    CompositeAdaptiveEstimator
)

from .zero_dsp_correlation import (
    ZeroDSPCorrelator,
    TwoBitCorrelator,
    ParallelZeroDSPCorrelator,
    ZeroDSPConfig
)

__version__ = "3.0.0"
__author__ = "Dr. Mladen Mešter"
__email__ = "mladen@nexellum.com"
__license__ = "AGPL-3.0-or-later"

__all__ = [
    # UKF
    'UnscentedKalmanFilter',
    'UKFState', 
    'UKFParams',
    'create_radar_ukf',
    
    # CKF
    'CubatureKalmanFilter',
    'CKFState',
    'SquareRootCKF',
    'create_high_dim_ckf',
    
    # Adaptive
    'AdaptiveNoiseState',
    'InnovationBasedEstimator',
    'CovarianceMatchingEstimator',
    'SageHusaEstimator',
    'MLENoiseEstimator',
    'CompositeAdaptiveEstimator',
    
    # Zero-DSP
    'ZeroDSPCorrelator',
    'TwoBitCorrelator',
    'ParallelZeroDSPCorrelator',
    'ZeroDSPConfig',
]
