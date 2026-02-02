"""
QEDMMA-Lite: Quantum-Enhanced Dynamically-Managed Multi-Model Algorithm

Open Source Edition for radar target tracking research and development.

Features:
- 4-model IMM Filter (CV, CA, CT, Jerk)
- Basic TDOA solver
- Hypersonic target optimization

For production deployment with:
- FPGA IP cores
- Anomaly Hunter™ (physics-agnostic tracking)
- Async multi-static fusion

Visit: https://www.mester-labs.com

MIT License - Copyright (c) 2026 Dr. Mladen Mešter
"""

from .tracker import (
    QEDMMATracker,
    TrackState,
    Measurement,
    IMMFilter,
    KalmanFilter,
    TDOASolver,
    # Kinematic models
    ConstantVelocity,
    ConstantAcceleration,
    CoordinatedTurn,
    ConstantJerk,
    # Utilities
    create_tracker,
    create_measurement,
    version,
    info,
)

__version__ = "1.0.0"
__author__ = "Dr. Mladen Mešter"
__email__ = "info@mester-labs.com"
__license__ = "MIT"
__url__ = "https://github.com/mladen1312/qedmma-lite"

__all__ = [
    # Main classes
    "QEDMMATracker",
    "TrackState",
    "Measurement",
    "IMMFilter",
    "KalmanFilter",
    "TDOASolver",
    # Models
    "ConstantVelocity",
    "ConstantAcceleration",
    "CoordinatedTurn",
    "ConstantJerk",
    # Utilities
    "create_tracker",
    "create_measurement",
    "version",
    "info",
]


def get_commercial_info():
    """Information about commercial QEDMMA-Pro products"""
    return """
╔═══════════════════════════════════════════════════════════════════════════╗
║                         QEDMMA-Pro™ Commercial Suite                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  Upgrade to QEDMMA-Pro for production-ready capabilities:                 ║
║                                                                           ║
║  ✅ FPGA IP Core - Synthesizable RTL for RFSoC/Zynq                       ║
║     Real-time tracking with < 1 μs latency                                ║
║                                                                           ║
║  ✅ Anomaly Hunter™ - Physics-agnostic tracking                           ║
║     Track targets that violate classical physics models                   ║
║                                                                           ║
║  ✅ Async Fusion Engine - Clock-bias estimation                           ║
║     Operate radar networks without expensive synchronization              ║
║                                                                           ║
║  ✅ DO-254 Support - Certification-ready documentation                    ║
║     Full traceability matrix and compliance artifacts                     ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Contact: sales@mester-labs.com | www.mester-labs.com                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
