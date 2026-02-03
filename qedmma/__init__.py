"""
QEDMMA — Quantum-Enhanced Dynamically-Managed Multi-Model Algorithm
====================================================================

Production-grade IMM tracking with True IMM Smoother.

Modules:
    trackers: QEDMMAv31 tracker with +48% smoother improvement
    smoother: IMMSmoother for per-model RTS smoothing
    models: CT/CV model builders

Example:
    >>> from qedmma import QEDMMAv31, QEDMMAConfig, TrackingMode
    >>> cfg = QEDMMAConfig(omega=0.196, mode=TrackingMode.FULL_SMOOTH)
    >>> tracker = QEDMMAv31(cfg)
    >>> x_filt, x_smooth = tracker.process_batch(measurements)

Author: Dr. Mladen Mešter / Nexellum d.o.o.
License: AGPL-3.0
Version: 3.1.0
"""

__version__ = "3.1.0"
__author__ = "Dr. Mladen Mešter"
__email__ = "mladen@nexellum.com"

from .trackers import QEDMMAv31, QEDMMAConfig, TrackingMode, track_maneuvering_target
from .smoother import IMMSmoother, compute_entropy_q_scale

__all__ = [
    "QEDMMAv31",
    "QEDMMAConfig", 
    "TrackingMode",
    "IMMSmoother",
    "track_maneuvering_target",
    "compute_entropy_q_scale",
]
