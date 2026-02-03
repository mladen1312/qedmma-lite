"""
QEDMMA v3.1 — Motion Models
===========================

CV (Constant Velocity) and CT (Coordinated Turn) models
for IMM filter bank.

Author: Dr. Mladen Mešter / Nexellum d.o.o.
"""

import numpy as np
from typing import Tuple


def build_cv_model(dt: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Constant Velocity (CV) model matrices.
    
    State: [x, y, vx, vy]
    
    Args:
        dt: Sample period [s]
        q: Process noise intensity
    
    Returns:
        F: State transition matrix (4x4)
        Q: Process noise matrix (4x4)
    """
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    Q = q * np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ], dtype=np.float64)
    
    return F, Q


def build_ct_model(dt: float, omega: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Coordinated Turn (CT) model matrices.
    
    State: [x, y, vx, vy]
    Turn rate omega is assumed known/estimated.
    
    Args:
        dt: Sample period [s]
        omega: Turn rate [rad/s], positive=left, negative=right
        q: Process noise intensity
    
    Returns:
        F: State transition matrix (4x4)
        Q: Process noise matrix (4x4)
    """
    omega = max(abs(omega), 0.001) * np.sign(omega) if omega != 0 else 0.001
    
    cw = np.cos(omega * dt)
    sw = np.sin(omega * dt)
    
    F = np.array([
        [1, 0, sw/omega, -(1-cw)/omega],
        [0, 1, (1-cw)/omega, sw/omega],
        [0, 0, cw, -sw],
        [0, 0, sw, cw]
    ], dtype=np.float64)
    
    Q = q * np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ], dtype=np.float64)
    
    return F, Q


def omega_from_g_load(g: float, velocity: float) -> float:
    """
    Compute turn rate from g-load and velocity.
    
    omega = g * 9.81 / v
    
    Args:
        g: G-load (e.g., 6.0 for 6g turn)
        velocity: Target velocity [m/s]
    
    Returns:
        omega: Turn rate [rad/s]
    """
    return 9.81 * g / max(velocity, 1.0)
