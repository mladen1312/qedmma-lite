#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    FAIR IMM vs IMM BENCHMARK
═══════════════════════════════════════════════════════════════════════════════

This compares:
1. FilterPy EKF (single CV model) - baseline
2. FilterPy IMM (CV + CA) - fair comparison 
3. QEDMMA IMM (CV + CA + CT + Jerk) - our implementation

The REAL question: Does QEDMMA's 4-model IMM beat FilterPy's 2-model IMM?
"""

import numpy as np
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/claude/qedmma_code')

from filterpy.kalman import KalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise

try:
    from tracker import IMMFilter, Measurement
    QEDMMA_OK = True
except:
    QEDMMA_OK = False


@dataclass
class Scenario:
    name: str
    duration: int
    dt: float
    process_noise: float
    measurement_noise: float
    initial_state: np.ndarray
    maneuvers: List[Tuple[int, int, float]]


SCENARIOS = {
    "linear": Scenario("Linear", 200, 0.1, 1.0, 30.0, np.array([0., 0., 100., 50.]), []),
    "maneuvering": Scenario("Maneuvering", 300, 0.1, 5.0, 50.0, np.array([0., 0., 200., 0.]),
                           [(50, 90, 0.12), (130, 170, -0.10), (210, 250, 0.15)]),
    "aggressive": Scenario("Aggressive", 400, 0.05, 10.0, 80.0, np.array([0., 0., 300., 100.]),
                          [(40, 80, 0.15), (120, 160, -0.18), (200, 240, 0.20), (300, 350, -0.12)]),
}


def generate_trajectory(scenario: Scenario, seed: int = 42):
    np.random.seed(seed)
    n = scenario.duration
    dt = scenario.dt
    
    truth = np.zeros((n, 4))
    truth[0] = scenario.initial_state.copy()
    
    for k in range(1, n):
        x, y, vx, vy = truth[k-1]
        
        omega = 0.0
        for start, end, w in scenario.maneuvers:
            if start <= k < end:
                omega = w
                break
        
        if abs(omega) > 1e-6:
            cos_w, sin_w = np.cos(omega * dt), np.sin(omega * dt)
            vx_new = vx * cos_w - vy * sin_w
            vy_new = vx * sin_w + vy * cos_w
        else:
            vx_new, vy_new = vx, vy
        
        vx_new += np.random.randn() * scenario.process_noise
        vy_new += np.random.randn() * scenario.process_noise
        
        truth[k] = [x + vx * dt, y + vy * dt, vx_new, vy_new]
    
    meas = truth[:, :2] + np.random.randn(n, 2) * scenario.measurement_noise
    return truth, meas


def make_cv_filter(dt, q, r):
    """Create CV Kalman filter."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=q**2, block_size=2)
    kf.R = np.eye(2) * r**2
    kf.P = np.diag([500., 500., 100., 100.])
    return kf


def make_ca_filter(dt, q, r):
    """Create CA Kalman filter (higher Q for acceleration)."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=(q*5)**2, block_size=2)  # Higher Q!
    kf.R = np.eye(2) * r**2
    kf.P = np.diag([500., 500., 100., 100.])
    return kf


def run_filterpy_ekf(scenario, meas, truth):
    """Single-model EKF (CV)."""
    kf = make_cv_filter(scenario.dt, scenario.process_noise*2, scenario.measurement_noise)
    kf.x = np.array([[meas[0, 0]], [meas[0, 1]], 
                     [scenario.initial_state[2]], [scenario.initial_state[3]]])
    
    est = np.zeros((len(meas), 4))
    est[0] = kf.x.flatten()
    
    t0 = time.perf_counter()
    for k in range(1, len(meas)):
        kf.predict()
        kf.update(meas[k])
        est[k] = kf.x.flatten()
    elapsed = (time.perf_counter() - t0) * 1000
    
    err = np.sqrt(np.sum((est[:, :2] - truth[:, :2])**2, axis=1))
    return np.sqrt(np.mean(err**2)), elapsed


def run_filterpy_imm(scenario, meas, truth):
    """FilterPy IMM (CV + CA) - fair comparison."""
    dt = scenario.dt
    q = scenario.process_noise
    r = scenario.measurement_noise
    
    # Create two filters
    kf_cv = make_cv_filter(dt, q, r)
    kf_ca = make_ca_filter(dt, q, r)
    
    # Initialize
    x0 = np.array([[meas[0, 0]], [meas[0, 1]], 
                   [scenario.initial_state[2]], [scenario.initial_state[3]]])
    kf_cv.x = x0.copy()
    kf_ca.x = x0.copy()
    
    # IMM setup
    mu = np.array([0.7, 0.3])  # Favor CV initially
    M = np.array([[0.90, 0.10], [0.10, 0.90]])  # Transition matrix
    
    imm = IMMEstimator([kf_cv, kf_ca], mu, M)
    
    est = np.zeros((len(meas), 4))
    est[0] = imm.x.flatten()
    
    t0 = time.perf_counter()
    for k in range(1, len(meas)):
        imm.predict()
        imm.update(meas[k])
        est[k] = imm.x.flatten()
    elapsed = (time.perf_counter() - t0) * 1000
    
    err = np.sqrt(np.sum((est[:, :2] - truth[:, :2])**2, axis=1))
    return np.sqrt(np.mean(err**2)), elapsed


def run_qedmma_imm(scenario, meas, truth):
    """QEDMMA 4-model IMM."""
    if not QEDMMA_OK:
        return np.inf, 0
    
    imm = IMMFilter(dt=scenario.dt, num_models=4)
    
    x0 = np.array([meas[0, 0], meas[0, 1], 0,
                   scenario.initial_state[2], scenario.initial_state[3], 0,
                   0, 0, 0])
    imm.initialize(x0)
    
    est = np.zeros((len(meas), 4))
    est[0] = [x0[0], x0[1], x0[3], x0[4]]
    
    t0 = time.perf_counter()
    for k in range(1, len(meas)):
        imm.predict()
        m = Measurement(
            pos=np.array([meas[k, 0], meas[k, 1], 0]),
            noise_pos=scenario.measurement_noise,
            noise_vel=100.0,
            time=k * scenario.dt
        )
        state = imm.update(m)
        est[k] = [state.pos[0], state.pos[1], state.vel[0], state.vel[1]]
    elapsed = (time.perf_counter() - t0) * 1000
    
    err = np.sqrt(np.sum((est[:, :2] - truth[:, :2])**2, axis=1))
    return np.sqrt(np.mean(err**2)), elapsed


def main():
    print("\n" + "="*80)
    print("          🔬 FAIR BENCHMARK: FilterPy EKF vs FilterPy IMM vs QEDMMA IMM")
    print("="*80)
    
    n_runs = 10
    results = {}
    
    for name, scenario in SCENARIOS.items():
        print(f"\n{'─'*80}")
        print(f"📊 {scenario.name} (Maneuvers: {len(scenario.maneuvers)})")
        print(f"{'─'*80}")
        
        ekf_rmses, imm_rmses, qedmma_rmses = [], [], []
        ekf_times, imm_times, qedmma_times = [], [], []
        
        for seed in range(n_runs):
            truth, meas = generate_trajectory(scenario, seed=42+seed)
            
            rmse, t = run_filterpy_ekf(scenario, meas, truth)
            ekf_rmses.append(rmse); ekf_times.append(t)
            
            rmse, t = run_filterpy_imm(scenario, meas, truth)
            imm_rmses.append(rmse); imm_times.append(t)
            
            rmse, t = run_qedmma_imm(scenario, meas, truth)
            qedmma_rmses.append(rmse); qedmma_times.append(t)
        
        results[name] = {
            'EKF': (np.mean(ekf_rmses), np.mean(ekf_times)),
            'FilterPy IMM': (np.mean(imm_rmses), np.mean(imm_times)),
            'QEDMMA IMM': (np.mean(qedmma_rmses), np.mean(qedmma_times))
        }
        
        print(f"  FilterPy EKF (CV)    │ RMSE: {np.mean(ekf_rmses):7.1f}m │ Time: {np.mean(ekf_times):6.1f}ms")
        print(f"  FilterPy IMM (CV+CA) │ RMSE: {np.mean(imm_rmses):7.1f}m │ Time: {np.mean(imm_times):6.1f}ms")
        print(f"  QEDMMA IMM (4 model) │ RMSE: {np.mean(qedmma_rmses):7.1f}m │ Time: {np.mean(qedmma_times):6.1f}ms")
    
    # Summary
    print("\n" + "="*80)
    print("                         📊 SUMMARY")
    print("="*80)
    print(f"\n{'Scenario':<15} │ {'EKF':^10} │ {'FilterPy IMM':^12} │ {'QEDMMA IMM':^12} │ {'QEDMMA vs EKF':^14} │")
    print("─"*15 + "─┼" + "─"*12 + "┼" + "─"*14 + "┼" + "─"*14 + "┼" + "─"*16 + "┤")
    
    for name in SCENARIOS:
        ekf = results[name]['EKF'][0]
        fimm = results[name]['FilterPy IMM'][0]
        qimm = results[name]['QEDMMA IMM'][0]
        imp = (1 - qimm/ekf) * 100
        
        winner = "QEDMMA" if qimm < min(ekf, fimm) else ("FilterPy" if fimm < ekf else "EKF")
        
        print(f"{name:<15} │ {ekf:>8.1f}m │ {fimm:>10.1f}m │ {qimm:>10.1f}m │ {imp:>+12.1f}% │ 🏆 {winner}")
    
    print("\n" + "="*80)
    print("                    ✅ FAIR BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
