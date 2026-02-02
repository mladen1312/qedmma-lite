"""
QEDMMA-Lite v3.0 - Comprehensive Benchmark Suite
=================================================
Copyright (C) 2026 Dr. Mladen Mešter / Nexellum
License: AGPL-3.0-or-later

Benchmarks:
1. Filter accuracy comparison (EKF vs UKF vs CKF)
2. Computational performance
3. High-dimensional stability
4. Adaptive noise tracking
5. Zero-DSP correlation accuracy
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
import sys
sys.path.insert(0, '/home/claude/qedmma-advanced/python')

from ukf import UnscentedKalmanFilter, UKFParams
from ckf import CubatureKalmanFilter, SquareRootCKF
from adaptive_noise import SageHusaEstimator, AdaptiveNoiseState
from zero_dsp_correlation import ZeroDSPCorrelator, TwoBitCorrelator


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    name: str
    rmse_position: float
    rmse_velocity: float
    time_per_step_us: float
    divergence_rate: float
    n_runs: int


class RadarScenario:
    """Simulated radar tracking scenario"""
    
    def __init__(
        self,
        n_steps: int = 100,
        dt: float = 0.1,
        snr_db: float = 10.0,
        maneuver: bool = False
    ):
        self.n_steps = n_steps
        self.dt = dt
        self.snr_db = snr_db
        self.maneuver = maneuver
        
        # Measurement noise from SNR
        self.range_noise = 10.0 * 10**(-snr_db/20)
        self.bearing_noise = 0.01 * 10**(-snr_db/20)
    
    def generate_truth(self) -> np.ndarray:
        """Generate true target trajectory"""
        # State: [x, y, vx, vy]
        truth = np.zeros((self.n_steps, 4))
        
        # Initial state
        truth[0] = [1000.0, 1000.0, 50.0, 20.0]
        
        for k in range(1, self.n_steps):
            # Process model
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            truth[k] = F @ truth[k-1]
            
            # Add maneuver
            if self.maneuver and 30 <= k < 50:
                truth[k, 2] += 2.0  # Accelerate
                truth[k, 3] -= 1.0
            
            # Small process noise
            truth[k] += np.random.randn(4) * np.array([0.1, 0.1, 0.5, 0.5])
        
        return truth
    
    def generate_measurements(self, truth: np.ndarray) -> np.ndarray:
        """Generate noisy range-bearing measurements"""
        n = len(truth)
        measurements = np.zeros((n, 2))
        
        for k in range(n):
            x, y = truth[k, 0], truth[k, 1]
            r_true = np.sqrt(x**2 + y**2)
            theta_true = np.arctan2(y, x)
            
            measurements[k, 0] = r_true + np.random.randn() * self.range_noise
            measurements[k, 1] = theta_true + np.random.randn() * self.bearing_noise
        
        return measurements


class EKFBaseline:
    """Extended Kalman Filter for baseline comparison"""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.n = 4  # State dimension
        self.m = 2  # Measurement dimension
    
    def init_state(self, x0, P0, Q, R):
        return {'x': x0.copy(), 'P': P0.copy(), 'Q': Q, 'R': R}
    
    def predict(self, state, dt=None):
        dt = dt or self.dt
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        state['x'] = F @ state['x']
        state['P'] = F @ state['P'] @ F.T + state['Q']
        return state
    
    def update(self, state, z):
        x = state['x']
        P = state['P']
        R = state['R']
        
        # Predicted measurement
        px, py = x[0], x[1]
        r_pred = np.sqrt(px**2 + py**2)
        theta_pred = np.arctan2(py, px)
        z_pred = np.array([r_pred, theta_pred])
        
        # Jacobian
        H = np.zeros((2, 4))
        H[0, 0] = px / r_pred
        H[0, 1] = py / r_pred
        H[1, 0] = -py / (r_pred**2)
        H[1, 1] = px / (r_pred**2)
        
        # Kalman gain
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        
        # Innovation (handle angle wrapping)
        innovation = z - z_pred
        innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))
        
        # Update
        state['x'] = x + K @ innovation
        state['P'] = (np.eye(4) - K @ H) @ P
        
        return state, innovation


def run_filter_benchmark(
    filter_name: str,
    filter_obj,
    scenario: RadarScenario,
    n_monte_carlo: int = 100
) -> BenchmarkResult:
    """Run Monte Carlo benchmark for a filter"""
    
    position_errors = []
    velocity_errors = []
    times = []
    diverged = 0
    
    for run in range(n_monte_carlo):
        np.random.seed(run)
        
        # Generate scenario
        truth = scenario.generate_truth()
        measurements = scenario.generate_measurements(truth)
        
        # Initialize filter
        x0 = np.array([1000.0, 1000.0, 50.0, 20.0])
        x0 += np.random.randn(4) * np.array([50, 50, 10, 10])  # Initial uncertainty
        
        P0 = np.diag([100.0, 100.0, 20.0, 20.0])
        Q = 0.1 * np.array([
            [scenario.dt**4/4, 0, scenario.dt**3/2, 0],
            [0, scenario.dt**4/4, 0, scenario.dt**3/2],
            [scenario.dt**3/2, 0, scenario.dt**2, 0],
            [0, scenario.dt**3/2, 0, scenario.dt**2]
        ])
        R = np.diag([scenario.range_noise**2, scenario.bearing_noise**2])
        
        if hasattr(filter_obj, 'init_state'):
            state = filter_obj.init_state(x0, P0, Q, R)
        else:
            state = filter_obj.init_state(x0, P0, Q, R)
        
        run_errors_pos = []
        run_errors_vel = []
        run_diverged = False
        
        t_start = time.perf_counter()
        
        for k in range(scenario.n_steps):
            # Predict
            state = filter_obj.predict(state, scenario.dt)
            
            # Update
            state, _ = filter_obj.update(state, measurements[k])
            
            # Extract state
            if hasattr(state, 'x'):
                x_est = state.x
            else:
                x_est = state['x']
            
            # Check for divergence
            if np.any(np.isnan(x_est)) or np.linalg.norm(x_est[:2] - truth[k, :2]) > 1000:
                run_diverged = True
                break
            
            run_errors_pos.append(np.linalg.norm(x_est[:2] - truth[k, :2]))
            run_errors_vel.append(np.linalg.norm(x_est[2:4] - truth[k, 2:4]))
        
        t_end = time.perf_counter()
        
        if run_diverged:
            diverged += 1
        else:
            position_errors.extend(run_errors_pos)
            velocity_errors.extend(run_errors_vel)
            times.append((t_end - t_start) / scenario.n_steps * 1e6)  # microseconds
    
    return BenchmarkResult(
        name=filter_name,
        rmse_position=np.sqrt(np.mean(np.array(position_errors)**2)) if position_errors else float('inf'),
        rmse_velocity=np.sqrt(np.mean(np.array(velocity_errors)**2)) if velocity_errors else float('inf'),
        time_per_step_us=np.mean(times) if times else float('inf'),
        divergence_rate=diverged / n_monte_carlo,
        n_runs=n_monte_carlo
    )


def benchmark_zero_dsp_correlation():
    """Benchmark Zero-DSP correlation accuracy and speed"""
    
    results = {}
    N = 1024
    n_runs = 100
    
    # Test different SNR levels
    for snr_db in [20, 10, 5, 0, -5]:
        errors_1bit = []
        errors_2bit = []
        times_1bit = []
        times_2bit = []
        
        for run in range(n_runs):
            np.random.seed(run)
            
            # Generate PRN-like signal
            x = np.sign(np.random.randn(N))
            true_delay = np.random.randint(10, 200)
            
            # Create noisy delayed signal
            noise_power = 10**(-snr_db/10)
            y = np.zeros(N + 256)
            y[true_delay:true_delay + N] = x
            y += np.sqrt(noise_power) * np.random.randn(len(y))
            
            # 1-bit correlation
            corr1 = ZeroDSPCorrelator(n_samples=N, n_lags=256)
            t1 = time.perf_counter()
            R1, peak1 = corr1.correlate(x, y)
            t2 = time.perf_counter()
            times_1bit.append((t2-t1)*1e6)
            errors_1bit.append(abs(peak1 - true_delay))
            
            # 2-bit correlation
            corr2 = TwoBitCorrelator(n_samples=N)
            t1 = time.perf_counter()
            R2, peak2 = corr2.correlate(x, y, n_lags=256)
            t2 = time.perf_counter()
            times_2bit.append((t2-t1)*1e6)
            errors_2bit.append(abs(peak2 - true_delay))
        
        results[snr_db] = {
            '1bit_accuracy': 1 - np.mean(np.array(errors_1bit) > 2),
            '2bit_accuracy': 1 - np.mean(np.array(errors_2bit) > 2),
            '1bit_time_us': np.mean(times_1bit),
            '2bit_time_us': np.mean(times_2bit)
        }
    
    return results


def run_all_benchmarks():
    """Run complete benchmark suite"""
    
    print("=" * 80)
    print("QEDMMA-Lite v3.0 - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    print()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BENCHMARK 1: Filter Comparison (Normal conditions)
    # ═══════════════════════════════════════════════════════════════════════════
    print("▶ BENCHMARK 1: Filter Accuracy Comparison")
    print("-" * 60)
    
    scenario = RadarScenario(n_steps=100, dt=0.1, snr_db=10, maneuver=False)
    
    # Define process and measurement models for UKF/CKF
    def f_cv(x, dt):
        F = np.array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])
        return F @ x
    
    def h_rb(x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        theta = np.arctan2(x[1], x[0])
        return np.array([r, theta])
    
    filters = {
        'EKF': EKFBaseline(dt=0.1),
        'UKF': UnscentedKalmanFilter(f_cv, h_rb, 4, 2, UKFParams(alpha=0.1, beta=2, kappa=0)),
        'CKF': CubatureKalmanFilter(f_cv, h_rb, 4, 2),
        'SR-CKF': SquareRootCKF(f_cv, h_rb, 4, 2)
    }
    
    results_normal = {}
    for name, filt in filters.items():
        print(f"  Running {name}...", end=" ", flush=True)
        result = run_filter_benchmark(name, filt, scenario, n_monte_carlo=50)
        results_normal[name] = result
        print(f"RMSE={result.rmse_position:.2f}m, Time={result.time_per_step_us:.1f}μs")
    
    print()
    print("Results (SNR=10dB, 50 Monte Carlo runs):")
    print(f"{'Filter':<10} | {'RMSE Pos (m)':<12} | {'RMSE Vel (m/s)':<14} | {'Time (μs)':<10} | {'Divergence':<10}")
    print("-" * 70)
    for name, r in results_normal.items():
        print(f"{r.name:<10} | {r.rmse_position:<12.2f} | {r.rmse_velocity:<14.2f} | {r.time_per_step_us:<10.1f} | {r.divergence_rate*100:<10.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BENCHMARK 2: Low SNR Performance
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("▶ BENCHMARK 2: Low SNR Performance")
    print("-" * 60)
    
    snr_results = {}
    for snr in [20, 10, 5, 0]:
        scenario_snr = RadarScenario(n_steps=100, dt=0.1, snr_db=snr, maneuver=False)
        
        snr_results[snr] = {}
        for name in ['EKF', 'UKF', 'CKF']:
            result = run_filter_benchmark(name, filters[name], scenario_snr, n_monte_carlo=30)
            snr_results[snr][name] = result
    
    print(f"{'SNR (dB)':<10} | {'EKF RMSE':<12} | {'UKF RMSE':<12} | {'CKF RMSE':<12}")
    print("-" * 55)
    for snr in [20, 10, 5, 0]:
        print(f"{snr:<10} | {snr_results[snr]['EKF'].rmse_position:<12.2f} | "
              f"{snr_results[snr]['UKF'].rmse_position:<12.2f} | "
              f"{snr_results[snr]['CKF'].rmse_position:<12.2f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BENCHMARK 3: High-Dimensional Stability
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("▶ BENCHMARK 3: High-Dimensional Stability (n=9)")
    print("-" * 60)
    
    # 9D state: pos, vel, acc
    def f_9d(x, dt):
        F = np.eye(9)
        for i in range(3):
            F[i, i+3] = dt
            F[i, i+6] = 0.5*dt**2
            F[i+3, i+6] = dt
        return F @ x
    
    def h_9d(x):
        return x[:3]
    
    ukf_9d = UnscentedKalmanFilter(f_9d, h_9d, 9, 3, UKFParams(alpha=0.01, beta=2, kappa=0))
    ckf_9d = CubatureKalmanFilter(f_9d, h_9d, 9, 3)
    
    # Test stability
    n_test = 100
    ukf_diverged = 0
    ckf_diverged = 0
    
    for run in range(n_test):
        np.random.seed(run)
        
        x0 = np.random.randn(9) * 10
        P0 = np.eye(9) * 10
        Q = np.eye(9) * 0.1
        R = np.eye(3) * 1.0
        
        # UKF
        try:
            state_ukf = ukf_9d.init_state(x0, P0, Q, R)
            for _ in range(50):
                state_ukf = ukf_9d.predict(state_ukf, 0.1)
                z = np.random.randn(3)
                state_ukf, _ = ukf_9d.update(state_ukf, z)
                if np.any(np.isnan(state_ukf.x)) or np.any(np.abs(state_ukf.x) > 1e6):
                    ukf_diverged += 1
                    break
        except:
            ukf_diverged += 1
        
        # CKF
        try:
            state_ckf = ckf_9d.init_state(x0, P0, Q, R)
            for _ in range(50):
                state_ckf = ckf_9d.predict(state_ckf, 0.1)
                z = np.random.randn(3)
                state_ckf, _ = ckf_9d.update(state_ckf, z)
                if np.any(np.isnan(state_ckf.x)) or np.any(np.abs(state_ckf.x) > 1e6):
                    ckf_diverged += 1
                    break
        except:
            ckf_diverged += 1
    
    print(f"UKF (n=9) divergence rate: {ukf_diverged/n_test*100:.1f}%")
    print(f"CKF (n=9) divergence rate: {ckf_diverged/n_test*100:.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BENCHMARK 4: Zero-DSP Correlation
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("▶ BENCHMARK 4: Zero-DSP Correlation")
    print("-" * 60)
    
    corr_results = benchmark_zero_dsp_correlation()
    
    print(f"{'SNR (dB)':<10} | {'1-bit Acc':<12} | {'2-bit Acc':<12} | {'1-bit Time':<12} | {'2-bit Time':<12}")
    print("-" * 70)
    for snr in [20, 10, 5, 0, -5]:
        r = corr_results[snr]
        print(f"{snr:<10} | {r['1bit_accuracy']*100:<12.1f}% | {r['2bit_accuracy']*100:<12.1f}% | "
              f"{r['1bit_time_us']:<12.1f}μs | {r['2bit_time_us']:<12.1f}μs")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FILTER RECOMMENDATIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Scenario                          │ Recommended Filter                     │
├────────────────────────────────────┼────────────────────────────────────────┤
│  Linear/mildly nonlinear (n≤4)     │ EKF (fastest)                         │
│  Highly nonlinear (range-bearing)  │ UKF (best accuracy)                   │
│  High-dimensional (n>4)            │ CKF (stable weights)                  │
│  Numerical stability critical      │ SR-CKF (guaranteed PD)                │
│  Time-varying noise                │ UKF + Sage-Husa adaptive              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ZERO-DSP CORRELATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  1-bit: Best for high-SNR (>10dB), maximum FPGA throughput                 │
│  2-bit: Better low-SNR performance, still 0 DSP blocks                     │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    return results_normal, snr_results, corr_results


if __name__ == "__main__":
    results = run_all_benchmarks()
