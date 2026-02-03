#!/usr/bin/env python3
"""
PROPERLY TUNED IMM vs EKF Benchmark

The key insight: IMM wins when models are CORRECTLY differentiated.
- CV model: LOW process noise (assumes constant velocity)
- CA model: MEDIUM process noise (allows acceleration)  
- CT model: HIGH process noise (allows rapid velocity changes)

If all models have similar Q, IMM degrades to weighted EKF with overhead.
"""

import numpy as np
import time
from typing import List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# PROPERLY TUNED IMM
# ═══════════════════════════════════════════════════════════════════════════════

def imm_step_tuned(x_est, P_est, z, models, H, R, pi, mu_prev):
    """
    Vectorized IMM step with proper tuning.
    """
    M = len(models)
    
    # --- Predicted mode probabilities ---
    bar_c = mu_prev @ pi
    bar_c = np.clip(bar_c, 1e-12, None)
    
    # --- Mixing probabilities ---
    mixing_probs = (pi * mu_prev[:, None]) / bar_c[None, :]
    
    # --- Mixed state ---
    x_mix = mixing_probs.T @ x_est
    
    # --- Mixed covariance ---
    P_mix = np.zeros((M, 4, 4))
    for j in range(M):
        for i in range(M):
            diff = x_est[i] - x_mix[j]
            P_mix[j] += mixing_probs[i, j] * (P_est[i] + np.outer(diff, diff))
    
    # --- Model-conditioned filtering ---
    x_upd = np.zeros((M, 4))
    P_upd = np.zeros((M, 4, 4))
    likelihood = np.zeros(M)
    
    for j in range(M):
        F, Q = models[j]
        
        # Predict
        x_pred = F @ x_mix[j]
        P_pred = F @ P_mix[j] @ F.T + Q
        
        # Update
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        
        # Numerical stability
        try:
            S_inv = np.linalg.inv(S)
            det_S = np.linalg.det(S)
        except:
            S_inv = np.eye(2) * 1e-6
            det_S = 1e-10
        
        K = P_pred @ H.T @ S_inv
        x_upd[j] = x_pred + K @ y
        
        # Joseph form
        I_KH = np.eye(4) - K @ H
        P_upd[j] = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        
        # Likelihood
        if det_S > 1e-10:
            likelihood[j] = np.exp(-0.5 * y @ S_inv @ y) / np.sqrt((2*np.pi)**2 * det_S)
        else:
            likelihood[j] = 1e-10
    
    # --- Mode probability update ---
    mu = likelihood * bar_c
    mu_sum = mu.sum()
    if mu_sum > 1e-12:
        mu /= mu_sum
    else:
        mu = np.ones(M) / M
    
    # --- Combined estimate ---
    x_combined = mu @ x_upd
    P_combined = np.zeros((4, 4))
    for j in range(M):
        diff = x_upd[j] - x_combined
        P_combined += mu[j] * (P_upd[j] + np.outer(diff, diff))
    
    return x_combined, P_combined, mu, x_upd, P_upd


class TunedIMM:
    """Properly tuned 3-model IMM."""
    
    def __init__(self, dt: float, meas_noise: float):
        self.dt = dt
        self.M = 3
        
        # State: [x, vx, y, vy]
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        # CRITICAL: Q matrices must be DIFFERENTIATED!
        # CV: Very low Q - assumes nearly constant velocity
        q_cv = 0.1
        Q_cv = np.array([
            [dt**4/4, dt**3/2, 0, 0],
            [dt**3/2, dt**2, 0, 0],
            [0, 0, dt**4/4, dt**3/2],
            [0, 0, dt**3/2, dt**2]
        ]) * q_cv**2
        
        # CA: Medium Q - allows moderate acceleration
        q_ca = 5.0
        Q_ca = np.array([
            [dt**4/4, dt**3/2, 0, 0],
            [dt**3/2, dt**2, 0, 0],
            [0, 0, dt**4/4, dt**3/2],
            [0, 0, dt**3/2, dt**2]
        ]) * q_ca**2
        
        # CT: High Q - allows rapid velocity changes (maneuvers)
        q_ct = 30.0
        Q_ct = np.array([
            [dt**4/4, dt**3/2, 0, 0],
            [dt**3/2, dt**2, 0, 0],
            [0, 0, dt**4/4, dt**3/2],
            [0, 0, dt**3/2, dt**2]
        ]) * q_ct**2
        
        self.models = [(F.copy(), Q_cv), (F.copy(), Q_ca), (F.copy(), Q_ct)]
        
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.R = np.eye(2) * meas_noise**2
        
        # Transition matrix - slight bias toward staying in same mode
        self.pi = np.array([
            [0.95, 0.025, 0.025],  # CV stays CV
            [0.025, 0.95, 0.025],  # CA stays CA
            [0.025, 0.025, 0.95]   # CT stays CT
        ])
        
        self.x_est = np.zeros((self.M, 4))
        self.P_est = np.zeros((self.M, 4, 4))
        self.mu = np.array([0.8, 0.1, 0.1])  # Start assuming CV
    
    def initialize(self, x0: np.ndarray, P0: np.ndarray):
        for j in range(self.M):
            self.x_est[j] = x0.copy()
            self.P_est[j] = P0.copy()
    
    def update(self, z: np.ndarray):
        x_comb, P_comb, self.mu, self.x_est, self.P_est = imm_step_tuned(
            self.x_est, self.P_est, z, self.models, self.H, self.R, self.pi, self.mu
        )
        return x_comb, self.mu


# ═══════════════════════════════════════════════════════════════════════════════
# EKF BASELINE (with matching CV assumptions)
# ═══════════════════════════════════════════════════════════════════════════════

class TunedEKF:
    """EKF tuned for same scenarios."""
    
    def __init__(self, dt: float, q: float, meas_noise: float):
        self.dt = dt
        
        self.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        
        self.Q = np.array([
            [dt**4/4, dt**3/2, 0, 0],
            [dt**3/2, dt**2, 0, 0],
            [0, 0, dt**4/4, dt**3/2],
            [0, 0, dt**3/2, dt**2]
        ]) * q**2
        
        self.R = np.eye(2) * meas_noise**2
        
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1000
    
    def initialize(self, x0, P0):
        self.x = x0.copy()
        self.P = P0.copy()
    
    def update(self, z):
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY WITH REALISTIC MANEUVERS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_realistic_trajectory(n_steps: int, dt: float, 
                                  maneuvers: List[Tuple[int, int, float]], 
                                  meas_noise: float, seed: int = 42):
    np.random.seed(seed)
    
    truth = np.zeros((n_steps, 4))
    truth[0] = [0, 200, 0, 50]  # Initial position and velocity
    
    for k in range(1, n_steps):
        x, vx, y, vy = truth[k-1]
        
        omega = 0.0
        for start, end, w in maneuvers:
            if start <= k < end:
                omega = w
                break
        
        if abs(omega) > 1e-6:
            # Coordinated turn
            cos_w, sin_w = np.cos(omega * dt), np.sin(omega * dt)
            vx_new = vx * cos_w - vy * sin_w
            vy_new = vx * sin_w + vy * cos_w
        else:
            vx_new, vy_new = vx, vy
        
        # Very small process noise in truth
        vx_new += np.random.randn() * 0.5
        vy_new += np.random.randn() * 0.5
        
        truth[k] = [x + vx * dt, vx_new, y + vy * dt, vy_new]
    
    # Noisy measurements
    meas = truth[:, [0, 2]] + np.random.randn(n_steps, 2) * meas_noise
    
    return truth, meas


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*80)
    print("     🔬 PROPERLY TUNED IMM vs EKF BENCHMARK")
    print("="*80)
    
    scenarios = {
        "Linear (No Maneuvers)": {
            "n_steps": 300,
            "dt": 0.1,
            "maneuvers": [],
            "meas_noise": 50.0,
            "ekf_q": 1.0  # Low Q for EKF (CV assumption)
        },
        "Light Maneuvers": {
            "n_steps": 400,
            "dt": 0.1,
            "maneuvers": [(100, 150, 0.05), (250, 300, -0.05)],
            "meas_noise": 50.0,
            "ekf_q": 5.0
        },
        "Heavy Maneuvers (3g)": {
            "n_steps": 400,
            "dt": 0.1,
            "maneuvers": [(50, 100, 0.12), (150, 200, -0.15), (280, 350, 0.18)],
            "meas_noise": 50.0,
            "ekf_q": 5.0
        },
        "Aggressive (5g+)": {
            "n_steps": 500,
            "dt": 0.05,
            "maneuvers": [(30, 70, 0.20), (100, 150, -0.25), (200, 260, 0.22), (350, 420, -0.18)],
            "meas_noise": 80.0,
            "ekf_q": 10.0
        },
    }
    
    n_runs = 10
    results = {}
    
    for name, params in scenarios.items():
        print(f"\n{'─'*80}")
        print(f"📊 {name}")
        print(f"   Maneuvers: {len(params['maneuvers'])}")
        print(f"{'─'*80}")
        
        ekf_rmses, ekf_times = [], []
        imm_rmses, imm_times = [], []
        final_mu = None
        
        for run in range(n_runs):
            truth, meas = generate_realistic_trajectory(
                params["n_steps"], params["dt"], params["maneuvers"],
                params["meas_noise"], seed=42+run
            )
            
            x0 = np.array([meas[0, 0], 200, meas[0, 1], 50])
            P0 = np.diag([500, 100, 500, 100])
            
            # EKF
            ekf = TunedEKF(params["dt"], params["ekf_q"], params["meas_noise"])
            ekf.initialize(x0.copy(), P0.copy())
            
            ekf_est = np.zeros((len(meas), 4))
            ekf_est[0] = x0
            
            t0 = time.perf_counter()
            for k in range(1, len(meas)):
                ekf_est[k] = ekf.update(meas[k])
            ekf_times.append((time.perf_counter() - t0) * 1000)
            
            err = np.sqrt((ekf_est[:, 0] - truth[:, 0])**2 + 
                         (ekf_est[:, 2] - truth[:, 2])**2)
            ekf_rmses.append(np.sqrt(np.mean(err**2)))
            
            # IMM
            imm = TunedIMM(params["dt"], params["meas_noise"])
            imm.initialize(x0.copy(), P0.copy())
            
            imm_est = np.zeros((len(meas), 4))
            imm_est[0] = x0
            
            t0 = time.perf_counter()
            for k in range(1, len(meas)):
                imm_est[k], mu = imm.update(meas[k])
            imm_times.append((time.perf_counter() - t0) * 1000)
            final_mu = mu
            
            err = np.sqrt((imm_est[:, 0] - truth[:, 0])**2 + 
                         (imm_est[:, 2] - truth[:, 2])**2)
            imm_rmses.append(np.sqrt(np.mean(err**2)))
        
        results[name] = {
            "EKF": (np.mean(ekf_rmses), np.std(ekf_rmses), np.mean(ekf_times)),
            "IMM": (np.mean(imm_rmses), np.std(imm_rmses), np.mean(imm_times))
        }
        
        improvement = (1 - np.mean(imm_rmses) / np.mean(ekf_rmses)) * 100
        winner = "✅ IMM" if improvement > 0 else "❌ EKF"
        
        print(f"  EKF      │ RMSE: {np.mean(ekf_rmses):6.1f}m ± {np.std(ekf_rmses):4.1f} │ Time: {np.mean(ekf_times):5.1f}ms")
        print(f"  IMM      │ RMSE: {np.mean(imm_rmses):6.1f}m ± {np.std(imm_rmses):4.1f} │ Time: {np.mean(imm_times):5.1f}ms")
        print(f"  ──────────────────────────────────────────────────────────────")
        print(f"  Result: {winner} ({improvement:+.1f}%)")
        if final_mu is not None:
            print(f"  Final mode probs: CV={final_mu[0]:.2f} CA={final_mu[1]:.2f} CT={final_mu[2]:.2f}")
    
    # Summary
    print("\n" + "="*80)
    print("                       📊 FINAL SUMMARY")
    print("="*80)
    print(f"\n{'Scenario':<25} │ {'EKF':^10} │ {'IMM':^10} │ {'Improve':^10} │")
    print("─"*25 + "─┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*12 + "┤")
    
    for name in scenarios:
        ekf_rmse = results[name]["EKF"][0]
        imm_rmse = results[name]["IMM"][0]
        improvement = (1 - imm_rmse / ekf_rmse) * 100
        winner = "🏆" if imm_rmse < ekf_rmse else "  "
        
        print(f"{name[:25]:<25} │ {ekf_rmse:>8.1f}m │ {imm_rmse:>8.1f}m │ {improvement:>+8.1f}% │{winner}")
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
