#!/usr/bin/env python3
"""
QEDMMA-Lite v3.0 - One-Click Demo
==================================
Copyright (C) 2026 Dr. Mladen Mešter / Nexellum
License: AGPL-3.0-or-later

Simply run: python run_demo.py

This will:
1. Generate a simulated maneuvering target
2. Track it with Standard EKF and QEDMMA-Lite
3. Show real-time visualization
4. Print performance comparison
"""

import numpy as np
import sys
import time

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("⚠️  For visualization, install matplotlib:")
    print("   pip install matplotlib")
    print()

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       ██████╗ ███████╗██████╗ ███╗   ███╗███╗   ███╗ █████╗                 ║
║      ██╔═══██╗██╔════╝██╔══██╗████╗ ████║████╗ ████║██╔══██╗                ║
║      ██║   ██║█████╗  ██║  ██║██╔████╔██║██╔████╔██║███████║                ║
║      ██║▄▄ ██║██╔══╝  ██║  ██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║                ║
║      ╚██████╔╝███████╗██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║                ║
║       ╚══▀▀═╝ ╚══════╝╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝                ║
║                                                                              ║
║                    LITE v3.0 - Interactive Demo                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

# ================= TRACKER IMPLEMENTATIONS =================

class StandardEKF:
    """Standard Extended Kalman Filter (the competition)"""
    
    def __init__(self, dt=0.1):
        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        q = 0.5
        self.Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ])
        self.R = np.diag([100.0, 100.0])
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100
    
    def step(self, z):
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()


class QEDMMALite:
    """QEDMMA-Lite IMM Tracker (our solution)"""
    
    def __init__(self, dt=0.1):
        self.dt = dt
        self.mu = np.array([0.6, 0.3, 0.1])
        self.TPM = np.array([
            [0.90, 0.08, 0.02],
            [0.10, 0.85, 0.05],
            [0.05, 0.10, 0.85]
        ])
        
        def make_Q(q):
            return q * np.array([
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2]
            ])
        
        self.filters = [
            {'Q': make_Q(1), 'x': np.zeros(4), 'P': np.eye(4)*100},
            {'Q': make_Q(50), 'x': np.zeros(4), 'P': np.eye(4)*100},
            {'Q': make_Q(100), 'x': np.zeros(4), 'P': np.eye(4)*100}
        ]
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.diag([100.0, 100.0])
        self.x = np.zeros(4)
        self.initialized = False
    
    def init_state(self, x0):
        """Initialize all filters with the same state"""
        self.x = x0.copy()
        for f in self.filters:
            f['x'] = x0.copy()
            f['P'] = np.eye(4) * 100
        self.initialized = True
    
    def step(self, z):
        # Initialize from first measurement if needed
        if not self.initialized:
            x0 = np.array([z[0], z[1], 0, 0])
            self.init_state(x0)
            return self.x.copy()
        
        # IMM Mixing
        c_bar = self.TPM.T @ self.mu
        x_mixed = []
        P_mixed = []
        
        for j in range(3):
            x_j = np.zeros(4)
            P_j = np.zeros((4, 4))
            for i in range(3):
                mu_ij = self.TPM[i, j] * self.mu[i] / (c_bar[j] + 1e-10)
                x_j += mu_ij * self.filters[i]['x']
            for i in range(3):
                mu_ij = self.TPM[i, j] * self.mu[i] / (c_bar[j] + 1e-10)
                diff = self.filters[i]['x'] - x_j
                P_j += mu_ij * (self.filters[i]['P'] + np.outer(diff, diff))
            x_mixed.append(x_j)
            P_mixed.append(P_j)
        
        # Predict and Update each filter
        likelihoods = []
        for j, f in enumerate(self.filters):
            f['x'] = self.F @ x_mixed[j]
            f['P'] = self.F @ P_mixed[j] @ self.F.T + f['Q']
            
            y = z - self.H @ f['x']
            S = self.H @ f['P'] @ self.H.T + self.R
            try:
                S_inv = np.linalg.inv(S)
                K = f['P'] @ self.H.T @ S_inv
                det_S = np.linalg.det(S)
                if det_S > 1e-10:
                    mahal = y @ S_inv @ y
                    L = np.exp(-0.5 * mahal) / (2*np.pi*np.sqrt(det_S))
                else:
                    L = 1e-10
            except:
                K = np.zeros((4, 2))
                L = 1e-10
            
            f['x'] = f['x'] + K @ y
            f['P'] = (np.eye(4) - K @ self.H) @ f['P']
            f['P'] = 0.5 * (f['P'] + f['P'].T)  # Symmetry
            likelihoods.append(L)
        
        # Update mode probabilities
        self.mu = c_bar * np.array(likelihoods)
        mu_sum = np.sum(self.mu)
        if mu_sum > 1e-10:
            self.mu /= mu_sum
        else:
            self.mu = np.array([0.6, 0.3, 0.1])
        
        # Combined estimate
        self.x = sum(self.mu[i] * self.filters[i]['x'] for i in range(3))
        return self.x.copy()


def generate_trajectory(n_steps=150, dt=0.1):
    """Generate a fighter aircraft trajectory with maneuvers"""
    states = []
    x = np.array([0.0, 0.0, 300.0, 50.0])
    
    for t in range(n_steps):
        if t < 40:
            ax, ay = 0, 0
        elif t < 70:
            ax, ay = -50, 35  # Hard turn
        elif t < 100:
            ax, ay = 40, -40  # Reverse
        else:
            ax = 30 * np.sin(t * dt * 2)
            ay = 30 * np.cos(t * dt * 3)
        
        x[0] += x[2] * dt + 0.5 * ax * dt**2
        x[1] += x[3] * dt + 0.5 * ay * dt**2
        x[2] += ax * dt
        x[3] += ay * dt
        states.append(x.copy())
    
    return np.array(states)


def run_console_demo():
    """Text-based demo for systems without matplotlib"""
    print_banner()
    print("Running text-based demo (install matplotlib for visualization)\n")
    
    np.random.seed(42)
    dt = 0.1
    n_steps = 150
    
    # Generate trajectory
    true_states = generate_trajectory(n_steps, dt)
    
    # Initialize trackers
    ekf = StandardEKF(dt)
    qedmma = QEDMMALite(dt)
    
    # Initialize with known starting state
    init_state = np.array([0, 0, 300, 50])
    ekf.x = init_state.copy()
    ekf.P = np.eye(4) * 100
    qedmma.init_state(init_state)
    
    ekf_errors = []
    qedmma_errors = []
    
    print("Tracking simulation started...")
    print(f"{'Step':>5} | {'Phase':<15} | {'EKF Error':>10} | {'QEDMMA Error':>12} | {'Winner':>8}")
    print("-" * 70)
    
    for t in range(n_steps):
        true = true_states[t]
        z = true[:2] + np.random.randn(2) * 50
        
        est_ekf = ekf.step(z)
        est_qedmma = qedmma.step(z)
        
        err_ekf = np.linalg.norm(est_ekf[:2] - true[:2])
        err_qedmma = np.linalg.norm(est_qedmma[:2] - true[:2])
        
        ekf_errors.append(err_ekf)
        qedmma_errors.append(err_qedmma)
        
        # Determine phase
        if t < 40:
            phase = "Straight"
        elif t < 70:
            phase = "Hard Turn"
        elif t < 100:
            phase = "Reverse Turn"
        else:
            phase = "Evasive"
        
        winner = "QEDMMA" if err_qedmma < err_ekf else "EKF"
        
        if t % 15 == 0:
            print(f"{t:>5} | {phase:<15} | {err_ekf:>8.1f} m | {err_qedmma:>10.1f} m | {winner:>8}")
    
    # Final stats
    rmse_ekf = np.sqrt(np.mean(np.array(ekf_errors)**2))
    rmse_qedmma = np.sqrt(np.mean(np.array(qedmma_errors)**2))
    improvement = (rmse_ekf - rmse_qedmma) / rmse_ekf * 100
    
    print("-" * 70)
    print()
    print("═" * 70)
    print("                         FINAL RESULTS")
    print("═" * 70)
    print(f"  Standard EKF RMSE:    {rmse_ekf:>8.1f} m")
    print(f"  QEDMMA-Lite RMSE:     {rmse_qedmma:>8.1f} m")
    print(f"  Improvement:          {improvement:>8.1f} %")
    print("═" * 70)
    print()
    print("💡 QEDMMA-Lite handles maneuvers better due to IMM mode switching!")
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  🚀 Want even better performance? QEDMMA-PRO achieves < 15m RMSE  ║")
    print("║                                                                    ║")
    print("║  📧 Contact: mladen@nexellum.com                                  ║")
    print("║  🌐 Web:     www.nexellum.com                                     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")


def run_visual_demo():
    """Animated visualization demo"""
    print_banner()
    print("Starting visual demo...\n")
    
    np.random.seed(42)
    dt = 0.1
    n_steps = 150
    
    # Generate trajectory
    true_states = generate_trajectory(n_steps, dt)
    measurements = true_states[:, :2] + np.random.randn(n_steps, 2) * 50
    
    # Initialize trackers
    ekf = StandardEKF(dt)
    qedmma = QEDMMALite(dt)
    
    # Initialize with known starting state
    init_state = np.array([0, 0, 300, 50])
    ekf.x = init_state.copy()
    ekf.P = np.eye(4) * 100
    qedmma.init_state(init_state)
    
    # Storage for animation
    ekf_history = [ekf.x[:2].copy()]
    qedmma_history = [qedmma.x[:2].copy()]
    ekf_errors = []
    qedmma_errors = []
    
    # Pre-compute all estimates
    for t in range(n_steps):
        z = measurements[t]
        est_ekf = ekf.step(z)
        est_qedmma = qedmma.step(z)
        
        ekf_history.append(est_ekf[:2].copy())
        qedmma_history.append(est_qedmma[:2].copy())
        
        err_ekf = np.linalg.norm(est_ekf[:2] - true_states[t, :2])
        err_qedmma = np.linalg.norm(est_qedmma[:2] - true_states[t, :2])
        ekf_errors.append(err_ekf)
        qedmma_errors.append(err_qedmma)
    
    ekf_history = np.array(ekf_history)
    qedmma_history = np.array(qedmma_history)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('QEDMMA-Lite v3.0 - Real-Time Tracking Demo', fontsize=14, fontweight='bold')
    
    # Trajectory plot
    ax1.set_xlim(true_states[:, 0].min() - 500, true_states[:, 0].max() + 500)
    ax1.set_ylim(true_states[:, 1].min() - 500, true_states[:, 1].max() + 500)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Target Tracking')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    true_line, = ax1.plot([], [], 'k-', linewidth=2, label='True Path')
    ekf_line, = ax1.plot([], [], 'r--', linewidth=1.5, alpha=0.7, label='EKF')
    qedmma_line, = ax1.plot([], [], 'g-', linewidth=1.5, alpha=0.7, label='QEDMMA-Lite')
    meas_scatter = ax1.scatter([], [], c='blue', s=10, alpha=0.3, label='Measurements')
    ax1.legend(loc='upper left')
    
    # Error plot
    ax2.set_xlim(0, n_steps * dt)
    ax2.set_ylim(0, max(max(ekf_errors), max(qedmma_errors)) * 1.2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Tracking Error Comparison')
    ax2.grid(True, alpha=0.3)
    
    ekf_err_line, = ax2.plot([], [], 'r-', label='EKF')
    qedmma_err_line, = ax2.plot([], [], 'g-', label='QEDMMA-Lite')
    ax2.legend(loc='upper right')
    
    # Text annotations
    rmse_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=10, 
                          verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        true_line.set_data([], [])
        ekf_line.set_data([], [])
        qedmma_line.set_data([], [])
        meas_scatter.set_offsets(np.c_[[], []])
        ekf_err_line.set_data([], [])
        qedmma_err_line.set_data([], [])
        return true_line, ekf_line, qedmma_line, meas_scatter, ekf_err_line, qedmma_err_line
    
    def animate(frame):
        # Update trajectory
        true_line.set_data(true_states[:frame+1, 0], true_states[:frame+1, 1])
        ekf_line.set_data(ekf_history[:frame+1, 0], ekf_history[:frame+1, 1])
        qedmma_line.set_data(qedmma_history[:frame+1, 0], qedmma_history[:frame+1, 1])
        meas_scatter.set_offsets(measurements[:frame+1])
        
        # Update errors
        t = np.arange(frame) * dt
        ekf_err_line.set_data(t, ekf_errors[:frame])
        qedmma_err_line.set_data(t, qedmma_errors[:frame])
        
        # Update RMSE text
        if frame > 10:
            rmse_ekf = np.sqrt(np.mean(np.array(ekf_errors[:frame])**2))
            rmse_qedmma = np.sqrt(np.mean(np.array(qedmma_errors[:frame])**2))
            improvement = (rmse_ekf - rmse_qedmma) / rmse_ekf * 100
            rmse_text.set_text(f'EKF RMSE:    {rmse_ekf:.1f} m\n'
                              f'QEDMMA RMSE: {rmse_qedmma:.1f} m\n'
                              f'Improvement: {improvement:.1f}%')
        
        return true_line, ekf_line, qedmma_line, meas_scatter, ekf_err_line, qedmma_err_line
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_steps, 
                         interval=50, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show()
    
    # Print final results
    rmse_ekf = np.sqrt(np.mean(np.array(ekf_errors)**2))
    rmse_qedmma = np.sqrt(np.mean(np.array(qedmma_errors)**2))
    improvement = (rmse_ekf - rmse_qedmma) / rmse_ekf * 100
    
    print("\n" + "═" * 60)
    print("                    DEMO COMPLETE")
    print("═" * 60)
    print(f"  Standard EKF RMSE:    {rmse_ekf:.1f} m")
    print(f"  QEDMMA-Lite RMSE:     {rmse_qedmma:.1f} m")
    print(f"  Improvement:          {improvement:.1f}%")
    print("═" * 60)
    print()
    print("🚀 For PRO version: mladen@nexellum.com")


if __name__ == "__main__":
    if HAS_PLOT:
        run_visual_demo()
    else:
        run_console_demo()
