#!/usr/bin/env python3
"""
IMM/EKF Benchmark Tool

Comprehensive comparison of tracking algorithms:
- Standard EKF (baseline)
- 2-Model IMM (CV + CA)
- 3-Model IMM (CV + CA + CT)
- QEDMMA-Lite (4-Model IMM with optimizations)

This tool demonstrates the significant improvements of modern
multi-model approaches over traditional single-model filters.

MIT License - Copyright (c) 2026 Dr. Mladen Mešter
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import argparse
from datetime import datetime
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qedmma_lite.tracker import (
    QEDMMATracker, Measurement, TrackState,
    KalmanFilter, ConstantVelocity, ConstantAcceleration,
    CoordinatedTurn, ConstantJerk
)


# =============================================================================
# Constants
# =============================================================================

G = 9.80665
SPEED_OF_SOUND = 340.0  # m/s at sea level


# =============================================================================
# Baseline EKF Implementation (for comparison)
# =============================================================================

class BaselineEKF:
    """
    Standard Extended Kalman Filter with Constant Velocity model.
    
    This represents the typical filter used in legacy radar systems.
    It performs poorly with maneuvering targets.
    """
    
    def __init__(self, dt: float = 0.0625):
        self.dt = dt
        self.state_dim = 6  # [x, y, z, vx, vy, vz] - no acceleration
        
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1000
        
        # Process noise
        q = 10.0
        self.Q = np.diag([q, q, q, q*0.1, q*0.1, q*0.1])
        
        # Measurement matrix
        self.H = np.eye(self.state_dim)
        
        self.initialized = False
    
    def _get_F(self) -> np.ndarray:
        """State transition matrix (CV model)"""
        F = np.eye(self.state_dim)
        dt = self.dt
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F
    
    def predict(self):
        """Prediction step"""
        F = self._get_F()
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, measurement: Measurement):
        """Update step"""
        if not self.initialized:
            self.x[:3] = measurement.pos
            if measurement.vel is not None:
                self.x[3:6] = measurement.vel
            self.initialized = True
            return
        
        # Construct measurement
        z = np.concatenate([measurement.pos, 
                          measurement.vel if measurement.vel is not None else np.zeros(3)])
        
        R = np.diag([measurement.noise_pos**2] * 3 + [measurement.noise_vel**2] * 3)
        
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
    
    def get_estimate(self) -> np.ndarray:
        """Return current estimate [x, y, z, vx, vy, vz, 0, 0, 0]"""
        return np.concatenate([self.x, np.zeros(3)])


class IMM2Model:
    """2-Model IMM (CV + CA) - Intermediate complexity"""
    
    def __init__(self, dt: float = 0.0625):
        self.dt = dt
        self.models = [
            ConstantVelocity(dt, process_noise=1.0),
            ConstantAcceleration(dt, process_noise=10.0)
        ]
        self.filters = [KalmanFilter(m) for m in self.models]
        self.mu = np.array([0.6, 0.4])
        self.trans_prob = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.initialized = False
    
    def predict(self):
        for kf in self.filters:
            kf.predict()
    
    def update(self, measurement: Measurement):
        if not self.initialized:
            x0 = np.zeros(9)
            x0[:3] = measurement.pos
            if measurement.vel is not None:
                x0[3:6] = measurement.vel
            for kf in self.filters:
                kf.set_state(x0, np.eye(9) * 1000)
            self.initialized = True
            return
        
        z = np.concatenate([measurement.pos, 
                          measurement.vel if measurement.vel is not None else np.zeros(3)])
        R = np.diag([measurement.noise_pos**2] * 3 + [measurement.noise_vel**2] * 3)
        
        likelihoods = np.array([kf.update(z, R) for kf in self.filters])
        
        c = self.trans_prob.T @ self.mu
        self.mu = likelihoods * c
        if np.sum(self.mu) > 1e-10:
            self.mu /= np.sum(self.mu)
    
    def get_estimate(self) -> np.ndarray:
        x = np.zeros(9)
        for j, kf in enumerate(self.filters):
            x += self.mu[j] * kf.get_state()
        return x


class IMM3Model:
    """3-Model IMM (CV + CA + CT) - Good complexity/performance tradeoff"""
    
    def __init__(self, dt: float = 0.0625):
        self.dt = dt
        self.models = [
            ConstantVelocity(dt, process_noise=1.0),
            ConstantAcceleration(dt, process_noise=10.0),
            CoordinatedTurn(dt, process_noise=50.0)
        ]
        self.filters = [KalmanFilter(m) for m in self.models]
        self.mu = np.array([0.5, 0.3, 0.2])
        self.trans_prob = np.array([
            [0.85, 0.10, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85]
        ])
        self.initialized = False
    
    def predict(self):
        for kf in self.filters:
            kf.predict()
    
    def update(self, measurement: Measurement):
        if not self.initialized:
            x0 = np.zeros(9)
            x0[:3] = measurement.pos
            if measurement.vel is not None:
                x0[3:6] = measurement.vel
            for kf in self.filters:
                kf.set_state(x0, np.eye(9) * 1000)
            self.initialized = True
            return
        
        z = np.concatenate([measurement.pos, 
                          measurement.vel if measurement.vel is not None else np.zeros(3)])
        R = np.diag([measurement.noise_pos**2] * 3 + [measurement.noise_vel**2] * 3)
        
        likelihoods = np.array([kf.update(z, R) for kf in self.filters])
        
        c = self.trans_prob.T @ self.mu
        self.mu = likelihoods * c
        if np.sum(self.mu) > 1e-10:
            self.mu /= np.sum(self.mu)
    
    def get_estimate(self) -> np.ndarray:
        x = np.zeros(9)
        for j, kf in enumerate(self.filters):
            x += self.mu[j] * kf.get_state()
        return x


# =============================================================================
# Trajectory Generator
# =============================================================================

@dataclass
class TrajectoryPoint:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    time: float


class TrajectoryGenerator:
    """Generate test trajectories for benchmarking"""
    
    @staticmethod
    def constant_velocity(duration: float, dt: float, 
                         initial_pos: np.ndarray,
                         velocity: np.ndarray) -> List[TrajectoryPoint]:
        """Generate constant velocity trajectory"""
        trajectory = []
        pos = initial_pos.copy()
        vel = velocity.copy()
        acc = np.zeros(3)
        t = 0.0
        
        while t < duration:
            trajectory.append(TrajectoryPoint(
                pos=pos.copy(), vel=vel.copy(), acc=acc.copy(), time=t
            ))
            pos += vel * dt
            t += dt
        
        return trajectory
    
    @staticmethod
    def hypersonic_cruise(duration: float, dt: float,
                         mach: float = 8.0) -> List[TrajectoryPoint]:
        """Generate hypersonic cruise trajectory"""
        speed = mach * SPEED_OF_SOUND
        return TrajectoryGenerator.constant_velocity(
            duration, dt,
            initial_pos=np.array([0.0, 0.0, 25000.0]),
            velocity=np.array([speed, 0.0, 0.0])
        )
    
    @staticmethod
    def high_g_maneuver(duration: float, dt: float,
                       mach: float = 8.0, g_load: float = 60.0,
                       maneuver_start: float = 3.0,
                       maneuver_duration: float = 2.0) -> List[TrajectoryPoint]:
        """Generate hypersonic trajectory with high-G pull-up maneuver"""
        trajectory = []
        
        speed = mach * SPEED_OF_SOUND
        pos = np.array([0.0, 0.0, 30000.0])
        vel = np.array([speed * np.cos(-0.5), 0.0, speed * np.sin(-0.5)])  # 30° dive
        acc = np.array([0.0, 0.0, 0.0])
        t = 0.0
        
        while t < duration:
            # Determine acceleration phase
            if maneuver_start <= t < maneuver_start + maneuver_duration:
                # Pull-up maneuver
                acc = np.array([0.0, 0.0, g_load * G])
            else:
                acc = np.array([0.0, 0.0, 0.0])
            
            trajectory.append(TrajectoryPoint(
                pos=pos.copy(), vel=vel.copy(), acc=acc.copy(), time=t
            ))
            
            # Integration
            pos += vel * dt + 0.5 * acc * dt**2
            vel += acc * dt
            
            # Limit speed
            current_speed = np.linalg.norm(vel)
            if current_speed > 12 * SPEED_OF_SOUND:
                vel = vel * 12 * SPEED_OF_SOUND / current_speed
            
            t += dt
        
        return trajectory
    
    @staticmethod
    def evasive_maneuver(duration: float, dt: float,
                        mach: float = 6.0) -> List[TrajectoryPoint]:
        """Generate S-turn evasive maneuver"""
        trajectory = []
        
        speed = mach * SPEED_OF_SOUND
        pos = np.array([0.0, 0.0, 20000.0])
        vel = np.array([speed, 0.0, 0.0])
        t = 0.0
        
        turn_frequency = 0.5  # Hz
        max_lateral_g = 30 * G
        
        while t < duration:
            # Sinusoidal lateral acceleration
            acc_y = max_lateral_g * np.sin(2 * np.pi * turn_frequency * t)
            acc = np.array([0.0, acc_y, 0.0])
            
            trajectory.append(TrajectoryPoint(
                pos=pos.copy(), vel=vel.copy(), acc=acc.copy(), time=t
            ))
            
            pos += vel * dt + 0.5 * acc * dt**2
            vel += acc * dt
            t += dt
        
        return trajectory


# =============================================================================
# Benchmark Engine
# =============================================================================

@dataclass
class BenchmarkResult:
    algorithm: str
    scenario: str
    pos_rmse: float
    vel_rmse: float
    track_loss_frames: int
    total_frames: int
    computation_time_ms: float
    
    def to_dict(self) -> dict:
        return {
            'algorithm': self.algorithm,
            'scenario': self.scenario,
            'pos_rmse': self.pos_rmse,
            'vel_rmse': self.vel_rmse,
            'track_loss_frames': self.track_loss_frames,
            'total_frames': self.total_frames,
            'computation_time_ms': self.computation_time_ms,
            'track_loss_percent': 100 * self.track_loss_frames / max(1, self.total_frames)
        }


class BenchmarkEngine:
    """Run comprehensive tracking algorithm benchmarks"""
    
    def __init__(self, dt: float = 0.0625, noise_pos: float = 50.0, noise_vel: float = 5.0):
        self.dt = dt
        self.noise_pos = noise_pos
        self.noise_vel = noise_vel
    
    def _add_noise(self, trajectory: List[TrajectoryPoint]) -> List[Measurement]:
        """Add measurement noise to trajectory"""
        measurements = []
        for pt in trajectory:
            noisy_pos = pt.pos + np.random.randn(3) * self.noise_pos
            noisy_vel = pt.vel + np.random.randn(3) * self.noise_vel
            measurements.append(Measurement(
                pos=noisy_pos, vel=noisy_vel,
                noise_pos=self.noise_pos, noise_vel=self.noise_vel,
                time=pt.time
            ))
        return measurements
    
    def _run_single_benchmark(self, tracker, measurements: List[Measurement],
                              ground_truth: List[TrajectoryPoint],
                              algorithm_name: str,
                              scenario_name: str) -> BenchmarkResult:
        """Run benchmark for a single algorithm"""
        import time
        
        pos_errors = []
        vel_errors = []
        track_loss = 0
        track_loss_threshold = 1000  # meters
        
        start_time = time.time()
        
        for i, (meas, truth) in enumerate(zip(measurements, ground_truth)):
            # Update tracker
            if hasattr(tracker, 'update'):
                if isinstance(tracker, QEDMMATracker):
                    estimate = tracker.update(meas)
                    est_state = np.concatenate([estimate.pos, estimate.vel, estimate.acc])
                else:
                    tracker.predict()
                    tracker.update(meas)
                    est_state = tracker.get_estimate()
            
            # Calculate errors
            pos_err = np.linalg.norm(truth.pos - est_state[:3])
            vel_err = np.linalg.norm(truth.vel - est_state[3:6])
            
            pos_errors.append(pos_err)
            vel_errors.append(vel_err)
            
            if pos_err > track_loss_threshold:
                track_loss += 1
        
        end_time = time.time()
        computation_time = (end_time - start_time) * 1000  # ms
        
        return BenchmarkResult(
            algorithm=algorithm_name,
            scenario=scenario_name,
            pos_rmse=np.sqrt(np.mean(np.array(pos_errors)**2)),
            vel_rmse=np.sqrt(np.mean(np.array(vel_errors)**2)),
            track_loss_frames=track_loss,
            total_frames=len(measurements),
            computation_time_ms=computation_time
        )
    
    def run_scenario(self, scenario_name: str, 
                    trajectory: List[TrajectoryPoint]) -> List[BenchmarkResult]:
        """Run all algorithms on a single scenario"""
        results = []
        
        # Add noise to create measurements
        measurements = self._add_noise(trajectory)
        
        # Test each algorithm
        algorithms = [
            ('EKF (Baseline)', lambda: BaselineEKF(self.dt)),
            ('IMM-2 (CV+CA)', lambda: IMM2Model(self.dt)),
            ('IMM-3 (CV+CA+CT)', lambda: IMM3Model(self.dt)),
            ('QEDMMA-Lite', lambda: QEDMMATracker(self.dt)),
        ]
        
        for name, create_tracker in algorithms:
            # Fresh tracker for each run
            tracker = create_tracker()
            
            # Run benchmark
            result = self._run_single_benchmark(
                tracker, measurements, trajectory, name, scenario_name
            )
            results.append(result)
            
            print(f"  {name:20s} | Pos RMSE: {result.pos_rmse:8.1f}m | "
                  f"Vel RMSE: {result.vel_rmse:6.1f}m/s | "
                  f"Track Loss: {result.track_loss_frames}/{result.total_frames}")
        
        return results
    
    def run_all_scenarios(self, duration: float = 10.0) -> Dict[str, List[BenchmarkResult]]:
        """Run all benchmark scenarios"""
        all_results = {}
        
        scenarios = [
            ('Hypersonic Cruise (Mach 8)', 
             lambda: TrajectoryGenerator.hypersonic_cruise(duration, self.dt, mach=8.0)),
            
            ('60g Pull-Up Maneuver', 
             lambda: TrajectoryGenerator.high_g_maneuver(duration, self.dt, mach=8.0, g_load=60.0)),
            
            ('Evasive S-Turn (Mach 6)', 
             lambda: TrajectoryGenerator.evasive_maneuver(duration, self.dt, mach=6.0)),
        ]
        
        print("=" * 80)
        print("QEDMMA-Lite Benchmark Suite")
        print("=" * 80)
        print()
        
        for scenario_name, gen_trajectory in scenarios:
            print(f"Scenario: {scenario_name}")
            print("-" * 60)
            
            trajectory = gen_trajectory()
            results = self.run_scenario(scenario_name, trajectory)
            all_results[scenario_name] = results
            
            print()
        
        return all_results


# =============================================================================
# Results Visualization
# =============================================================================

def plot_benchmark_results(results: Dict[str, List[BenchmarkResult]], 
                          output_path: str = None):
    """Generate comparison plots"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scenarios = list(results.keys())
    algorithms = [r.algorithm for r in results[scenarios[0]]]
    
    # Position RMSE comparison
    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.2
    
    for i, alg in enumerate(algorithms):
        rmses = [results[s][i].pos_rmse for s in scenarios]
        ax.bar(x + i*width, rmses, width, label=alg)
    
    ax.set_ylabel('Position RMSE (m)')
    ax.set_title('Position Accuracy Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Velocity RMSE comparison
    ax = axes[1]
    for i, alg in enumerate(algorithms):
        rmses = [results[s][i].vel_rmse for s in scenarios]
        ax.bar(x + i*width, rmses, width, label=alg)
    
    ax.set_ylabel('Velocity RMSE (m/s)')
    ax.set_title('Velocity Accuracy Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Track loss comparison
    ax = axes[2]
    for i, alg in enumerate(algorithms):
        losses = [100 * results[s][i].track_loss_frames / results[s][i].total_frames 
                 for s in scenarios]
        ax.bar(x + i*width, losses, width, label=alg)
    
    ax.set_ylabel('Track Loss (%)')
    ax.set_title('Track Loss Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def generate_markdown_report(results: Dict[str, List[BenchmarkResult]]) -> str:
    """Generate markdown benchmark report"""
    report = []
    report.append("# QEDMMA-Lite Benchmark Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Summary\n")
    
    for scenario, scenario_results in results.items():
        report.append(f"\n### {scenario}\n")
        report.append("| Algorithm | Pos RMSE (m) | Vel RMSE (m/s) | Track Loss (%) |")
        report.append("|-----------|-------------|----------------|----------------|")
        
        for r in scenario_results:
            track_loss_pct = 100 * r.track_loss_frames / max(1, r.total_frames)
            report.append(f"| {r.algorithm} | {r.pos_rmse:.1f} | {r.vel_rmse:.1f} | {track_loss_pct:.1f}% |")
    
    report.append("\n## Key Findings\n")
    
    # Find best performer
    best_algorithm = None
    best_avg_rmse = float('inf')
    
    algorithms = list(set(r.algorithm for r in list(results.values())[0]))
    for alg in algorithms:
        avg_rmse = np.mean([
            results[s][i].pos_rmse 
            for s in results 
            for i, r in enumerate(results[s]) 
            if r.algorithm == alg
        ])
        if avg_rmse < best_avg_rmse:
            best_avg_rmse = avg_rmse
            best_algorithm = alg
    
    report.append(f"- **Best Overall**: {best_algorithm}")
    
    # Compare to baseline
    baseline_rmses = [results[s][0].pos_rmse for s in results]
    qedmma_rmses = [results[s][-1].pos_rmse for s in results]
    improvement = np.mean(baseline_rmses) / np.mean(qedmma_rmses)
    
    report.append(f"- **QEDMMA vs EKF Improvement**: {improvement:.1f}x better accuracy")
    
    return "\n".join(report)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='QEDMMA-Lite Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --all                    Run all scenarios
  python benchmark.py --scenario hypersonic    Run hypersonic scenario only
  python benchmark.py --plot --output results  Generate plots
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all scenarios')
    parser.add_argument('--scenario', type=str, choices=['hypersonic', 'maneuver', 'evasive'],
                       help='Run specific scenario')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration (s)')
    parser.add_argument('--dt', type=float, default=0.0625, help='Time step (s)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', type=str, default='benchmark_results', 
                       help='Output filename prefix')
    parser.add_argument('--json', action='store_true', help='Export results as JSON')
    
    args = parser.parse_args()
    
    # Create benchmark engine
    engine = BenchmarkEngine(dt=args.dt)
    
    # Run benchmarks
    if args.all or args.scenario is None:
        results = engine.run_all_scenarios(duration=args.duration)
    else:
        # Run specific scenario
        scenario_map = {
            'hypersonic': ('Hypersonic Cruise', 
                          lambda: TrajectoryGenerator.hypersonic_cruise(args.duration, args.dt)),
            'maneuver': ('60g Pull-Up', 
                        lambda: TrajectoryGenerator.high_g_maneuver(args.duration, args.dt)),
            'evasive': ('Evasive S-Turn', 
                       lambda: TrajectoryGenerator.evasive_maneuver(args.duration, args.dt))
        }
        name, gen = scenario_map[args.scenario]
        print(f"Scenario: {name}")
        print("-" * 60)
        trajectory = gen()
        results = {name: engine.run_scenario(name, trajectory)}
    
    # Generate outputs
    if args.plot:
        plot_benchmark_results(results, f"{args.output}.png")
    
    if args.json:
        json_results = {
            scenario: [r.to_dict() for r in res]
            for scenario, res in results.items()
        }
        with open(f"{args.output}.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"JSON results saved to: {args.output}.json")
    
    # Generate markdown report
    report = generate_markdown_report(results)
    with open(f"{args.output}.md", 'w') as f:
        f.write(report)
    print(f"Markdown report saved to: {args.output}.md")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
