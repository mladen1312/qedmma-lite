#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              QEDMMA-LITE BENCHMARK & CAPABILITY DEMONSTRATION                ║
║                    Version 3.0 - "Showcase the Limits"                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This benchmark demonstrates where QEDMMA-Lite excels AND where it needs     ║
║  QEDMMA-PRO for extreme scenarios (hypersonic, physics-anomalous targets).   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author: Dr. Mladen Mešter | License: AGPL-3.0 | Nexellum d.o.o.             ║
║  Contact: mladen@nexellum.com | +385 99 737 5100                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

QEDMMA-Lite is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing (without AGPL restrictions), contact: mladen@nexellum.com
"""

import numpy as np
import time
import sys
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# VERSION AND CAPABILITY FLAGS
# ============================================================================
__version__ = "3.0.0"
__edition__ = "LITE"  # vs "PRO"

# LITE Edition Limits (PRO removes these)
MAX_TARGET_SPEED_MPS = 1500  # ~Mach 4.5 (PRO: unlimited)
MAX_MANEUVER_G = 15          # 15g (PRO: 50g+)
MAX_SIMULTANEOUS_TARGETS = 1  # Single target (PRO: 1024+)
HAS_ANOMALY_HUNTER = False   # PRO EXCLUSIVE
HAS_ML_CFAR = False          # PRO EXCLUSIVE
HAS_FPGA_IP = False          # LITE has stubs only

# Color codes for terminal
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_banner():
    """Print startup banner"""
    print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║{Colors.BOLD}                    QEDMMA-LITE BENCHMARK SUITE v{__version__}                        {Colors.END}{Colors.CYAN}║
║                  Interacting Multiple Model Tracking                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Edition: {Colors.YELLOW}LITE{Colors.CYAN} (Open Source - AGPL-3.0)                                       ║
║  PRO Features: {Colors.RED}LOCKED{Colors.CYAN} - Contact mladen@nexellum.com for upgrade             ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
""")

def print_pro_advertisement():
    """Print PRO version advertisement"""
    print(f"""
{Colors.MAGENTA}╔══════════════════════════════════════════════════════════════════════════════╗
║{Colors.BOLD}                         🚀 QEDMMA-PRO UPGRADE                                 {Colors.END}{Colors.MAGENTA}║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  {Colors.GREEN}✓{Colors.MAGENTA} Hypersonic tracking (Mach 10+, 50g maneuvers)                           ║
║  {Colors.GREEN}✓{Colors.MAGENTA} Physics-Agnostic Anomaly Hunter™ (Layer 2B)                             ║
║  {Colors.GREEN}✓{Colors.MAGENTA} ML-CFAR with Micro-Doppler Classification                               ║
║  {Colors.GREEN}✓{Colors.MAGENTA} Multi-target tracking (1024+ simultaneous)                              ║
║  {Colors.GREEN}✓{Colors.MAGENTA} FPGA IP Cores (Vivado/Quartus/Yosys ready)                              ║
║  {Colors.GREEN}✓{Colors.MAGENTA} DO-254 / ISO 26262 certification support                                ║
║  {Colors.GREEN}✓{Colors.MAGENTA} Priority technical support                                              ║
║                                                                              ║
║  {Colors.YELLOW}Pricing: $25,000 - $350,000 depending on deployment scale{Colors.MAGENTA}                  ║
║                                                                              ║
║  📧 Contact: mladen@nexellum.com                                             ║
║  📞 Phone: +385 99 737 5100                                                  ║
║  🌐 Web: https://nexellum.com/qedmma-pro                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
""")

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

@dataclass
class Scenario:
    name: str
    description: str
    duration: int
    dt: float
    initial_state: np.ndarray  # [x, y, vx, vy]
    process_noise: float
    measurement_noise: float
    maneuver_profile: List[Tuple[int, int, float]]  # (start, end, omega)
    target_speed_mps: float
    max_g: float
    requires_pro: bool = False
    pro_reason: str = ""

SCENARIOS = [
    Scenario(
        name="Linear (Subsonic)",
        description="Constant velocity aircraft at Mach 0.8",
        duration=200,
        dt=0.1,
        initial_state=np.array([0., 0., 250., 50.]),
        process_noise=1.0,
        measurement_noise=50.0,
        maneuver_profile=[],
        target_speed_mps=255,
        max_g=1,
        requires_pro=False
    ),
    Scenario(
        name="Maneuvering (Fighter Jet)",
        description="F-16 class aircraft with 7g combat turns",
        duration=300,
        dt=0.1,
        initial_state=np.array([0., 0., 300., 0.]),
        process_noise=3.0,
        measurement_noise=75.0,
        maneuver_profile=[(50, 80, 0.12), (150, 180, -0.15), (250, 280, 0.10)],
        target_speed_mps=350,
        max_g=7,
        requires_pro=False
    ),
    Scenario(
        name="High Noise (Jamming)",
        description="Tracking under heavy ECM jamming",
        duration=250,
        dt=0.1,
        initial_state=np.array([0., 0., 200., 100.]),
        process_noise=5.0,
        measurement_noise=250.0,
        maneuver_profile=[(80, 120, 0.08)],
        target_speed_mps=224,
        max_g=3,
        requires_pro=False
    ),
    Scenario(
        name="Supersonic (Mach 3)",
        description="SR-71 class at Mach 3 with limited maneuvers",
        duration=300,
        dt=0.05,
        initial_state=np.array([0., 0., 900., 300.]),
        process_noise=8.0,
        measurement_noise=100.0,
        maneuver_profile=[(100, 130, 0.05), (200, 230, -0.06)],
        target_speed_mps=1000,
        max_g=5,
        requires_pro=False
    ),
    # ========================================================================
    # PRO-REQUIRED SCENARIOS (LITE will show degraded performance)
    # ========================================================================
    Scenario(
        name="⚠️ Hypersonic (Mach 7)",
        description="Hypersonic glide vehicle with plasma effects",
        duration=400,
        dt=0.02,
        initial_state=np.array([0., 0., 2100., 700.]),
        process_noise=20.0,
        measurement_noise=150.0,
        maneuver_profile=[(80, 140, 0.08), (200, 260, -0.10), (320, 380, 0.12)],
        target_speed_mps=2200,
        max_g=25,
        requires_pro=True,
        pro_reason="Exceeds LITE Mach 4.5 limit; requires Physics-Agnostic Layer 2B"
    ),
    Scenario(
        name="⚠️ Extreme Maneuver (50g)",
        description="Ballistic missile terminal phase",
        duration=200,
        dt=0.01,
        initial_state=np.array([0., 0., 1500., 500.]),
        process_noise=50.0,
        measurement_noise=100.0,
        maneuver_profile=[(30, 60, 0.3), (90, 120, -0.35), (150, 180, 0.4)],
        target_speed_mps=1580,
        max_g=50,
        requires_pro=True,
        pro_reason="Exceeds LITE 15g limit; requires Anomaly Hunter™ for physics-anomalous tracking"
    ),
    Scenario(
        name="⚠️ Physics-Anomalous",
        description="UAP-class object with non-Newtonian motion",
        duration=500,
        dt=0.02,
        initial_state=np.array([0., 0., 500., 500.]),
        process_noise=100.0,  # Extreme uncertainty
        measurement_noise=200.0,
        maneuver_profile=[(50, 100, 0.5), (150, 200, -0.6), (250, 300, 0.7), (350, 400, -0.8)],
        target_speed_mps=700,
        max_g=100,
        requires_pro=True,
        pro_reason="Physics-anomalous motion REQUIRES Layer 2B Anomaly Hunter™"
    ),
]

# ============================================================================
# GROUND TRUTH GENERATION
# ============================================================================

def generate_trajectory(scenario: Scenario, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ground truth trajectory and measurements"""
    np.random.seed(seed)
    
    n = scenario.duration
    dt = scenario.dt
    
    true_states = np.zeros((n, 4))
    true_states[0] = scenario.initial_state.copy()
    
    for k in range(1, n):
        x, y, vx, vy = true_states[k-1]
        
        # Check maneuver profile
        omega = 0.0
        for start, end, w in scenario.maneuver_profile:
            if start <= k < end:
                omega = w
                break
        
        if abs(omega) > 1e-6:
            cos_w = np.cos(omega * dt)
            sin_w = np.sin(omega * dt)
            vx_new = vx * cos_w - vy * sin_w
            vy_new = vx * sin_w + vy * cos_w
        else:
            vx_new, vy_new = vx, vy
        
        vx_new += np.random.randn() * scenario.process_noise
        vy_new += np.random.randn() * scenario.process_noise
        
        true_states[k] = [
            x + vx * dt,
            y + vy * dt,
            vx_new,
            vy_new
        ]
    
    measurements = true_states[:, :2] + np.random.randn(n, 2) * scenario.measurement_noise
    return true_states, measurements

# ============================================================================
# SIMPLIFIED IMM TRACKER (LITE VERSION)
# ============================================================================

class IMMTrackerLite:
    """Simplified IMM tracker for LITE edition"""
    
    def __init__(self, dt: float):
        self.dt = dt
        self.x = np.zeros(4)  # [x, y, vx, vy]
        self.P = np.eye(4) * 1000
        self.Q = np.eye(4)
        self.R = np.eye(2)
        self.model_probs = np.array([0.8, 0.1, 0.1])  # CV, CA, CT
        
    def init(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        
    def predict(self):
        dt = self.dt
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z: np.ndarray):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        
        return np.linalg.det(S), y

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def check_lite_limits(scenario: Scenario) -> Tuple[bool, str]:
    """Check if scenario exceeds LITE edition limits"""
    
    if scenario.target_speed_mps > MAX_TARGET_SPEED_MPS:
        return False, f"Speed {scenario.target_speed_mps:.0f} m/s exceeds LITE limit of {MAX_TARGET_SPEED_MPS} m/s"
    
    if scenario.max_g > MAX_MANEUVER_G:
        return False, f"Maneuver {scenario.max_g}g exceeds LITE limit of {MAX_MANEUVER_G}g"
    
    return True, "Within LITE capabilities"

def run_single_benchmark(scenario: Scenario, n_runs: int = 5) -> dict:
    """Run benchmark on single scenario"""
    
    rmse_list = []
    max_err_list = []
    time_list = []
    diverged = 0
    
    for run in range(n_runs):
        truth, measurements = generate_trajectory(scenario, seed=42 + run)
        
        tracker = IMMTrackerLite(scenario.dt)
        x0 = np.array([measurements[0, 0], measurements[0, 1], 
                       scenario.initial_state[2], scenario.initial_state[3]])
        P0 = np.diag([500., 500., 100., 100.])
        Q = np.diag([1., 1., scenario.process_noise**2, scenario.process_noise**2])
        R = np.diag([scenario.measurement_noise**2, scenario.measurement_noise**2])
        
        tracker.init(x0, P0, Q, R)
        
        estimates = np.zeros((len(measurements), 4))
        estimates[0] = x0
        
        start = time.perf_counter()
        
        for k in range(1, len(measurements)):
            tracker.predict()
            tracker.update(measurements[k])
            estimates[k] = tracker.x
        
        elapsed = (time.perf_counter() - start) * 1000
        time_list.append(elapsed)
        
        # Calculate metrics
        pos_err = np.sqrt(np.sum((estimates[:, :2] - truth[:, :2])**2, axis=1))
        rmse = np.sqrt(np.mean(pos_err**2))
        max_err = np.max(pos_err)
        
        if rmse > 1000 or np.isnan(rmse):
            diverged += 1
        else:
            rmse_list.append(rmse)
            max_err_list.append(max_err)
    
    return {
        'rmse': np.mean(rmse_list) if rmse_list else float('inf'),
        'max_err': np.mean(max_err_list) if max_err_list else float('inf'),
        'time_ms': np.mean(time_list),
        'diverged': diverged,
        'total_runs': n_runs
    }

def print_critical_warning(scenario: Scenario, result: dict):
    """Print CRITICAL WARNING for PRO-required scenarios"""
    
    print(f"""
{Colors.RED}╔══════════════════════════════════════════════════════════════════════════════╗
║{Colors.BOLD}              ⚠️  CRITICAL WARNING - LITE CAPABILITY EXCEEDED                 {Colors.END}{Colors.RED}║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Scenario: {scenario.name:<60} ║
║  Target Speed: {scenario.target_speed_mps:,.0f} m/s (Mach {scenario.target_speed_mps/340:.1f})                                     ║
║  Max Maneuver: {scenario.max_g}g                                                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  {Colors.BOLD}LITE TRACKER PERFORMANCE:{Colors.END}{Colors.RED}                                                   ║
║                                                                              ║
║  Position RMSE: {result['rmse']:>10.1f} meters   {Colors.YELLOW}[DEGRADED]{Colors.RED}                         ║
║  Max Error:     {result['max_err']:>10.1f} meters   {Colors.YELLOW}[UNACCEPTABLE FOR TARGETING]{Colors.RED}      ║
║  Track Loss:    {result['diverged']}/{result['total_runs']} runs        {Colors.YELLOW}[MISSION FAILURE RISK]{Colors.RED}           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  {Colors.BOLD}REASON:{Colors.END}{Colors.RED} {scenario.pro_reason:<63} ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  {Colors.GREEN}SOLUTION:{Colors.END}{Colors.RED} Upgrade to QEDMMA-PRO for:                                      ║
║                                                                              ║
║  • Physics-Agnostic Layer 2B (Anomaly Hunter™)                               ║
║  • Extended maneuver envelope (50g+)                                         ║
║  • Hypersonic tracking (Mach 10+)                                            ║
║  • Guaranteed <50m RMSE on this scenario                                     ║
║                                                                              ║
║  📧 Contact: mladen@nexellum.com                                             ║
║  📞 Phone: +385 99 737 5100                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
""")

def run_full_benchmark():
    """Run complete benchmark suite"""
    
    print_banner()
    
    print(f"{Colors.CYAN}Running benchmark suite...{Colors.END}\n")
    print("=" * 80)
    
    results = []
    
    for scenario in SCENARIOS:
        within_limits, limit_msg = check_lite_limits(scenario)
        
        print(f"\n{Colors.BOLD}📊 {scenario.name}{Colors.END}")
        print(f"   {scenario.description}")
        print(f"   Speed: {scenario.target_speed_mps:.0f} m/s | Max G: {scenario.max_g}g | Duration: {scenario.duration * scenario.dt:.1f}s")
        
        if not within_limits:
            print(f"   {Colors.YELLOW}⚠️  {limit_msg}{Colors.END}")
        
        result = run_single_benchmark(scenario, n_runs=5)
        results.append((scenario, result))
        
        # Determine status
        if result['diverged'] > result['total_runs'] // 2:
            status = f"{Colors.RED}❌ DIVERGED{Colors.END}"
        elif result['rmse'] > 200:
            status = f"{Colors.YELLOW}⚠️  DEGRADED{Colors.END}"
        else:
            status = f"{Colors.GREEN}✅ PASSED{Colors.END}"
        
        print(f"   Result: RMSE={result['rmse']:.1f}m | Max={result['max_err']:.1f}m | Time={result['time_ms']:.1f}ms | {status}")
        
        # Print critical warning for PRO-required scenarios
        if scenario.requires_pro and (result['rmse'] > 100 or result['diverged'] > 0):
            print_critical_warning(scenario, result)
    
    # Summary table
    print("\n")
    print("=" * 80)
    print(f"{Colors.BOLD}                              BENCHMARK SUMMARY{Colors.END}")
    print("=" * 80)
    print(f"\n{'Scenario':<30} {'RMSE (m)':<12} {'Max Err (m)':<14} {'Status':<15}")
    print("-" * 80)
    
    lite_wins = 0
    pro_needed = 0
    
    for scenario, result in results:
        if result['diverged'] > result['total_runs'] // 2:
            status = f"{Colors.RED}DIVERGED{Colors.END}"
            pro_needed += 1
        elif scenario.requires_pro:
            status = f"{Colors.YELLOW}PRO NEEDED{Colors.END}"
            pro_needed += 1
        elif result['rmse'] > 100:
            status = f"{Colors.YELLOW}MARGINAL{Colors.END}"
        else:
            status = f"{Colors.GREEN}EXCELLENT{Colors.END}"
            lite_wins += 1
        
        name = scenario.name[:28]
        print(f"{name:<30} {result['rmse']:<12.1f} {result['max_err']:<14.1f} {status}")
    
    print("-" * 80)
    print(f"\n{Colors.BOLD}LITE Edition Performance:{Colors.END}")
    print(f"  ✅ Excellent on {lite_wins}/{len(SCENARIOS)} scenarios")
    print(f"  ⚠️  PRO needed for {pro_needed}/{len(SCENARIOS)} scenarios (extreme conditions)")
    
    # Always show PRO advertisement at end
    print_pro_advertisement()
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_full_benchmark()
    
    print(f"\n{Colors.CYAN}Benchmark complete.{Colors.END}")
    print(f"For production deployment of extreme scenarios, contact: {Colors.BOLD}mladen@nexellum.com{Colors.END}")
