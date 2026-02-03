#!/usr/bin/env python3
"""
QEDMMA Fer Benchmark Visualization
===================================
Generates publication-quality plots showing IMM advantage.

Copyright (c) 2026 Dr. Mladen Mešter / Nexellum d.o.o.
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from fer_benchmark import (
    CONFIG, generate_trajectory, generate_measurements,
    StandardEKF, IMMFilter
)

def run_single_trial_with_viz():
    """Run single trial and collect data for visualization."""
    rng = np.random.default_rng(CONFIG['seed_base'])
    
    # Generate trajectory and measurements
    times, true_states, phases = generate_trajectory(CONFIG)
    measurements = generate_measurements(true_states, CONFIG['meas_noise_std'], rng)
    
    # Calculate turn rate
    omega = 9.81 * CONFIG['g_load'] / CONFIG['velocity']
    
    # Initialize trackers
    ekf_low_q = StandardEKF(CONFIG['dt'], process_noise=0.1, meas_noise=CONFIG['meas_noise_std'])
    ekf_high_q = StandardEKF(CONFIG['dt'], process_noise=5.0, meas_noise=CONFIG['meas_noise_std'])
    imm = IMMFilter(CONFIG['dt'], CONFIG['meas_noise_std'], turn_rate=omega)
    
    # Initialize
    z0 = measurements[0]
    ekf_low_q.initialize(z0)
    ekf_high_q.initialize(z0)
    imm.initialize(z0)
    
    # Storage
    estimates = {
        'ekf_low_q': [ekf_low_q.x.copy()],
        'ekf_high_q': [ekf_high_q.x.copy()],
        'imm': [imm.filters[0]['x'].copy()],
    }
    errors = {
        'ekf_low_q': [0],
        'ekf_high_q': [0],
        'imm': [0],
    }
    mode_probs = [[0.6, 0.2, 0.2]]  # Initial
    
    for i in range(1, len(measurements)):
        z = measurements[i]
        true = true_states[i]
        
        # EKF Low-Q
        x_low = ekf_low_q.step(z)
        estimates['ekf_low_q'].append(x_low.copy())
        errors['ekf_low_q'].append(np.linalg.norm(x_low[:2] - true[:2]))
        
        # EKF High-Q
        x_high = ekf_high_q.step(z)
        estimates['ekf_high_q'].append(x_high.copy())
        errors['ekf_high_q'].append(np.linalg.norm(x_high[:2] - true[:2]))
        
        # IMM
        x_imm, probs = imm.step(z)
        estimates['imm'].append(x_imm.copy())
        errors['imm'].append(np.linalg.norm(x_imm[:2] - true[:2]))
        mode_probs.append(probs.copy())
    
    # Convert to arrays
    for k in estimates:
        estimates[k] = np.array(estimates[k])
        errors[k] = np.array(errors[k])
    mode_probs = np.array(mode_probs)
    
    return times, true_states, measurements, estimates, errors, mode_probs, phases


def create_visualization():
    """Create publication-quality benchmark visualization."""
    times, true_states, measurements, estimates, errors, mode_probs, phases = run_single_trial_with_viz()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Color scheme
    colors = {
        'true': '#2C3E50',
        'ekf_low_q': '#E74C3C',
        'ekf_high_q': '#F39C12',
        'imm': '#27AE60',
        'meas': '#BDC3C7',
    }
    
    # ==== Plot 1: Trajectory ====
    ax1 = fig.add_subplot(2, 2, 1)
    
    # True trajectory
    ax1.plot(true_states[:, 0], true_states[:, 1], 
             color=colors['true'], linewidth=2, label='Ground Truth', zorder=5)
    
    # Measurements
    ax1.scatter(measurements[::5, 0], measurements[::5, 1], 
                color=colors['meas'], s=10, alpha=0.5, label='Measurements', zorder=1)
    
    # Estimates
    ax1.plot(estimates['ekf_low_q'][:, 0], estimates['ekf_low_q'][:, 1],
             color=colors['ekf_low_q'], linewidth=1.5, linestyle='--', 
             label='EKF Low-Q', alpha=0.8, zorder=3)
    ax1.plot(estimates['ekf_high_q'][:, 0], estimates['ekf_high_q'][:, 1],
             color=colors['ekf_high_q'], linewidth=1.5, linestyle=':', 
             label='EKF High-Q', alpha=0.8, zorder=3)
    ax1.plot(estimates['imm'][:, 0], estimates['imm'][:, 1],
             color=colors['imm'], linewidth=2, 
             label='IMM (QEDMMA)', zorder=4)
    
    # Mark maneuver region
    turn_start_idx = int(CONFIG['turn_start'] / CONFIG['dt'])
    turn_end_idx = int(CONFIG['turn_end'] / CONFIG['dt'])
    ax1.scatter(true_states[turn_start_idx, 0], true_states[turn_start_idx, 1],
                marker='o', s=100, color='black', zorder=6, label='Maneuver Start')
    ax1.scatter(true_states[turn_end_idx, 0], true_states[turn_end_idx, 1],
                marker='s', s=100, color='black', zorder=6, label='Maneuver End')
    
    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title('2D Trajectory: 6g Coordinated Turn', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # ==== Plot 2: Position Error over Time ====
    ax2 = fig.add_subplot(2, 2, 2)
    
    ax2.fill_between(times, 0, 500, where=[p == 'ct' for p in phases],
                     color='yellow', alpha=0.2, label='Maneuver Region')
    
    ax2.plot(times, errors['ekf_low_q'], color=colors['ekf_low_q'], 
             linewidth=1.5, label='EKF Low-Q')
    ax2.plot(times, errors['ekf_high_q'], color=colors['ekf_high_q'], 
             linewidth=1.5, label='EKF High-Q')
    ax2.plot(times, errors['imm'], color=colors['imm'], 
             linewidth=2, label='IMM (QEDMMA)')
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Position Error (m)', fontsize=11)
    ax2.set_title('Position Error: IMM vs Single-Model EKF', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(500, max(errors['ekf_low_q']) * 1.1))
    
    # Add RMSE annotations
    rmse_low = np.sqrt(np.mean(errors['ekf_low_q']**2))
    rmse_high = np.sqrt(np.mean(errors['ekf_high_q']**2))
    rmse_imm = np.sqrt(np.mean(errors['imm']**2))
    ax2.text(0.02, 0.95, f'RMSE: EKF Low-Q={rmse_low:.1f}m, High-Q={rmse_high:.1f}m, IMM={rmse_imm:.1f}m',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ==== Plot 3: IMM Mode Probabilities ====
    ax3 = fig.add_subplot(2, 2, 3)
    
    ax3.fill_between(times, 0, 1, where=[p == 'ct' for p in phases],
                     color='yellow', alpha=0.2, label='True Maneuver')
    
    ax3.plot(times, mode_probs[:, 0], color='#3498DB', linewidth=2, label='P(CV)')
    ax3.plot(times, mode_probs[:, 1], color='#9B59B6', linewidth=2, label='P(CA)')
    ax3.plot(times, mode_probs[:, 2], color='#E67E22', linewidth=2, label='P(CT)')
    
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Mode Probability', fontsize=11)
    ax3.set_title('IMM Automatic Mode Switching', fontsize=12, fontweight='bold')
    ax3.legend(loc='right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # Annotations
    ax3.annotate('IMM detects\nmaneuver onset', xy=(5.5, 0.7), xytext=(7, 0.4),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))
    ax3.annotate('Returns to CV\nafter turn', xy=(15.5, 0.8), xytext=(17, 0.5),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))
    
    # ==== Plot 4: Summary Bar Chart ====
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Calculate segment-specific RMSE
    cv_mask = np.array([p == 'cv' for p in phases])
    ct_mask = np.array([p == 'ct' for p in phases])
    
    trackers = ['EKF\nLow-Q', 'EKF\nHigh-Q', 'IMM\n(QEDMMA)']
    cv_rmse = [
        np.sqrt(np.mean(errors['ekf_low_q'][cv_mask]**2)),
        np.sqrt(np.mean(errors['ekf_high_q'][cv_mask]**2)),
        np.sqrt(np.mean(errors['imm'][cv_mask]**2)),
    ]
    ct_rmse = [
        np.sqrt(np.mean(errors['ekf_low_q'][ct_mask]**2)),
        np.sqrt(np.mean(errors['ekf_high_q'][ct_mask]**2)),
        np.sqrt(np.mean(errors['imm'][ct_mask]**2)),
    ]
    
    x = np.arange(len(trackers))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, cv_rmse, width, label='CV Segments', color='#3498DB')
    bars2 = ax4.bar(x + width/2, ct_rmse, width, label='Turn Segment', color='#E74C3C')
    
    ax4.set_xlabel('Tracker', fontsize=11)
    ax4.set_ylabel('Position RMSE (m)', fontsize=11)
    ax4.set_title('Segment-Specific Performance', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(trackers)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.0f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height < 500:  # Only label if reasonable
            ax4.annotate(f'{height:.0f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    
    # Clip y-axis for readability
    ax4.set_ylim(0, min(200, max(max(cv_rmse), max(ct_rmse)) * 1.2))
    
    # Overall title
    fig.suptitle('QEDMMA Fer Benchmark: Why IMM Matters', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('fer_benchmark_results.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Saved: fer_benchmark_results.png")
    
    plt.savefig('fer_benchmark_results.svg', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Saved: fer_benchmark_results.svg")
    
    return fig


if __name__ == "__main__":
    fig = create_visualization()
    print("\n📊 Visualization complete!")
    print("Files generated:")
    print("  • fer_benchmark_results.png (for GitHub/web)")
    print("  • fer_benchmark_results.svg (for publications)")
