"""
QEDMMA v3.1 — Unit Tests
"""
import numpy as np
import pytest
from qedmma import QEDMMAv31, QEDMMAConfig, TrackingMode


def generate_scenario(seed=42, n=201, dt=0.1, v=300.0, g=6.0, noise=2.5):
    """Generate maneuvering target test scenario."""
    rng = np.random.default_rng(seed)
    omega = 9.81 * g / v
    
    true = np.zeros((n, 4))
    x, y, vx, vy = 0.0, 0.0, v, 0.0
    
    for i in range(n):
        t = i * dt
        true[i] = [x, y, vx, vy]
        if 5.0 <= t < 15.0:
            cw, sw = np.cos(omega * dt), np.sin(omega * dt)
            vx, vy = vx * cw - vy * sw, vx * sw + vy * cw
        x += vx * dt
        y += vy * dt
    
    meas = true[:, :2] + rng.normal(0, noise, (n, 2))
    return true, meas


def rmse(x_est, x_true, start=2):
    n = min(len(x_est), len(x_true))
    errs = [np.linalg.norm(x_est[k, :2] - x_true[k, :2]) for k in range(start, n)]
    return np.sqrt(np.mean(np.array(errs)**2))


class TestQEDMMAv31:
    """Test suite for QEDMMA v3.1 tracker."""
    
    def test_basic_tracking(self):
        """Test basic tracking functionality."""
        true, meas = generate_scenario(seed=42)
        cfg = QEDMMAConfig(omega=9.81*6.0/300.0)
        tracker = QEDMMAv31(cfg)
        
        x_filt, x_smooth = tracker.process_batch(meas)
        
        assert x_filt.shape == (len(meas), 4)
        assert x_smooth.shape == (len(meas), 4)
    
    def test_smoother_improvement(self):
        """Test that smoother improves over filter."""
        true, meas = generate_scenario(seed=42)
        cfg = QEDMMAConfig(omega=9.81*6.0/300.0)
        tracker = QEDMMAv31(cfg)
        
        x_filt, x_smooth = tracker.process_batch(meas)
        
        filt_rmse = rmse(x_filt, true)
        smooth_rmse = rmse(x_smooth, true)
        
        improvement = (filt_rmse - smooth_rmse) / filt_rmse * 100
        
        assert improvement > 40, f"Expected >40% improvement, got {improvement:.1f}%"
    
    def test_monte_carlo_6g(self):
        """Monte Carlo test for 6g scenario."""
        improvements = []
        
        for seed in range(10):
            true, meas = generate_scenario(seed=seed, g=6.0)
            cfg = QEDMMAConfig(omega=9.81*6.0/300.0)
            tracker = QEDMMAv31(cfg)
            
            x_filt, x_smooth = tracker.process_batch(meas)
            
            filt_rmse = rmse(x_filt, true)
            smooth_rmse = rmse(x_smooth, true)
            improvement = (filt_rmse - smooth_rmse) / filt_rmse * 100
            improvements.append(improvement)
        
        mean_imp = np.mean(improvements)
        assert mean_imp > 35, f"Expected mean >35%, got {mean_imp:.1f}%"
    
    def test_omega_from_g(self):
        """Test omega calculation from g-load."""
        omega = QEDMMAv31.omega_from_g(6.0, 300.0)
        expected = 9.81 * 6.0 / 300.0
        assert abs(omega - expected) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
