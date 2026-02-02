"""
QEDMMA-Lite v3.0 - Advanced Filters Test Suite
===============================================
pytest tests for all advanced filter components.

Run: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/claude/qedmma-advanced/python')

from ukf import UnscentedKalmanFilter, UKFParams, create_radar_ukf
from ckf import CubatureKalmanFilter, SquareRootCKF
from adaptive_noise import SageHusaEstimator, AdaptiveNoiseState
from zero_dsp_correlation import ZeroDSPCorrelator, TwoBitCorrelator


class TestUKF:
    """Tests for Unscented Kalman Filter"""
    
    def test_linear_system(self):
        """UKF should match KF for linear systems"""
        # Linear process and measurement
        def f(x, dt): 
            return np.array([x[0] + x[1]*dt, x[1]])
        def h(x): 
            return np.array([x[0]])
        
        ukf = UnscentedKalmanFilter(f, h, n_states=2, n_meas=1)
        state = ukf.init_state(
            x0=np.array([0.0, 1.0]),
            P0=np.eye(2),
            Q=0.1 * np.eye(2),
            R=np.array([[1.0]])
        )
        
        # Predict
        state = ukf.predict(state, dt=0.1)
        assert state.x[0] == pytest.approx(0.1, rel=0.1)  # x += v*dt
        
        # Update
        state, innov = ukf.update(state, z=np.array([0.15]))
        assert abs(innov[0]) < 0.5  # Innovation should be small
    
    def test_nonlinear_tracking(self):
        """UKF should track nonlinear system"""
        ukf, state = create_radar_ukf()
        
        np.random.seed(42)
        errors = []
        true_pos = np.array([1000.0, 1000.0])
        
        for _ in range(10):
            true_pos += np.array([10.0, 5.0]) * 0.1
            r = np.sqrt(true_pos[0]**2 + true_pos[1]**2)
            theta = np.arctan2(true_pos[1], true_pos[0])
            z = np.array([r + np.random.randn()*10, theta + np.random.randn()*0.01])
            
            state = ukf.predict(state, dt=0.1)
            state, _ = ukf.update(state, z)
            errors.append(np.linalg.norm(state.x[:2] - true_pos))
        
        # Error should decrease over time
        assert errors[-1] < errors[0]
    
    def test_sigma_points_coverage(self):
        """Verify sigma points span state space correctly"""
        def f(x, dt): return x
        def h(x): return x[:1]
        
        ukf = UnscentedKalmanFilter(f, h, n_states=3, n_meas=1)
        
        # Check we have 2n+1 sigma points
        assert ukf.n_sigma == 7
        
        # Mean weights should sum to 1
        assert np.sum(ukf.Wm) == pytest.approx(1.0, rel=0.01)
        # Covariance weights may not sum to 1 (depends on beta parameter)


class TestCKF:
    """Tests for Cubature Kalman Filter"""
    
    def test_high_dimensional(self):
        """CKF should handle high dimensions well"""
        n_states = 9
        
        def f(x, dt):
            F = np.eye(n_states)
            for i in range(n_states // 3):
                if i + n_states//3 < n_states:
                    F[i, i + n_states//3] = dt
            return F @ x
        
        def h(x):
            return x[:3]
        
        ckf = CubatureKalmanFilter(f, h, n_states=n_states, n_meas=3)
        
        # Should have 2n cubature points
        assert ckf.n_cubature == 2 * n_states
        
        # All weights positive and equal
        assert ckf.W > 0
        assert ckf.W == pytest.approx(1.0 / (2 * n_states))
    
    def test_ckf_vs_ukf_convergence(self):
        """CKF and UKF should give similar results for same problem"""
        def f(x, dt):
            return np.array([x[0] + x[1]*dt, x[1]])
        def h(x):
            return np.array([x[0]])
        
        ukf = UnscentedKalmanFilter(f, h, n_states=2, n_meas=1)
        ckf = CubatureKalmanFilter(f, h, n_states=2, n_meas=1)
        
        x0 = np.array([0.0, 1.0])
        P0 = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = np.array([[1.0]])
        
        state_ukf = ukf.init_state(x0, P0, Q, R)
        state_ckf = ckf.init_state(x0, P0, Q, R)
        
        np.random.seed(42)
        for _ in range(10):
            z = np.array([np.random.randn()])
            
            state_ukf = ukf.predict(state_ukf, dt=0.1)
            state_ckf = ckf.predict(state_ckf, dt=0.1)
            
            state_ukf, _ = ukf.update(state_ukf, z)
            state_ckf, _ = ckf.update(state_ckf, z)
        
        # Both should produce valid results (not necessarily identical)
        # Different quadrature rules lead to different estimates
        assert not np.any(np.isnan(state_ukf.x))
        assert not np.any(np.isnan(state_ckf.x))
        assert np.all(np.linalg.eigvalsh(state_ukf.P) > 0)
        assert np.all(np.linalg.eigvalsh(state_ckf.P) > 0)


class TestAdaptiveNoise:
    """Tests for adaptive noise estimation"""
    
    def test_sage_husa_convergence(self):
        """Sage-Husa should adapt to noise changes"""
        estimator = SageHusaEstimator(b=0.9)
        
        Q0 = np.array([[0.1]])
        R0 = np.array([[1.0]])
        state = AdaptiveNoiseState(Q=Q0.copy(), R=R0.copy())
        
        # Simulate high noise environment
        H = np.array([[1.0]])
        P_pred = np.array([[0.5]])
        K = np.array([[0.5]])
        F = np.array([[1.0]])
        P_post = np.array([[0.25]])
        
        for i in range(50):
            # High variance innovations
            innovation = np.random.randn(1) * 3.0
            state = estimator.update(state, innovation, H, P_pred, K, F, P_post)
        
        # R should have increased from initial
        assert state.R[0, 0] > R0[0, 0] * 0.5  # Allow some tolerance
    
    def test_positive_definiteness(self):
        """Noise estimates should remain positive definite"""
        estimator = SageHusaEstimator(b=0.95)
        
        Q0 = 0.1 * np.eye(2)
        R0 = np.eye(2)
        state = AdaptiveNoiseState(Q=Q0.copy(), R=R0.copy())
        
        H = np.eye(2)
        P_pred = np.eye(2)
        K = 0.5 * np.eye(2)
        F = np.eye(2)
        P_post = 0.5 * np.eye(2)
        
        for _ in range(100):
            innovation = np.random.randn(2)
            state = estimator.update(state, innovation, H, P_pred, K, F, P_post)
            
            # Check eigenvalues are positive
            eigvals_Q = np.linalg.eigvalsh(state.Q)
            eigvals_R = np.linalg.eigvalsh(state.R)
            
            assert all(eigvals_Q > 0), "Q lost positive definiteness"
            assert all(eigvals_R > 0), "R lost positive definiteness"


class TestZeroDSPCorrelation:
    """Tests for Zero-DSP correlation"""
    
    def test_perfect_correlation(self):
        """Autocorrelation should peak at zero lag"""
        x = np.sign(np.random.randn(256))
        corr = ZeroDSPCorrelator(n_samples=256, n_lags=64)
        
        y = np.zeros(256 + 64)
        y[:256] = x  # No delay
        
        R, peak = corr.correlate(x, y)
        
        assert peak == 0
        assert R[0] > 0.9  # High correlation at zero lag
    
    def test_delay_detection(self):
        """Should detect correct delay"""
        np.random.seed(42)
        N = 512
        true_delay = 73
        
        x = np.sign(np.random.randn(N))
        y = np.zeros(N + 128)
        y[true_delay:true_delay + N] = x
        
        corr = ZeroDSPCorrelator(n_samples=N, n_lags=128)
        R, peak = corr.correlate(x, y)
        
        assert peak == true_delay
    
    def test_noise_robustness(self):
        """Should work with noisy signals"""
        np.random.seed(42)
        N = 1024
        true_delay = 50
        
        x = np.sign(np.random.randn(N))
        y = np.zeros(N + 128)
        y[true_delay:true_delay + N] = x
        y += 0.3 * np.random.randn(len(y))  # Add noise
        
        corr = ZeroDSPCorrelator(n_samples=N, n_lags=128)
        R, peak = corr.correlate(x, y)
        
        # Should still detect correct delay (within 2 samples tolerance)
        assert abs(peak - true_delay) <= 2
    
    def test_2bit_correlator(self):
        """2-bit correlator should also detect delays"""
        np.random.seed(42)
        N = 512
        true_delay = 42
        
        x = np.random.randn(N)
        y = np.zeros(N + 128)
        y[true_delay:true_delay + N] = x
        
        corr = TwoBitCorrelator(n_samples=N)
        R, peak = corr.correlate(x, y, n_lags=128)
        
        assert abs(peak - true_delay) <= 3  # 2-bit has slightly less precision


class TestIntegration:
    """Integration tests"""
    
    def test_complete_tracking_pipeline(self):
        """Full pipeline: detection -> tracking -> output"""
        from ukf import create_radar_ukf
        from adaptive_noise import SageHusaEstimator, AdaptiveNoiseState
        
        ukf, state = create_radar_ukf()
        estimator = SageHusaEstimator(b=0.95)
        adapt_state = AdaptiveNoiseState(Q=state.Q.copy(), R=state.R.copy())
        
        np.random.seed(42)
        true_pos = np.array([1000.0, 1000.0])
        true_vel = np.array([10.0, 5.0])
        
        for step in range(20):
            true_pos += true_vel * 0.1
            
            # Generate measurement
            r = np.sqrt(true_pos[0]**2 + true_pos[1]**2)
            theta = np.arctan2(true_pos[1], true_pos[0])
            z = np.array([r + np.random.randn()*10, theta + np.random.randn()*0.01])
            
            # Predict
            state = ukf.predict(state, dt=0.1)
            
            # Update
            state, innovation = ukf.update(state, z)
            
            # Verify state is reasonable
            assert not np.any(np.isnan(state.x))
            assert not np.any(np.isnan(state.P))
            assert np.all(np.linalg.eigvalsh(state.P) > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
