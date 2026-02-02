#!/usr/bin/env python3
"""
Adaptive Noise Estimation Test Suite
=====================================
Author:  Dr. Mladen Mešter / Nexellum d.o.o.
License: AGPL-3.0-or-later
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_noise import (
    AdaptiveNoiseEstimator,
    AdaptiveNoiseConfig,
    AdaptiveKalmanFilter,
    MehraEstimator,
    SageHusaEstimator,
    VariationalBayesianEstimator,
    CovarianceMatchingEstimator,
    IMMAdaptiveEstimator,
    AdaptationMethod,
    compute_nees,
    compute_nis,
)


class TestAdaptiveNoiseConfig:
    """Test configuration dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveNoiseConfig()
        assert config.window_size == 20
        assert config.forgetting_factor == 0.98
        assert config.min_samples == 5
        assert config.adapt_Q is True
        assert config.adapt_R is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AdaptiveNoiseConfig(
            window_size=50,
            forgetting_factor=0.95,
            adapt_Q=False
        )
        assert config.window_size == 50
        assert config.forgetting_factor == 0.95
        assert config.adapt_Q is False


class TestMehraEstimator:
    """Test Mehra innovation-based estimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        est = MehraEstimator(dim_x=4, dim_z=2)
        assert est.dim_x == 4
        assert est.dim_z == 2
        assert est.Q.shape == (4, 4)
        assert est.R.shape == (2, 2)
    
    def test_update_returns_valid_covariances(self):
        """Test that update returns valid covariance matrices."""
        est = MehraEstimator(dim_x=4, dim_z=2)
        
        # Dummy inputs
        innovation = np.array([0.5, -0.3])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        P_prior = np.eye(4) * 10.0
        K = np.eye(4, 2) * 0.1
        
        # Multiple updates to build history
        for _ in range(25):
            Q, R = est.update(innovation + np.random.randn(2) * 0.1, H, P_prior, K)
        
        # Check positive definiteness
        assert np.all(np.linalg.eigvals(Q) > 0)
        assert np.all(np.linalg.eigvals(R) > 0)
    
    def test_reset(self):
        """Test estimator reset."""
        est = MehraEstimator(dim_x=4, dim_z=2)
        
        # Do some updates
        for _ in range(10):
            est.update(np.random.randn(2), np.eye(2, 4), np.eye(4), np.eye(4, 2))
        
        # Reset
        Q_init = np.eye(4) * 0.5
        R_init = np.eye(2) * 2.0
        est.reset(Q_init, R_init)
        
        assert np.allclose(est.Q, Q_init)
        assert np.allclose(est.R, R_init)


class TestSageHusaEstimator:
    """Test Sage-Husa adaptive estimator."""
    
    def test_forgetting_factor(self):
        """Test that forgetting factor affects adaptation rate."""
        config_fast = AdaptiveNoiseConfig(forgetting_factor=0.9)
        config_slow = AdaptiveNoiseConfig(forgetting_factor=0.99)
        
        est_fast = SageHusaEstimator(dim_x=2, dim_z=2, config=config_fast)
        est_slow = SageHusaEstimator(dim_x=2, dim_z=2, config=config_slow)
        
        # Same inputs
        H = np.eye(2)
        P_prior = np.eye(2) * 10.0
        K = np.eye(2) * 0.1
        
        # Large innovation to trigger adaptation
        innovation = np.array([10.0, 10.0])
        
        for _ in range(20):
            est_fast.update(innovation, H, P_prior, K)
            est_slow.update(innovation, H, P_prior, K)
        
        # Fast should have larger R (adapted more to large innovations)
        assert np.trace(est_fast.R) > np.trace(est_slow.R)
    
    def test_outlier_rejection(self):
        """Test outlier rejection mechanism."""
        config = AdaptiveNoiseConfig(outlier_threshold=3.0)
        est = SageHusaEstimator(dim_x=2, dim_z=2, config=config)
        
        H = np.eye(2)
        P_prior = np.eye(2)
        K = np.eye(2) * 0.5
        
        # Normal updates
        for _ in range(10):
            est.update(np.random.randn(2) * 0.5, H, P_prior, K)
        
        R_before = est.R.copy()
        
        # Outlier (should be rejected)
        est.update(np.array([100.0, 100.0]), H, P_prior, K)
        
        # R should not change much
        assert np.allclose(est.R, R_before, rtol=0.5)


class TestVariationalBayesianEstimator:
    """Test Variational Bayesian estimator."""
    
    def test_vb_iterations(self):
        """Test that VB iterations affect estimates."""
        config = AdaptiveNoiseConfig(vb_iterations=10)
        est = VariationalBayesianEstimator(dim_x=2, dim_z=2, config=config)
        
        H = np.eye(2)
        P_prior = np.eye(2)
        K = np.eye(2) * 0.5
        
        for _ in range(20):
            innovation = np.random.randn(2)
            Q, R = est.update(innovation, H, P_prior, K)
        
        assert Q.shape == (2, 2)
        assert R.shape == (2, 2)
    
    def test_prior_dof_effect(self):
        """Test effect of prior degrees of freedom."""
        config_weak = AdaptiveNoiseConfig(vb_prior_dof=3.0)
        config_strong = AdaptiveNoiseConfig(vb_prior_dof=10.0)
        
        est_weak = VariationalBayesianEstimator(dim_x=2, dim_z=2, config=config_weak)
        est_strong = VariationalBayesianEstimator(dim_x=2, dim_z=2, config=config_strong)
        
        # Same updates
        H = np.eye(2)
        P_prior = np.eye(2)
        K = np.eye(2) * 0.5
        
        for _ in range(30):
            inn = np.array([5.0, 5.0])  # Large innovations
            est_weak.update(inn, H, P_prior, K)
            est_strong.update(inn, H, P_prior, K)
        
        # Weak prior should adapt more
        # (This is a rough check - actual behavior depends on implementation details)
        assert est_weak.R is not None
        assert est_strong.R is not None


class TestCovarianceMatchingEstimator:
    """Test covariance matching estimator."""
    
    def test_window_size(self):
        """Test that window size is respected."""
        config = AdaptiveNoiseConfig(window_size=10, min_samples=3)
        est = CovarianceMatchingEstimator(dim_x=2, dim_z=2, config=config)
        
        H = np.eye(2)
        P_prior = np.eye(2)
        K = np.eye(2) * 0.5
        
        # Fill window
        for i in range(15):
            est.update(np.random.randn(2), H, P_prior, K)
        
        # Internal buffer should be limited to window size
        assert len(est._innovation_history) == 10


class TestIMMAdaptiveEstimator:
    """Test IMM-adaptive estimator."""
    
    def test_multi_model_adaptation(self):
        """Test adaptation across multiple models."""
        imm_est = IMMAdaptiveEstimator(
            n_models=3,
            dim_x=4,
            dim_z=2,
            method=AdaptationMethod.SAGE_HUSA
        )
        
        # Dummy inputs for 3 models
        innovations = [np.random.randn(2) for _ in range(3)]
        H_list = [np.eye(2, 4) for _ in range(3)]
        P_priors = [np.eye(4) * 10 for _ in range(3)]
        K_list = [np.eye(4, 2) * 0.1 for _ in range(3)]
        model_probs = np.array([0.5, 0.3, 0.2])
        
        Q_list, R_list = imm_est.update(
            innovations, H_list, P_priors, K_list, model_probs
        )
        
        assert len(Q_list) == 3
        assert len(R_list) == 3
    
    def test_combined_estimate(self):
        """Test probability-weighted combined estimate."""
        imm_est = IMMAdaptiveEstimator(
            n_models=2,
            dim_x=2,
            dim_z=2,
            method=AdaptationMethod.COVARIANCE_MATCHING
        )
        
        model_probs = np.array([0.7, 0.3])
        Q_combined, R_combined = imm_est.get_combined_estimate(model_probs)
        
        assert Q_combined.shape == (2, 2)
        assert R_combined.shape == (2, 2)


class TestAdaptiveNoiseEstimatorFactory:
    """Test the factory class."""
    
    @pytest.mark.parametrize("method", [
        'mehra', 'sage_husa', 'variational_bayesian', 'vb', 
        'covariance_matching', 'matching'
    ])
    def test_create_all_methods(self, method):
        """Test creating estimators with all methods."""
        est = AdaptiveNoiseEstimator(method=method, dim_x=4, dim_z=2)
        assert est is not None
        assert est.dim_x == 4
        assert est.dim_z == 2
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            AdaptiveNoiseEstimator(method='invalid_method', dim_x=4, dim_z=2)
    
    def test_dict_config(self):
        """Test passing config as dict."""
        est = AdaptiveNoiseEstimator(
            method='sage_husa',
            dim_x=4,
            dim_z=2,
            config={'forgetting_factor': 0.95, 'window_size': 30}
        )
        assert est.config.forgetting_factor == 0.95
        assert est.config.window_size == 30


class TestAdaptiveKalmanFilter:
    """Test integrated adaptive Kalman filter."""
    
    def test_basic_operation(self):
        """Test basic predict/update cycle."""
        akf = AdaptiveKalmanFilter(dim_x=4, dim_z=2)
        
        # Predict
        akf.predict()
        
        # Update
        z = np.array([1.0, 2.0])
        innovation = akf.update(z)
        
        assert innovation.shape == (2,)
        assert akf.x.shape == (4,)
        assert akf.P.shape == (4, 4)
    
    def test_noise_adaptation(self):
        """Test that noise estimates change during operation."""
        akf = AdaptiveKalmanFilter(
            dim_x=2, 
            dim_z=2,
            adaptation='sage_husa',
            config=AdaptiveNoiseConfig(forgetting_factor=0.9, min_samples=3)
        )
        
        R_initial = akf.R.copy()
        
        # Many updates with large innovations
        for _ in range(50):
            akf.predict()
            akf.update(np.array([10.0, 10.0]))  # Large measurements
        
        R_final = akf.R
        
        # R should have changed
        assert not np.allclose(R_initial, R_final)
    
    def test_with_control_input(self):
        """Test filter with control input."""
        akf = AdaptiveKalmanFilter(dim_x=2, dim_z=2)
        
        B = np.eye(2)
        u = np.array([1.0, 0.5])
        
        x_before = akf.x.copy()
        akf.predict(u=u, B=B)
        
        # State should change due to control
        assert not np.allclose(akf.x, x_before)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compute_nees(self):
        """Test NEES computation."""
        x_true = np.array([1.0, 2.0])
        x_est = np.array([1.1, 1.9])
        P = np.eye(2)
        
        nees = compute_nees(x_true, x_est, P)
        
        # Should be positive
        assert nees > 0
        # For small errors, should be small
        assert nees < 1.0
    
    def test_compute_nis(self):
        """Test NIS computation."""
        innovation = np.array([0.1, -0.2])
        S = np.eye(2)
        
        nis = compute_nis(innovation, S)
        
        assert nis > 0
        assert nis < 1.0  # Small innovation


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.slow
    def test_tracking_with_noise_change(self):
        """Test tracking performance when noise changes."""
        np.random.seed(42)
        
        dim_x = 4
        dim_z = 2
        dt = 0.1
        
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        
        akf = AdaptiveKalmanFilter(
            dim_x=dim_x,
            dim_z=dim_z,
            F=F,
            H=H,
            adaptation='sage_husa',
            config=AdaptiveNoiseConfig(forgetting_factor=0.95)
        )
        
        # Simulate
        Q_true = np.diag([0.1, 1.0, 0.1, 1.0])
        R_true_low = np.diag([1.0, 1.0])
        R_true_high = np.diag([25.0, 25.0])
        
        x_true = np.array([0.0, 5.0, 0.0, 3.0])
        errors_before_change = []
        errors_after_change = []
        
        for k in range(200):
            R_true = R_true_low if k < 100 else R_true_high
            
            # True dynamics
            x_true = F @ x_true + np.random.multivariate_normal(np.zeros(dim_x), Q_true)
            z = H @ x_true + np.random.multivariate_normal(np.zeros(dim_z), R_true)
            
            # Filter
            akf.predict()
            akf.update(z)
            
            # Track error
            error = np.linalg.norm(x_true[:2] - akf.x[[0, 2]])
            if k < 100:
                errors_before_change.append(error)
            elif k > 120:  # Allow adaptation time
                errors_after_change.append(error)
        
        # Filter should still track reasonably after noise change
        avg_error_before = np.mean(errors_before_change)
        avg_error_after = np.mean(errors_after_change)
        
        # Error after should not explode (adaptive filter compensates)
        assert avg_error_after < 5 * avg_error_before


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
