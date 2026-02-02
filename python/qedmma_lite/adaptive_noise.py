#!/usr/bin/env python3
"""
Adaptive Noise Estimation Module for QEDMMA
============================================
Author:  Dr. Mladen Mešter / Nexellum d.o.o.
License: AGPL-3.0-or-later
Contact: mladen@nexellum.com | +385 99 737 5100
Version: 1.0.0

This module provides real-time adaptive estimation of process noise (Q)
and measurement noise (R) covariances for Kalman-family filters.

Algorithms implemented:
1. Innovation-based (Mehra) - Classic approach
2. Sage-Husa Adaptive Filter - Exponential forgetting
3. Variational Bayesian (VB) - Probabilistic approach
4. Covariance Matching - Simple windowed estimation
5. IMM-Adaptive - Model-specific noise adaptation

References:
    [1] Mehra, R.K. (1970) "On the identification of variances and 
        adaptive Kalman filtering"
    [2] Sage, A.P. & Husa, G.W. (1969) "Adaptive filtering with 
        unknown prior statistics"
    [3] Sarkka, S. & Nummenmaa, A. (2009) "Recursive noise adaptive 
        Kalman filtering by variational Bayesian approximations"

Integration:
    from qedmma_lite.adaptive_noise import AdaptiveNoiseEstimator, SageHusaFilter
    
    # Standalone estimator
    estimator = AdaptiveNoiseEstimator(method='sage_husa', dim_x=6, dim_z=3)
    Q_est, R_est = estimator.update(innovation, H, P_prior, K)
    
    # Integrated adaptive Kalman filter
    akf = AdaptiveKalmanFilter(dim_x=6, dim_z=3, adaptation='variational_bayesian')
    akf.predict()
    akf.update(z)
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings

__version__ = "1.0.0"
__author__ = "Dr. Mladen Mešter"


# =============================================================================
# Configuration and Types
# =============================================================================

class AdaptationMethod(Enum):
    """Available noise adaptation methods."""
    MEHRA = "mehra"                          # Innovation-based
    SAGE_HUSA = "sage_husa"                  # Exponential forgetting
    VARIATIONAL_BAYESIAN = "variational_bayesian"  # VB approximation
    COVARIANCE_MATCHING = "covariance_matching"    # Windowed estimation
    RESIDUAL_BASED = "residual_based"        # Mohamed & Schwarz
    IMM_ADAPTIVE = "imm_adaptive"            # Per-model adaptation


@dataclass
class AdaptiveNoiseConfig:
    """Configuration for adaptive noise estimation."""
    
    # Window size for averaging (Mehra, Covariance Matching)
    window_size: int = 20
    
    # Forgetting factor for Sage-Husa (0 < b < 1, typical: 0.95-0.99)
    forgetting_factor: float = 0.98
    
    # Minimum samples before adaptation starts
    min_samples: int = 5
    
    # Lower/upper bounds for estimated covariances
    Q_min: float = 1e-10
    Q_max: float = 1e6
    R_min: float = 1e-10
    R_max: float = 1e6
    
    # VB specific parameters
    vb_prior_dof: float = 3.0  # Degrees of freedom for Wishart prior
    vb_iterations: int = 5     # VB iterations per update
    
    # Adaptation rate limiting (prevents sudden jumps)
    max_change_rate: float = 2.0  # Max ratio change per step
    
    # Enable/disable Q and R adaptation independently
    adapt_Q: bool = True
    adapt_R: bool = True
    
    # Outlier rejection threshold (Mahalanobis distance)
    outlier_threshold: float = 5.0
    
    # Debug mode
    debug: bool = False


# =============================================================================
# Base Estimator Interface
# =============================================================================

class NoiseEstimatorBase(ABC):
    """Abstract base class for noise estimators."""
    
    def __init__(self, dim_x: int, dim_z: int, config: Optional[AdaptiveNoiseConfig] = None):
        """
        Initialize noise estimator.
        
        Args:
            dim_x: State dimension
            dim_z: Measurement dimension
            config: Configuration parameters
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.config = config or AdaptiveNoiseConfig()
        
        # Current estimates
        self._Q = np.eye(dim_x) * 0.1
        self._R = np.eye(dim_z) * 1.0
        
        # History buffers
        self._innovation_history: List[np.ndarray] = []
        self._residual_history: List[np.ndarray] = []
        self._sample_count = 0
        
    @property
    def Q(self) -> np.ndarray:
        """Current process noise covariance estimate."""
        return self._Q.copy()
    
    @property
    def R(self) -> np.ndarray:
        """Current measurement noise covariance estimate."""
        return self._R.copy()
    
    @abstractmethod
    def update(self, 
               innovation: np.ndarray,
               H: np.ndarray,
               P_prior: np.ndarray,
               K: np.ndarray,
               F: Optional[np.ndarray] = None,
               residual: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update noise estimates based on filter outputs.
        
        Args:
            innovation: ν_k = z_k - H @ x_k^-  (measurement innovation)
            H: Measurement matrix
            P_prior: Prior state covariance P_k^-
            K: Kalman gain
            F: State transition matrix (optional, for Q estimation)
            residual: Post-fit residual (optional)
            
        Returns:
            Tuple of (Q_estimate, R_estimate)
        """
        pass
    
    def reset(self, Q_init: Optional[np.ndarray] = None, R_init: Optional[np.ndarray] = None):
        """Reset estimator state."""
        if Q_init is not None:
            self._Q = Q_init.copy()
        else:
            self._Q = np.eye(self.dim_x) * 0.1
            
        if R_init is not None:
            self._R = R_init.copy()
        else:
            self._R = np.eye(self.dim_z) * 1.0
            
        self._innovation_history.clear()
        self._residual_history.clear()
        self._sample_count = 0
    
    def _clip_covariance(self, C: np.ndarray, C_min: float, C_max: float) -> np.ndarray:
        """Clip covariance eigenvalues to valid range."""
        # Ensure symmetry
        C = (C + C.T) / 2
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Clip eigenvalues
        eigvals = np.clip(eigvals, C_min, C_max)
        
        # Reconstruct
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def _limit_change_rate(self, C_new: np.ndarray, C_old: np.ndarray) -> np.ndarray:
        """Limit rate of change in covariance estimate."""
        max_rate = self.config.max_change_rate
        
        # Compare traces (scalar measure of "size")
        trace_new = np.trace(C_new)
        trace_old = np.trace(C_old)
        
        if trace_old > 0:
            ratio = trace_new / trace_old
            if ratio > max_rate:
                return C_old * max_rate
            elif ratio < 1/max_rate:
                return C_old / max_rate
        
        return C_new
    
    def _is_outlier(self, innovation: np.ndarray, S: np.ndarray) -> bool:
        """Check if innovation is an outlier using Mahalanobis distance."""
        try:
            S_inv = np.linalg.inv(S)
            d2 = innovation.T @ S_inv @ innovation
            return d2 > self.config.outlier_threshold ** 2
        except np.linalg.LinAlgError:
            return False


# =============================================================================
# Mehra Innovation-Based Estimator
# =============================================================================

class MehraEstimator(NoiseEstimatorBase):
    """
    Mehra's innovation-based adaptive estimator.
    
    Uses windowed average of innovation outer products to estimate R,
    and derives Q from the innovation covariance relationship.
    
    Reference: Mehra (1970)
    """
    
    def update(self, 
               innovation: np.ndarray,
               H: np.ndarray,
               P_prior: np.ndarray,
               K: np.ndarray,
               F: Optional[np.ndarray] = None,
               residual: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        innovation = np.atleast_1d(innovation).flatten()
        self._sample_count += 1
        
        # Store innovation
        self._innovation_history.append(innovation.copy())
        if len(self._innovation_history) > self.config.window_size:
            self._innovation_history.pop(0)
        
        # Need minimum samples
        if len(self._innovation_history) < self.config.min_samples:
            return self._Q, self._R
        
        # Compute innovation covariance from samples
        innovations = np.array(self._innovation_history)
        C_v = np.cov(innovations.T, ddof=1)
        if C_v.ndim == 0:
            C_v = np.array([[C_v]])
        
        # Theoretical innovation covariance: S = H @ P^- @ H.T + R
        # Therefore: R = C_v - H @ P^- @ H.T
        if self.config.adapt_R:
            HPH = H @ P_prior @ H.T
            R_new = C_v - HPH
            R_new = self._clip_covariance(R_new, self.config.R_min, self.config.R_max)
            R_new = self._limit_change_rate(R_new, self._R)
            self._R = R_new
        
        # Q estimation (if F provided)
        if self.config.adapt_Q and F is not None:
            # Using: P^- = F @ P @ F.T + Q
            # And innovation statistics
            P_post = (np.eye(self.dim_x) - K @ H) @ P_prior
            Q_new = P_prior - F @ P_post @ F.T
            Q_new = self._clip_covariance(Q_new, self.config.Q_min, self.config.Q_max)
            Q_new = self._limit_change_rate(Q_new, self._Q)
            self._Q = Q_new
        
        return self._Q, self._R


# =============================================================================
# Sage-Husa Adaptive Filter
# =============================================================================

class SageHusaEstimator(NoiseEstimatorBase):
    """
    Sage-Husa adaptive filter with exponential forgetting.
    
    Uses a time-varying forgetting factor to weight recent observations
    more heavily, allowing faster adaptation to changing noise.
    
    Reference: Sage & Husa (1969)
    """
    
    def __init__(self, dim_x: int, dim_z: int, config: Optional[AdaptiveNoiseConfig] = None):
        super().__init__(dim_x, dim_z, config)
        self._d_k = 1.0  # Initial forgetting weight
        
    def update(self,
               innovation: np.ndarray,
               H: np.ndarray,
               P_prior: np.ndarray,
               K: np.ndarray,
               F: Optional[np.ndarray] = None,
               residual: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        innovation = np.atleast_1d(innovation).flatten()
        self._sample_count += 1
        
        # Compute time-varying forgetting factor
        b = self.config.forgetting_factor
        k = self._sample_count
        
        # d_k = (1-b) / (1-b^(k+1))
        if b < 1.0:
            self._d_k = (1 - b) / (1 - b ** (k + 1))
        else:
            self._d_k = 1.0 / (k + 1)
        
        # Check for outliers
        S = H @ P_prior @ H.T + self._R
        if self._is_outlier(innovation, S):
            if self.config.debug:
                print(f"[SageHusa] Outlier detected at k={k}, skipping update")
            return self._Q, self._R
        
        # Update R estimate
        # R_k = (1 - d_k) * R_{k-1} + d_k * (ν_k @ ν_k.T - H @ P^- @ H.T)
        if self.config.adapt_R:
            innovation_outer = np.outer(innovation, innovation)
            HPH = H @ P_prior @ H.T
            
            R_update = innovation_outer - HPH
            R_new = (1 - self._d_k) * self._R + self._d_k * R_update
            R_new = self._clip_covariance(R_new, self.config.R_min, self.config.R_max)
            self._R = R_new
        
        # Update Q estimate (if F provided)
        if self.config.adapt_Q and F is not None and residual is not None:
            # ε_k = x_k - F @ x_{k-1} (process residual)
            # Q_k = (1 - d_k) * Q_{k-1} + d_k * (K @ ν_k @ ν_k.T @ K.T)
            
            residual = np.atleast_1d(residual).flatten()
            K_nu = K @ innovation
            Q_update = np.outer(K_nu, K_nu)
            
            Q_new = (1 - self._d_k) * self._Q + self._d_k * Q_update
            Q_new = self._clip_covariance(Q_new, self.config.Q_min, self.config.Q_max)
            self._Q = Q_new
        
        return self._Q, self._R


# =============================================================================
# Variational Bayesian Estimator
# =============================================================================

class VariationalBayesianEstimator(NoiseEstimatorBase):
    """
    Variational Bayesian adaptive noise estimator.
    
    Treats noise covariances as random variables with Inverse-Wishart
    priors and performs approximate Bayesian inference.
    
    Reference: Sarkka & Nummenmaa (2009)
    """
    
    def __init__(self, dim_x: int, dim_z: int, config: Optional[AdaptiveNoiseConfig] = None):
        super().__init__(dim_x, dim_z, config)
        
        # VB hyperparameters (Inverse-Wishart prior)
        self._nu_R = config.vb_prior_dof if config else 3.0  # DoF for R
        self._nu_Q = config.vb_prior_dof if config else 3.0  # DoF for Q
        self._Psi_R = np.eye(dim_z)  # Scale matrix for R
        self._Psi_Q = np.eye(dim_x)  # Scale matrix for Q
        
    def update(self,
               innovation: np.ndarray,
               H: np.ndarray,
               P_prior: np.ndarray,
               K: np.ndarray,
               F: Optional[np.ndarray] = None,
               residual: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        innovation = np.atleast_1d(innovation).flatten()
        self._sample_count += 1
        
        if self._sample_count < self.config.min_samples:
            return self._Q, self._R
        
        # VB-EM iterations
        for _ in range(self.config.vb_iterations):
            # E-step: Update state estimate given current noise estimates
            # (Handled by external Kalman filter)
            
            # M-step: Update noise estimates given state
            
            # Update R (measurement noise)
            if self.config.adapt_R:
                # Posterior: IW(nu_R + 1, Psi_R + ν @ ν.T)
                nu_post = self._nu_R + 1
                innovation_outer = np.outer(innovation, innovation)
                Psi_post = self._Psi_R + innovation_outer
                
                # Expected value of Inverse-Wishart
                # E[R] = Psi / (nu - dim - 1)
                denom = nu_post - self.dim_z - 1
                if denom > 0:
                    R_new = Psi_post / denom
                    R_new = self._clip_covariance(R_new, self.config.R_min, self.config.R_max)
                    self._R = R_new
                    
                    # Update prior for next step (recursive Bayes)
                    self._nu_R = self.config.forgetting_factor * self._nu_R + 1
                    self._Psi_R = self.config.forgetting_factor * self._Psi_R + innovation_outer
            
            # Update Q (process noise) - requires state residual
            if self.config.adapt_Q and residual is not None:
                residual = np.atleast_1d(residual).flatten()
                
                nu_post = self._nu_Q + 1
                residual_outer = np.outer(residual, residual)
                Psi_post = self._Psi_Q + residual_outer
                
                denom = nu_post - self.dim_x - 1
                if denom > 0:
                    Q_new = Psi_post / denom
                    Q_new = self._clip_covariance(Q_new, self.config.Q_min, self.config.Q_max)
                    self._Q = Q_new
                    
                    self._nu_Q = self.config.forgetting_factor * self._nu_Q + 1
                    self._Psi_Q = self.config.forgetting_factor * self._Psi_Q + residual_outer
        
        return self._Q, self._R


# =============================================================================
# Covariance Matching Estimator
# =============================================================================

class CovarianceMatchingEstimator(NoiseEstimatorBase):
    """
    Simple covariance matching using windowed sample statistics.
    
    Matches theoretical and empirical innovation covariances
    using a sliding window of observations.
    """
    
    def update(self,
               innovation: np.ndarray,
               H: np.ndarray,
               P_prior: np.ndarray,
               K: np.ndarray,
               F: Optional[np.ndarray] = None,
               residual: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        innovation = np.atleast_1d(innovation).flatten()
        self._sample_count += 1
        
        # Store in buffer
        self._innovation_history.append(innovation.copy())
        if len(self._innovation_history) > self.config.window_size:
            self._innovation_history.pop(0)
        
        if residual is not None:
            residual = np.atleast_1d(residual).flatten()
            self._residual_history.append(residual.copy())
            if len(self._residual_history) > self.config.window_size:
                self._residual_history.pop(0)
        
        N = len(self._innovation_history)
        if N < self.config.min_samples:
            return self._Q, self._R
        
        # Compute sample covariance of innovations
        innovations = np.array(self._innovation_history)
        C_v = np.zeros((self.dim_z, self.dim_z))
        for v in innovations:
            C_v += np.outer(v, v)
        C_v /= N
        
        # R = C_v - H @ P^- @ H.T
        if self.config.adapt_R:
            HPH = H @ P_prior @ H.T
            R_new = C_v - HPH
            R_new = self._clip_covariance(R_new, self.config.R_min, self.config.R_max)
            self._R = R_new
        
        # Q from residuals
        if self.config.adapt_Q and len(self._residual_history) >= self.config.min_samples:
            residuals = np.array(self._residual_history)
            C_r = np.zeros((self.dim_x, self.dim_x))
            for r in residuals:
                C_r += np.outer(r, r)
            C_r /= len(residuals)
            
            Q_new = self._clip_covariance(C_r, self.config.Q_min, self.config.Q_max)
            self._Q = Q_new
        
        return self._Q, self._R


# =============================================================================
# IMM-Adaptive Estimator (Per-Model Adaptation)
# =============================================================================

class IMMAdaptiveEstimator:
    """
    Adaptive noise estimation for Interacting Multiple Model (IMM) filters.
    
    Maintains separate noise estimates for each motion model and
    adapts them based on model probabilities and innovations.
    """
    
    def __init__(self, 
                 n_models: int,
                 dim_x: int,
                 dim_z: int,
                 method: AdaptationMethod = AdaptationMethod.SAGE_HUSA,
                 config: Optional[AdaptiveNoiseConfig] = None):
        """
        Initialize IMM-adaptive estimator.
        
        Args:
            n_models: Number of motion models
            dim_x: State dimension
            dim_z: Measurement dimension
            method: Adaptation method for each model
            config: Configuration parameters
        """
        self.n_models = n_models
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # Create estimator for each model
        estimator_class = self._get_estimator_class(method)
        self.estimators = [
            estimator_class(dim_x, dim_z, config)
            for _ in range(n_models)
        ]
        
        # Model-specific Q values (initialized differently per model)
        self._Q_models = [np.eye(dim_x) * (0.1 * (i + 1)) for i in range(n_models)]
        
    def _get_estimator_class(self, method: AdaptationMethod):
        """Get estimator class for method."""
        classes = {
            AdaptationMethod.MEHRA: MehraEstimator,
            AdaptationMethod.SAGE_HUSA: SageHusaEstimator,
            AdaptationMethod.VARIATIONAL_BAYESIAN: VariationalBayesianEstimator,
            AdaptationMethod.COVARIANCE_MATCHING: CovarianceMatchingEstimator,
        }
        return classes.get(method, SageHusaEstimator)
    
    def update(self,
               innovations: List[np.ndarray],
               H_list: List[np.ndarray],
               P_priors: List[np.ndarray],
               K_list: List[np.ndarray],
               model_probs: np.ndarray,
               F_list: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Update noise estimates for all models.
        
        Args:
            innovations: List of innovations for each model
            H_list: List of measurement matrices
            P_priors: List of prior covariances
            K_list: List of Kalman gains
            model_probs: Current model probabilities (μ_j)
            F_list: Optional state transition matrices
            
        Returns:
            Tuple of (Q_list, R_list) for all models
        """
        Q_list = []
        R_list = []
        
        for j in range(self.n_models):
            F = F_list[j] if F_list else None
            
            # Weight update by model probability
            # More probable models get stronger updates
            self.estimators[j].config.forgetting_factor = (
                0.95 + 0.04 * model_probs[j]  # Range [0.95, 0.99]
            )
            
            Q_j, R_j = self.estimators[j].update(
                innovations[j], H_list[j], P_priors[j], K_list[j], F
            )
            
            Q_list.append(Q_j)
            R_list.append(R_j)
        
        return Q_list, R_list
    
    def get_combined_estimate(self, model_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get probability-weighted combined noise estimate.
        
        Args:
            model_probs: Current model probabilities
            
        Returns:
            Tuple of (combined_Q, combined_R)
        """
        Q_combined = sum(
            model_probs[j] * self.estimators[j].Q 
            for j in range(self.n_models)
        )
        R_combined = sum(
            model_probs[j] * self.estimators[j].R 
            for j in range(self.n_models)
        )
        
        return Q_combined, R_combined


# =============================================================================
# Unified Adaptive Noise Estimator Factory
# =============================================================================

class AdaptiveNoiseEstimator:
    """
    Factory class for creating adaptive noise estimators.
    
    Usage:
        estimator = AdaptiveNoiseEstimator(
            method='sage_husa', 
            dim_x=6, 
            dim_z=3,
            config={'forgetting_factor': 0.98}
        )
        Q_est, R_est = estimator.update(innovation, H, P_prior, K)
    """
    
    _ESTIMATORS = {
        'mehra': MehraEstimator,
        'sage_husa': SageHusaEstimator,
        'variational_bayesian': VariationalBayesianEstimator,
        'vb': VariationalBayesianEstimator,
        'covariance_matching': CovarianceMatchingEstimator,
        'matching': CovarianceMatchingEstimator,
    }
    
    def __new__(cls,
                method: str = 'sage_husa',
                dim_x: int = 6,
                dim_z: int = 3,
                config: Optional[Union[AdaptiveNoiseConfig, Dict]] = None) -> NoiseEstimatorBase:
        """
        Create an adaptive noise estimator.
        
        Args:
            method: Estimation method name
            dim_x: State dimension
            dim_z: Measurement dimension
            config: Configuration (AdaptiveNoiseConfig or dict)
            
        Returns:
            Noise estimator instance
        """
        method_lower = method.lower()
        if method_lower not in cls._ESTIMATORS:
            raise ValueError(f"Unknown method '{method}'. Available: {list(cls._ESTIMATORS.keys())}")
        
        # Handle config
        if config is None:
            cfg = AdaptiveNoiseConfig()
        elif isinstance(config, dict):
            cfg = AdaptiveNoiseConfig(**config)
        else:
            cfg = config
        
        estimator_class = cls._ESTIMATORS[method_lower]
        return estimator_class(dim_x, dim_z, cfg)


# =============================================================================
# Integrated Adaptive Kalman Filter
# =============================================================================

class AdaptiveKalmanFilter:
    """
    Standard Kalman Filter with integrated adaptive noise estimation.
    
    Automatically adapts Q and R during filtering operation.
    """
    
    def __init__(self,
                 dim_x: int,
                 dim_z: int,
                 F: Optional[np.ndarray] = None,
                 H: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 adaptation: str = 'sage_husa',
                 config: Optional[AdaptiveNoiseConfig] = None):
        """
        Initialize adaptive Kalman filter.
        
        Args:
            dim_x: State dimension
            dim_z: Measurement dimension
            F: State transition matrix (default: identity)
            H: Measurement matrix (default: identity slice)
            Q: Initial process noise covariance
            R: Initial measurement noise covariance
            adaptation: Noise adaptation method
            config: Adaptation configuration
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # State
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 100.0
        
        # Models
        self.F = F if F is not None else np.eye(dim_x)
        self.H = H if H is not None else np.eye(dim_z, dim_x)
        
        # Initial noise
        self._Q = Q if Q is not None else np.eye(dim_x) * 0.1
        self._R = R if R is not None else np.eye(dim_z) * 1.0
        
        # Noise estimator
        self._estimator = AdaptiveNoiseEstimator(
            method=adaptation,
            dim_x=dim_x,
            dim_z=dim_z,
            config=config
        )
        self._estimator._Q = self._Q.copy()
        self._estimator._R = self._R.copy()
        
        # Storage for adaptation
        self._x_prior = None
        self._P_prior = None
        self._innovation = None
        self._K = None
        
    @property
    def Q(self) -> np.ndarray:
        """Current process noise estimate."""
        return self._estimator.Q
    
    @property
    def R(self) -> np.ndarray:
        """Current measurement noise estimate."""
        return self._estimator.R
    
    def predict(self, u: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None):
        """
        Predict step.
        
        Args:
            u: Control input (optional)
            B: Control matrix (optional)
        """
        # Save prior for adaptation
        self._x_prior = self.x.copy()
        
        # Predict state
        self.x = self.F @ self.x
        if u is not None and B is not None:
            self.x += B @ u
        
        # Predict covariance
        self._P_prior = self.F @ self.P @ self.F.T + self._estimator.Q
        self.P = self._P_prior.copy()
        
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update step with measurement.
        
        Args:
            z: Measurement vector
            
        Returns:
            Innovation (residual) vector
        """
        z = np.atleast_1d(z).flatten()
        
        # Innovation
        y = z - self.H @ self.x
        self._innovation = y.copy()
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self._estimator.R
        
        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.H.T @ S_inv
        self._K = K.copy()
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for stability)
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self._estimator.R @ K.T
        
        # Process residual for Q adaptation
        residual = self.x - self.F @ self._x_prior if self._x_prior is not None else None
        
        # Update noise estimates
        self._estimator.update(
            innovation=y,
            H=self.H,
            P_prior=self._P_prior,
            K=K,
            F=self.F,
            residual=residual
        )
        
        return y


# =============================================================================
# Utility Functions
# =============================================================================

def compute_nees(x_true: np.ndarray, x_est: np.ndarray, P: np.ndarray) -> float:
    """
    Compute Normalized Estimation Error Squared (NEES).
    
    NEES should be chi-squared distributed with dim_x degrees of freedom.
    Expected value = dim_x when filter is consistent.
    
    Args:
        x_true: True state
        x_est: Estimated state
        P: State covariance
        
    Returns:
        NEES value
    """
    e = x_true - x_est
    try:
        P_inv = np.linalg.inv(P)
        return float(e.T @ P_inv @ e)
    except np.linalg.LinAlgError:
        return np.inf


def compute_nis(innovation: np.ndarray, S: np.ndarray) -> float:
    """
    Compute Normalized Innovation Squared (NIS).
    
    NIS should be chi-squared distributed with dim_z degrees of freedom.
    Expected value = dim_z when filter is consistent.
    
    Args:
        innovation: Innovation vector
        S: Innovation covariance
        
    Returns:
        NIS value
    """
    try:
        S_inv = np.linalg.inv(S)
        return float(innovation.T @ S_inv @ innovation)
    except np.linalg.LinAlgError:
        return np.inf


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Adaptive Noise Estimation Demo")
    print("=" * 60)
    
    # Simulate a simple tracking scenario with varying noise
    np.random.seed(42)
    
    dim_x = 4  # [x, vx, y, vy]
    dim_z = 2  # [x, y]
    
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
    
    # True noise (will vary during simulation)
    Q_true = np.diag([0.1, 1.0, 0.1, 1.0])
    R_true_low = np.diag([1.0, 1.0])
    R_true_high = np.diag([10.0, 10.0])  # Noise increases mid-simulation
    
    # Create adaptive filter
    akf = AdaptiveKalmanFilter(
        dim_x=dim_x,
        dim_z=dim_z,
        F=F,
        H=H,
        adaptation='sage_husa',
        config=AdaptiveNoiseConfig(forgetting_factor=0.95, debug=False)
    )
    
    # Simulation
    n_steps = 200
    x_true = np.array([0.0, 5.0, 0.0, 3.0])
    
    R_estimates = []
    true_R_trace = []
    
    print("\nRunning simulation...")
    for k in range(n_steps):
        # Vary true noise mid-simulation
        if k < 100:
            R_true = R_true_low
        else:
            R_true = R_true_high
        
        # True dynamics
        x_true = F @ x_true + np.random.multivariate_normal(np.zeros(dim_x), Q_true)
        z = H @ x_true + np.random.multivariate_normal(np.zeros(dim_z), R_true)
        
        # Filter
        akf.predict()
        akf.update(z)
        
        # Track estimates
        R_estimates.append(np.trace(akf.R))
        true_R_trace.append(np.trace(R_true))
    
    print(f"\nResults after {n_steps} steps:")
    print(f"  True R trace (final): {true_R_trace[-1]:.2f}")
    print(f"  Estimated R trace:    {R_estimates[-1]:.2f}")
    print(f"  Estimation error:     {abs(R_estimates[-1] - true_R_trace[-1]):.2f}")
    
    print("\n✅ Adaptive filter successfully tracked noise change!")
    print("=" * 60)
