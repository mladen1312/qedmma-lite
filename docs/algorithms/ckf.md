# Cubature Kalman Filter (CKF)

The Cubature Kalman Filter uses spherical-radial cubature rules to approximate Gaussian-weighted integrals, providing a parameter-free alternative to UKF.

---

## 🎯 When to Use CKF

| Scenario | Recommendation |
|----------|----------------|
| State dimension n ≤ 3 | UKF or CKF (similar performance) |
| State dimension n > 3 | **CKF recommended** |
| No tuning desired | **CKF (no parameters)** |
| Maximum numerical stability | **Square-Root CKF** |

---

## 📐 Mathematical Foundation

### Motivation: UKF Weight Problem

UKF weights can become negative for higher dimensions:

$$W_0^{(m)} = \frac{\lambda}{n+\lambda} = \frac{\alpha^2(n+\kappa) - n}{n + \alpha^2(n+\kappa) - n}$$

For typical $\alpha=0.001$, $\kappa=0$:
- n=3: $W_0 \approx -999$ (negative!)
- This causes numerical instability

### CKF Solution: Cubature Rule

CKF uses the **third-degree spherical-radial cubature rule**:

$$\int_{\mathbb{R}^n} f(x) \mathcal{N}(x|\bar{x},P) dx \approx \sum_{i=1}^{2n} W_i f(\mathcal{X}_i)$$

#### Cubature Points

Generate 2n points using unit vectors:

$$\xi_i = \sqrt{n} \cdot e_i, \quad i=1,...,n$$
$$\xi_{i+n} = -\sqrt{n} \cdot e_i, \quad i=1,...,n$$

Transform to state space:

$$\mathcal{X}_i = S \xi_i + \bar{x}$$

Where $S$ is the Cholesky factor: $P = SS^T$

#### Weights

All weights are **equal and positive**:

$$W_i = \frac{1}{2n}, \quad \forall i \in \{1, ..., 2n\}$$

!!! success "Key Advantage"
    No negative weights, regardless of dimension!

---

## 🔧 Implementation

### Basic CKF

```python
from qedmma.advanced import CubatureKalmanFilter
import numpy as np

# 9D state: position, velocity, acceleration (3D each)
n_states = 9

def f(x, dt):
    """Constant acceleration model"""
    F = np.eye(9)
    # Position += velocity * dt + 0.5 * acc * dt²
    for i in range(3):
        F[i, i+3] = dt
        F[i, i+6] = 0.5 * dt**2
        F[i+3, i+6] = dt
    return F @ x

def h(x):
    """Position-only measurement"""
    return x[:3]

# Create CKF (no tuning parameters!)
ckf = CubatureKalmanFilter(
    f=f,
    h=h,
    n_states=9,
    n_meas=3
)

# Initialize
state = ckf.init_state(
    x0=np.zeros(9),
    P0=np.diag([100]*3 + [10]*3 + [1]*3),
    Q=0.01 * np.eye(9),
    R=np.diag([5.0, 5.0, 5.0])
)

# Track
for z in measurements:
    state = ckf.predict(state, dt=0.1)
    state, innovation = ckf.update(state, z)
```

### Square-Root CKF

For maximum numerical stability:

```python
from qedmma.advanced import SquareRootCKF

sr_ckf = SquareRootCKF(
    f=f,
    h=h,
    n_states=9,
    n_meas=3
)

# SR-CKF propagates Cholesky factor directly
# Guarantees positive semi-definiteness
```

---

## 📊 CKF vs UKF Comparison

### Numerical Stability (n=9)

```
Test: 1000 Monte Carlo runs, varying SNR

SNR (dB) | UKF Divergence | CKF Divergence
---------|----------------|----------------
20       | 0%             | 0%
10       | 5%             | 0%
5        | 18%            | 2%
0        | 42%            | 8%
```

### Computational Complexity

| Aspect | UKF | CKF |
|--------|-----|-----|
| Sigma/Cubature points | 2n+1 | 2n |
| Weight computation | Complex | Trivial (1/2n) |
| Tuning parameters | 3 (α, β, κ) | 0 |
| Stability for n>3 | Problematic | Excellent |

---

## 📐 Algorithm Steps

### Predict

```
1. Generate cubature points from (x, P):
   X_i = chol(P) @ ξ_i + x

2. Propagate through process model:
   X*_i = f(X_i, dt)

3. Compute predicted mean:
   x⁻ = (1/2n) Σ X*_i

4. Compute predicted covariance:
   P⁻ = (1/2n) Σ (X*_i - x⁻)(X*_i - x⁻)ᵀ + Q
```

### Update

```
1. Generate cubature points from (x⁻, P⁻):
   X_i = chol(P⁻) @ ξ_i + x⁻

2. Transform through measurement model:
   Z_i = h(X_i)

3. Predicted measurement:
   z̄ = (1/2n) Σ Z_i

4. Innovation covariance:
   Pzz = (1/2n) Σ (Z_i - z̄)(Z_i - z̄)ᵀ + R

5. Cross covariance:
   Pxz = (1/2n) Σ (X_i - x⁻)(Z_i - z̄)ᵀ

6. Kalman gain:
   K = Pxz @ Pzz⁻¹

7. Update:
   x = x⁻ + K(z - z̄)
   P = P⁻ - K @ Pzz @ Kᵀ
```

---

## 🔧 High-Dimensional Example

```python
from qedmma.advanced import create_high_dim_ckf

# Create CKF for 9D tracking (position, velocity, acceleration)
ckf = create_high_dim_ckf(n_states=9, n_meas=3)

# CKF handles this gracefully while UKF might struggle
state = ckf.init_state(
    x0=np.zeros(9),
    P0=np.eye(9),
    Q=0.01 * np.eye(9),
    R=np.eye(3)
)
```

---

## 📚 References

1. Arasaratnam, I., Haykin, S. "Cubature Kalman Filters" IEEE TAC (2009)
2. Arasaratnam, I. "Cubature Kalman Filtering: Theory & Applications" (2009)
3. Jia, B., Xin, M., Cheng, Y. "High-Degree Cubature Kalman Filter" (2013)

---

## 🔗 See Also

- [UKF](ukf.md) - Alternative for lower dimensions
- [Adaptive Noise](adaptive-noise.md) - Combine with CKF for robust tracking
- [Benchmarks](../benchmarks/filter-comparison.md) - Performance comparison
