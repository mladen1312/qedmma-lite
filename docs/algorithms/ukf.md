# Unscented Kalman Filter (UKF)

The Unscented Kalman Filter is a derivative-free alternative to the Extended Kalman Filter (EKF) for nonlinear state estimation.

---

## 🎯 When to Use UKF

| Scenario | Recommendation |
|----------|----------------|
| Linear system | Use standard KF (faster) |
| Mildly nonlinear | EKF is sufficient |
| Highly nonlinear | **UKF recommended** |
| No analytical Jacobian | **UKF required** |
| Range-bearing measurements | **UKF optimal** |

---

## 📐 Mathematical Foundation

### The Problem with EKF

EKF linearizes nonlinear functions using first-order Taylor expansion:

$$f(x) \approx f(\bar{x}) + F(\bar{x})(x - \bar{x})$$

Where $F = \frac{\partial f}{\partial x}$ is the Jacobian. This:

- Loses higher-order moment information
- Requires analytical derivatives
- Can diverge for strong nonlinearities

### UKF Solution: Sigma Points

Instead of linearizing, UKF propagates a set of **sigma points** through the nonlinear function.

#### Sigma Point Generation (Van der Merwe)

For state $\bar{x}$ with covariance $P$:

$$\mathcal{X}_0 = \bar{x}$$

$$\mathcal{X}_i = \bar{x} + \left(\sqrt{(n+\lambda)P}\right)_i, \quad i=1,...,n$$

$$\mathcal{X}_{i+n} = \bar{x} - \left(\sqrt{(n+\lambda)P}\right)_i, \quad i=1,...,n$$

Where:
- $n$ = state dimension
- $\lambda = \alpha^2(n+\kappa) - n$ = scaling parameter
- $\sqrt{P}$ = matrix square root (Cholesky)

#### Weights

**Mean weights:**
$$W_0^{(m)} = \frac{\lambda}{n+\lambda}$$
$$W_i^{(m)} = \frac{1}{2(n+\lambda)}, \quad i=1,...,2n$$

**Covariance weights:**
$$W_0^{(c)} = \frac{\lambda}{n+\lambda} + (1-\alpha^2+\beta)$$
$$W_i^{(c)} = \frac{1}{2(n+\lambda)}, \quad i=1,...,2n$$

---

## 🔧 Parameters

| Parameter | Symbol | Typical Value | Effect |
|-----------|--------|---------------|--------|
| Alpha | $\alpha$ | 0.001 - 1 | Spread of sigma points |
| Beta | $\beta$ | 2 | Prior knowledge (2 = Gaussian) |
| Kappa | $\kappa$ | 0 or 3-n | Secondary scaling |

!!! tip "Parameter Selection"
    - Start with $\alpha=0.1$, $\beta=2$, $\kappa=0$
    - Increase $\alpha$ if filter diverges
    - $\beta=2$ is optimal for Gaussian distributions

---

## 💻 Implementation

### Basic Usage

```python
from qedmma.advanced import UnscentedKalmanFilter, UKFParams
import numpy as np

# Define process model: constant velocity
def f(x, dt):
    """State: [px, py, vx, vy]"""
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return F @ x

# Define measurement model: range + bearing
def h(x):
    """Measurement: [range, bearing]"""
    px, py = x[0], x[1]
    r = np.sqrt(px**2 + py**2)
    theta = np.arctan2(py, px)
    return np.array([r, theta])

# Create UKF
ukf = UnscentedKalmanFilter(
    f=f,
    h=h,
    n_states=4,
    n_meas=2,
    params=UKFParams(alpha=0.1, beta=2.0, kappa=0.0)
)

# Initialize state
state = ukf.init_state(
    x0=np.array([1000.0, 1000.0, 10.0, 5.0]),  # Initial estimate
    P0=np.diag([100.0, 100.0, 10.0, 10.0]),     # Initial covariance
    Q=0.1 * np.eye(4),                          # Process noise
    R=np.diag([10.0**2, 0.01**2])               # Measurement noise
)

# Tracking loop
for z in measurements:
    state = ukf.predict(state, dt=0.1)
    state, innovation = ukf.update(state, z)
    print(f"Position: ({state.x[0]:.1f}, {state.x[1]:.1f})")
```

### Radar Tracking Preset

```python
from qedmma.advanced import create_radar_ukf

# Quick setup for radar tracking
ukf, state = create_radar_ukf(
    dt=0.1,
    process_noise=0.1,
    measurement_noise_range=10.0,
    measurement_noise_bearing=0.01
)
```

---

## 📊 Comparison with EKF

### Accuracy (Monte Carlo, 100 runs)

```
Target: Maneuvering aircraft, range-bearing radar
SNR: 10 dB

Filter | RMSE Position | RMSE Velocity | Divergence Rate
-------|---------------|---------------|----------------
EKF    | 15.2 m        | 3.1 m/s       | 12%
UKF    | 9.8 m         | 2.0 m/s       | 2%
```

### Computational Cost

```
State dimension n=4, measurement dimension m=2

Filter | Sigma Points | Complexity | Time (μs)
-------|--------------|------------|----------
EKF    | N/A          | O(n³)      | 45
UKF    | 2n+1 = 9     | O(n³)      | 120
```

---

## ⚠️ Limitations

1. **Computational cost**: 2-3x slower than EKF
2. **Memory**: Stores 2n+1 sigma points
3. **Negative weights**: Can occur for n > 3 with default parameters

!!! note "High Dimensions"
    For n > 3, consider using CKF instead, which has all positive weights.

---

## 📚 References

1. Julier, S.J., Uhlmann, J.K. "Unscented Filtering and Nonlinear Estimation" (2004)
2. Van der Merwe, R. "Sigma-Point Kalman Filters for Probabilistic Inference" (2004)
3. Wan, E.A., Van der Merwe, R. "The Unscented Kalman Filter for Nonlinear Estimation" (2000)

---

## 🔗 See Also

- [CKF](ckf.md) - Better for high dimensions
- [Adaptive Noise](adaptive-noise.md) - Time-varying noise estimation
- [API Reference](../api/advanced.md) - Full API documentation
