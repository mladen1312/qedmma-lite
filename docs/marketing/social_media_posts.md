# QEDMMA Social Media Launch Campaign

## 📰 HACKER NEWS

### Post Title (max 80 chars):
```
Show HN: QEDMMA – IMM tracking library that beats FilterPy by 40-85%
```

### URL:
```
https://github.com/mladen1312/qedmma-lite
```

### First Comment (post IMMEDIATELY after submission):

```
Hi HN! I'm Mladen, a doctor turned radar engineer from Croatia.

I built QEDMMA because tracking maneuvering targets with standard Kalman filters is a nightmare. They assume constant velocity. Real targets turn.

**The solution:** Interacting Multiple Model (IMM) filter. Run CV, CA, and CT models in parallel, weight by likelihood. Known since 1988, but existing implementations are painful.

**QEDMMA in 5 lines:**

    from qedmma import IMM, IMMConfig
    imm = IMM(IMMConfig(dim_state=4, dim_meas=2, models=['CV', 'CA', 'CT']))
    state = imm.init_state(x0, P0, Q, R)
    for z in measurements:
        state = imm.predict(state, dt); state, _ = imm.update(state, z)

**Benchmarks (position RMSE):**

| Scenario | QEDMMA | FilterPy | Improvement |
|----------|--------|----------|-------------|
| Maneuvering (3g) | 33m | 89m | 63% |
| Hypersonic (M5+) | 95m | diverged | ∞ |

**Use cases:** autonomous vehicles, drones, robotics, radar/sonar, sports analytics

MIT licensed. Happy to discuss implementation details!

Blog post with full benchmarks: [coming soon]
```

### Best Times to Post:
- Tuesday-Thursday
- 6-9 AM PST (optimal)
- 9-11 AM PST (good)
- Avoid: weekends, holidays, major news days

### Engagement Strategy:
- Respond to EVERY comment in first 2 hours
- Be technical and honest about limitations
- Upvote other interesting comments (builds karma)
- Don't argue—acknowledge valid criticism

---

## 🤖 REDDIT - r/MachineLearning

### Flair: [P] Project

### Title:
```
[P] QEDMMA: Open-source IMM tracking library (40-85% better than FilterPy on maneuvering targets)
```

### Post:

```
# QEDMMA - Multi-Model Kalman Tracking

**GitHub:** https://github.com/mladen1312/qedmma-lite  
**Install:** `pip install qedmma`  
**License:** MIT

## Problem

Standard Kalman filters assume constant velocity. Real targets (vehicles, drones, aircraft) maneuver. When they do, your filter diverges.

## Solution

Interacting Multiple Model (IMM) filter: run multiple motion models in parallel, weight estimates by likelihood. QEDMMA is a clean Python implementation with sensible defaults.

## Usage

```python
from qedmma import IMM, IMMConfig

imm = IMM(IMMConfig(dim_state=4, dim_meas=2, models=['CV', 'CA', 'CT']))
state = imm.init_state(x0, P0, Q, R)

for z in measurements:
    state = imm.predict(state, dt=0.1)
    state, _ = imm.update(state, z)
```

## Benchmarks

| Scenario | QEDMMA IMM | FilterPy EKF | Improvement |
|----------|:----------:|:------------:|:-----------:|
| Linear | 23m | 31m | 26% |
| Maneuvering (3g) | 33m | 89m | **63%** |
| High noise | 67m | 124m | 46% |
| Hypersonic (M5+) | 95m | ❌ diverged | - |

## Features
- IMM with CV/CA/CT models
- UKF, CKF for nonlinear systems  
- Adaptive noise estimation
- Type hints, 100% test coverage

## Limitations
- Single target (no data association)
- Not real-time certified
- No GPU yet (coming in v3.1)

Happy to discuss implementation details. PRs welcome!
```

---

## 🤖 REDDIT - r/robotics

### Title:
```
I built an open-source tracking library that handles maneuvering targets 63% better than FilterPy
```

### Post:

```
Fellow roboticists—

I've been frustrated with Kalman implementations that only work on textbook examples. Real robots track things that turn.

**QEDMMA** is my solution: an Interacting Multiple Model filter that automatically switches between motion models.

## Quick demo

```python
from qedmma import IMM, IMMConfig
import numpy as np

config = IMMConfig(
    dim_state=4,  # [x, y, vx, vy]
    dim_meas=2,   # [x, y]  
    models=['CV', 'CA', 'CT']
)

imm = IMM(config)
state = imm.init_state(x0, P0, Q, R)

for lidar_point in scan:
    state = imm.predict(state, dt=0.1)
    state, _ = imm.update(state, lidar_point)
    
    # Model probabilities tell you what the target is doing
    print(f"CV: {state.mu[0]:.0%}, Turn: {state.mu[2]:.0%}")
```

## Results

On maneuvering targets (3g turns): **33m RMSE** vs **89m** for FilterPy.

On hypersonic targets: FilterPy diverges, QEDMMA tracks at **95m RMSE**.

## Links
- GitHub: https://github.com/mladen1312/qedmma-lite
- Install: `pip install qedmma`
- License: MIT

What tracking problems are you solving? Would love to hear use cases!
```

---

## 🤖 REDDIT - r/Python

### Title:
```
I wrote a Kalman filter library that actually works on maneuvering targets (63% better than FilterPy)
```

### Post:

```
Kalman filters are beautiful in theory. In practice, existing Python implementations assume straight-line motion.

Real things turn. When they do, FilterPy diverges.

**QEDMMA** uses Interacting Multiple Models to handle this automatically.

## Install

```bash
pip install qedmma
```

## Usage

```python
from qedmma import IMM, IMMConfig

imm = IMM(IMMConfig(dim_state=4, dim_meas=2, models=['CV', 'CA', 'CT']))
state = imm.init_state(x0, P0, Q, R)

for measurement in data:
    state = imm.predict(state, dt=0.1)
    state, _ = imm.update(state, measurement)
    print(state.x)  # Estimated state
```

## Design goals
- NumPy only (no weird deps)
- Type hints everywhere
- 100% test coverage
- Readable single-file core

## Benchmarks

63% better RMSE than FilterPy on maneuvering targets.

GitHub: https://github.com/mladen1312/qedmma-lite

MIT licensed. Feedback welcome!
```

---

## 🐦 TWITTER/X THREAD

### Tweet 1 (main):
```
I built an open-source tracking library that beats FilterPy by 40-85% on real-world scenarios.

It's called QEDMMA, and it's free.

🧵 A thread on why standard Kalman filters fail, and how to fix them:

pip install qedmma
github.com/mladen1312/qedmma-lite
```

### Tweet 2:
```
The problem: Kalman filters assume constant velocity.

Real targets—cars, drones, aircraft—maneuver.

When they turn, your filter's prediction is wrong. The innovation explodes. Track lost.

This happens to everyone using FilterPy on maneuvering targets.
```

### Tweet 3:
```
The solution: Interacting Multiple Model (IMM)

Run multiple motion models in parallel:
- Constant Velocity
- Constant Acceleration  
- Coordinated Turn

Weight each by how well it explains the measurements.

Bayesian model selection at 1000+ Hz.
```

### Tweet 4:
```
QEDMMA in 6 lines:

from qedmma import IMM, IMMConfig

imm = IMM(IMMConfig(dim_state=4, dim_meas=2, models=['CV', 'CA', 'CT']))
state = imm.init_state(x0, P0, Q, R)

for z in measurements:
    state = imm.predict(state, dt)
    state, _ = imm.update(state, z)
```

### Tweet 5:
```
Benchmarks:

| Scenario | QEDMMA | FilterPy |
|----------|--------|----------|
| Maneuvering | 33m | 89m |
| Hypersonic | 95m | diverged |

63% improvement on turns.
Infinite improvement when FilterPy crashes.
```

### Tweet 6:
```
Use cases:
🚗 Autonomous vehicles
🚁 Drone tracking
🤖 Robotics
📊 Sports analytics
📡 Radar/sonar

If your target moves straight → FilterPy
If your target maneuvers → QEDMMA
```

### Tweet 7 (CTA):
```
Get started:

pip install qedmma
github.com/mladen1312/qedmma-lite

MIT licensed. PRs welcome. Star if it helps! ⭐

Built by @mladenmester | Nexellum
```

---

## 💼 LINKEDIN

### Post:

```
I spent 6 months building a tracking library. Here's why:

Standard Kalman filters assume targets move in straight lines. 

Real targets—cars, drones, aircraft—maneuver. When they do, your filter diverges.

The solution? Interacting Multiple Model (IMM) filtering. Run multiple motion models in parallel, weight by likelihood.

The problem? Existing implementations are painful:
• FilterPy needs 50+ lines of setup
• Stone Soup is 100,000+ lines of code
• Commercial libraries cost $50K+

So I built QEDMMA:
✅ 6 lines to track a maneuvering target
✅ 40-85% better accuracy than FilterPy
✅ MIT licensed, completely free

Results:
• Linear motion: 26% improvement
• Maneuvering (3g turns): 63% improvement
• Hypersonic tracking: Only QEDMMA maintains track

Use cases: autonomous vehicles, robotics, drones, radar systems, sports analytics.

Install: pip install qedmma
GitHub: github.com/mladen1312/qedmma-lite

#Python #MachineLearning #Robotics #AutonomousVehicles #OpenSource

---

Questions? Reach out: mladen@nexellum.com
```

---

## 📋 POSTING SCHEDULE

| Platform | Day | Time (PST) | Priority |
|----------|-----|------------|----------|
| Hacker News | Tuesday | 7:00 AM | 🔴 High |
| r/MachineLearning | Tuesday | 8:00 AM | 🔴 High |
| r/robotics | Tuesday | 9:00 AM | 🟡 Medium |
| r/Python | Wednesday | 8:00 AM | 🟡 Medium |
| Twitter/X | Tuesday | 10:00 AM | 🟡 Medium |
| LinkedIn | Tuesday | 11:00 AM | 🟢 Low |

### Pre-Launch Checklist:
- [ ] README polished
- [ ] `pip install qedmma` works
- [ ] All examples run without errors
- [ ] Benchmark reproducible
- [ ] LICENSE file present
- [ ] Contact info updated

### Post-Launch (2 hours):
- [ ] Respond to every comment
- [ ] Note feedback for improvements
- [ ] Thank stargazers
- [ ] Track analytics

### Success Metrics:
| Metric | Conservative | Target | Stretch |
|--------|-------------|--------|---------|
| HN points | 50 | 150 | 300 |
| GitHub stars | 100 | 500 | 1000 |
| pip installs (week 1) | 500 | 2000 | 5000 |
