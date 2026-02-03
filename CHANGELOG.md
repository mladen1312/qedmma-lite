# Changelog

## [3.1.0] - 2026-02-03

### Added
- **True IMM Smoother** achieving +48% RMSE improvement
- Per-model RTS smoothing with probability-weighted combination
- `IMMSmoother` class in `qedmma/smoother.py`
- `compute_entropy_q_scale()` for adaptive Q tuning
- Comprehensive test suite
- Military validation benchmarks (missile guidance, fighter tracking)

### Changed
- Tracker stores predictions from forward pass for correct RTS
- Default `p_stay` changed to 0.88 for better maneuver response
- Improved numerical stability with regularization

### Fixed
- **Critical:** RTS smoother now uses `F @ x_mixed` from forward pass, not recomputed `F @ x_filt`
- Mode probability normalization edge cases

### Performance
| Scenario | Filter | Smoother | Improvement |
|----------|--------|----------|-------------|
| 3g Turn | 1.81m | 0.88m | +51.1% |
| 6g Turn | 1.91m | 0.98m | +48.5% |
| 9g Turn | 1.95m | 1.02m | +47.7% |

## [3.0.0] - 2026-01-15

### Added
- Initial public release
- CV/CT+ /CT- IMM filter bank
- Markov chain mode transitions
- Basic forward filtering

---
*Nexellum d.o.o. | mladen@nexellum.com*
