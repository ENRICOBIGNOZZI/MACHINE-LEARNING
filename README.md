# SN-RCPS

[![CI](https://github.com/ENRICOBIGNOZZI/MACHINE-LEARNING/actions/workflows/ci.yml/badge.svg)](https://github.com/ENRICOBIGNOZZI/MACHINE-LEARNING/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

**Self-normalized risk-controlling prediction sets under unknown temporal dependence.**

Software companion to:

> Enrico Bignozzi, *Risk-Controlling Prediction Sets under Unknown Temporal Dependence: Finite-Sample Limits, Tuning-Free Self-Normalization, and Event-Triggered Online Certification* (2026).

The complete archival release distributed with the paper contains the 40-page manuscript, all experiment scripts, repetition-level outputs, 25 vector figures, 25 high-resolution PNGs, and a SHA256 manifest. This GitHub checkout keeps the installable and continuously tested public API, executable examples, documentation, critical values, and the unified graphics layer.

[Algorithmic contract](docs/ALGORITHM.md) · [Figure system](docs/FIGURES.md) · [Reproducibility audit](PREFLIGHT.txt)

## Method

For a frozen location model $\widehat\mu$, a positive feature-dependent scale $\widehat\sigma$, and a nested threshold family,

$$
C_q(x)=\left[\widehat\mu(x)-q\widehat\sigma(x),\widehat\mu(x)+q\widehat\sigma(x)\right],
$$

SN-RCPS selects the narrowest candidate whose risk is certified below a target $\alpha$ using one stationary dependent certification trajectory. It estimates neither a mixing coefficient nor a long-run variance.

The implementation supports:

1. **Offline SN-RCPS** for one frozen deployment;
2. **scheduled episodic certification** with confidence spending;
3. **event-triggered recertification** at deterministic checkpoints.

Every replacement rule must pass a separate chronological certification step. The monitor does not create arbitrary-stopping or per-observation anytime validity.

## Main online result

In a 200-replication piecewise-stationary stress test:

| Method | Coverage | Mean width | Unsafe service fraction | Mean recertifications | Mean recovery delay |
|---|---:|---:|---:|---:|---:|
| Static SN-RCPS | 0.766 | 3.297 | 0.754 | 0.00 | 3000 |
| Scheduled SN-RCPS | 0.911 | 4.831 | 0.205 | 44.00 | 609 |
| **Triggered SN-RCPS** | **0.910** | **4.819** | **0.240** | **5.67** | **952** |
| Change-point oracle | 0.900 | 4.633 | 0.000 | 3.00 | 0 |

Triggered recertification uses **87.1% fewer updates** than fixed-cadence recertification while retaining nearly identical mean coverage and width. Its slower recovery and larger unsafe-service fraction are reported explicitly; this experiment is not presented as a theorem for arbitrary drift.

## Install and test

```bash
python -m pip install -e ".[test]"
pytest
python examples/quickstart.py
python examples/event_triggered.py
```

## Minimal use

```python
import numpy as np
from snrcps import SNRCPS, critical_value

rng = np.random.default_rng(20260814)
proposal_scores = np.abs(rng.normal(size=2_000))
certification_scores = np.abs(rng.normal(size=2_000))

certifier = SNRCPS(
    risk_target=0.10,
    critical_value=critical_value(delta=0.10),
    fallback="largest",
)
result = certifier.fit(
    proposal_scores,
    certification_scores,
    levels=[0.90, 0.92, 0.94, 0.96, 0.98, 0.995],
    safe_threshold=4.5,
)

location = np.array([0.0, 0.5, 1.0])
scale = np.array([0.5, 0.8, 1.2])
lower, upper = certifier.predict_interval(location, scale)
```

## Event-triggered service

```python
from snrcps import EventTriggeredSNRCPS, SelfNormalizedRiskMonitor

monitor = SelfNormalizedRiskMonitor(
    risk_target=0.10,
    critical_value=2.0,
    window=256,
    minimum_observations=128,
    safety_margin=0.01,
    patience=2,
    mode="evidence",
)
service = EventTriggeredSNRCPS(current=certifier, monitor=monitor)

for score in deployment_scores:
    service.observe_score(score)

# Evaluated only at a deterministic checkpoint.
event = service.checkpoint(
    checkpoint=1,
    replacement=newly_certified_model,
)
```

A trigger alone never certifies a replacement. See [`examples/event_triggered.py`](examples/event_triggered.py) and [`docs/ALGORITHM.md`](docs/ALGORITHM.md).

## Scope

The paper establishes dependence-robust asymptotic certification, post-selection guarantees, oracle adaptation, a first-order efficiency law, scheduled confidence spending, and no extra selection penalty from event-triggered activation at deterministic checkpoints.

It does **not** claim finite-sample dependence-blind validity over unrestricted mixing speeds, arbitrary-stopping validity, or validity under arbitrary distribution shift.

## Graphics

`code/plot_style.py` defines the colorblind-safe publication grammar used consistently across simulations, electricity, carbon dioxide, and event-triggered deployment. The archival release contains the complete 25-figure vector and high-resolution suite. See [`docs/FIGURES.md`](docs/FIGURES.md).

## Citation and license

Citation metadata are in [`CITATION.cff`](CITATION.cff). Code is released under the [BSD 3-Clause License](LICENSE).
