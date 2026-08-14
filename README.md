# SN-RCPS

[![CI](https://github.com/ENRICOBIGNOZZI/MACHINE-LEARNING/actions/workflows/ci.yml/badge.svg)](https://github.com/ENRICOBIGNOZZI/MACHINE-LEARNING/actions/workflows/ci.yml)

**Self-normalized risk-controlling prediction sets under unknown temporal dependence.**

This repository contains the installable Python implementation and manuscript source for:

> Enrico Bignozzi, *Risk-Controlling Prediction Sets under Unknown Temporal Dependence: Finite-Sample Limits, Tuning-Free Self-Normalization, and Event-Triggered Online Certification* (2026).

The method certifies feature-dependent prediction sets from one dependent trajectory without estimating a mixing coefficient or a long-run variance. The online extension monitors a frozen certified rule at deterministic checkpoints and recertifies only after an alarm; every replacement must pass a separate chronological certification step.

In a 200-replication piecewise-stationary stress test, triggered SN-RCPS attains mean coverage 0.910 and mean width 4.819 with 5.67 updates, versus 0.911, 4.831, and 44 updates for scheduled recertification. This is an 87.1% reduction in updates. The paper reports the corresponding delay cost and does not claim arbitrary-stopping or arbitrary-drift validity.

## Install

```bash
python -m pip install -e ".[experiments,test]"
pytest
```

## Minimal use

```python
from snrcps import SNRCPS, critical_value

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
lower, upper = certifier.predict_interval(location, scale)
```

See `docs/ALGORITHM.md`, `examples/`, `tests/`, and `main.tex` for the full contract, event-triggered API, and theory.

## Scope

The release distinguishes:

1. offline certification of a frozen rule;
2. scheduled episodic confidence spending;
3. event-triggered activation at deterministic checkpoints;
4. arbitrary-stopping validity, which is not established here.

Code is released under the BSD 3-Clause License.