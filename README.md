# SN-RCPS

[![CI](https://github.com/ENRICOBIGNOZZI/MACHINE-LEARNING/actions/workflows/ci.yml/badge.svg)](https://github.com/ENRICOBIGNOZZI/MACHINE-LEARNING/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

**Self-normalized risk-controlling prediction sets under unknown temporal dependence.**

This project contains the paper, an installable Python implementation, tests, publication-quality figures, and complete replication material for:

> Enrico Bignozzi, *Risk-Controlling Prediction Sets under Unknown Temporal Dependence: Finite-Sample Limits, Tuning-Free Self-Normalization, and Event-Triggered Online Certification* (2026).

[Read the manuscript](main.pdf) · [Algorithmic contract](docs/ALGORITHM.md) · [Figure system](docs/FIGURES.md) · [Reproducibility audit](PREFLIGHT.txt)

## What the method does

Given a frozen location model $\widehat\mu$, a positive feature-dependent scale $\widehat\sigma$, and a nested threshold family,

$$
C_q(x)=\left[\widehat\mu(x)-q\widehat\sigma(x),\widehat\mu(x)+q\widehat\sigma(x)\right],
$$

SN-RCPS selects the narrowest candidate whose risk is certified below a target $\alpha$ using one stationary dependent certification trajectory. The method uses every certification observation and estimates neither a mixing coefficient nor a long-run variance.

The repository implements three deployment modes:

1. **Offline SN-RCPS:** construct a proposal library, certify it on a later block, and deploy the selected rule frozen.
2. **Scheduled episodic SN-RCPS:** refit and recertify at predetermined epochs while spending one global confidence budget across all deployments.
3. **Event-Triggered SN-RCPS:** monitor losses of the deployed rule at deterministic checkpoints and recertify only after an alarm. Every replacement is separately certified before entering service.

The event trigger changes *when* computation and recertification occur. It is not an arbitrary-stopping or per-observation anytime-valid guarantee.

## Main empirical result of the online upgrade

In a 200-replication piecewise-stationary stress test, event-triggered SN-RCPS nearly matches fixed-cadence recertification while eliminating most updates:

| Method | Coverage | Mean width | Unsafe service fraction | Mean recertifications | Mean recovery delay |
|---|---:|---:|---:|---:|---:|
| Static SN-RCPS | 0.766 | 3.297 | 0.754 | 0.00 | 3000 |
| Scheduled SN-RCPS | 0.911 | 4.831 | 0.205 | 44.00 | 609 |
| **Triggered SN-RCPS** | **0.910** | **4.819** | **0.240** | **5.67** | **952** |
| Change-point oracle | 0.900 | 4.633 | 0.000 | 3.00 | 0 |

Triggered recertification reduces the mean update count by **87.1%** relative to the fixed schedule. The experiment also reports its cost honestly: slower post-change recovery and approximately 3.5 percentage points more unsafe service time than scheduled recertification. This is a stress test under regime deterioration, not a theorem for arbitrary drift.

## Install

For the core package:

```bash
python -m pip install -e .
```

For experiments and tests:

```bash
python -m pip install -e ".[experiments,test]"
```

## Minimal offline example

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

print(result.selected_threshold)
print(lower, upper)
```

A complete executable version is in [`examples/quickstart.py`](examples/quickstart.py).

## Event-triggered deployment

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
    loss = service.observe_score(score)

# Check only at a deterministic operational checkpoint.
event = service.checkpoint(
    checkpoint=1,
    replacement=newly_certified_model,
)
```

A replacement is ignored when no alarm fires. When an alarm fires, the wrapper accepts either a separately fitted `SNRCPS` object or chronological proposal and certification scores from which to construct one. See [`examples/event_triggered.py`](examples/event_triggered.py).

## Theoretical scope

The paper proves:

- a finite-sample impossibility result showing that no uniformly informative certificate can be independent of an unrestricted mixing-speed envelope;
- an explicit blocked finite-sample benchmark when a dependence envelope is supplied;
- asymptotically valid self-normalized certification under a sequential empirical-process condition;
- post-selection stationary-risk, marginal next-step, and deployment-period guarantees for random candidate libraries;
- oracle adaptation, selection consistency, and a first-order law for the selected threshold and expected width;
- joint guarantees across scheduled deployment epochs by confidence spending;
- no additional selection penalty from data-dependent activation at deterministic checkpoints, provided every candidate replacement is separately certified.

The paper does **not** claim:

- finite-sample dependence-blind validity over unrestricted mixing speeds;
- arbitrary-stopping validity of the monitor;
- validity across arbitrary distribution shifts;
- that realized rolling coverage is the same object as stationary population risk.

## Reproduce the results

The archival ZIP includes stored repetition-level outputs and the full deployment paths. A lightweight GitHub checkout keeps the manuscript, software, aggregate outputs, and vector publication assets; the expensive raw trajectories are regenerated by the experiment scripts or recovered from the archival ZIP.

```bash
python code/run_simulations.py
python code/run_certification_efficiency.py --reuse-repetitions
python code/run_first_order.py
python code/run_electricity.py
python code/run_co2.py
python code/run_triggered_recertification.py --reuse-repetitions
python code/make_revision_tables.py
python code/make_top_paper_figures.py
```

Or run the complete pipeline:

```bash
./reproduce.sh
```

The archive includes `main.bbl`, so the paper compiles without rerunning BibTeX:

```bash
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
```

## Publication figure system

The archival release contains **25 publication figures**, each written both as an embedded-font vector PDF and as a high-resolution PNG. The GitHub source keeps the manuscript figures and a compact archive of the complete vector suite. With the full stored outputs, all figures are regenerated by one command:

```bash
python code/make_top_paper_figures.py
```

`code/plot_style.py` defines one colorblind-safe visual grammar for every method. Consequently, SN-RCPS, i.i.d. RCPS, blocking, adaptive conformal baselines, and oracle benchmarks retain the same colors, markers, line styles, target bands, and typography across simulations, electricity, carbon dioxide, and the event-triggered stress test. No numerical output is changed by the graphics layer. See [`docs/FIGURES.md`](docs/FIGURES.md).

## Repository layout

```text
src/snrcps/          installable implementation
examples/            minimal offline and event-triggered examples
tests/               unit tests
code/                replication and unified figure-generation scripts
figures/             manuscript figures; full vector suite in the archive
tables/              aggregate outputs; raw trajectories in the archival ZIP
docs/                algorithmic documentation
main.tex, main.pdf   manuscript source and compiled paper
```

## Data

The raw PJM East CSV is not redistributed. [`code/download_data.py`](code/download_data.py) downloads the public mirror and verifies the expected SHA256 checksum. The Mauna Loa data are obtained through the documented public source in the replication script.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Until a journal or proceedings version is available, cite the 2026 preprint and software release.

## License

Code is released under the [BSD 3-Clause License](LICENSE). The manuscript and empirical outputs are provided for research reproducibility.
