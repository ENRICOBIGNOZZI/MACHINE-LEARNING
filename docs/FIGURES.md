# Publication figure system

The project uses one visual grammar for every empirical result. The design layer is deliberately separate from the statistical computations: figures consume stored CSV outputs and never modify numerical results.

## Regeneration

```bash
python -m pip install -e ".[experiments]"
python code/make_top_paper_figures.py
```

With the archival outputs present, the command writes 25 vector PDFs and 25 high-resolution PNGs to `figures/`. The manuscript includes the vector versions.

## Visual contract

- SN-RCPS and its feature-dependent/episodic variants use green diamonds or squares.
- i.i.d. RCPS uses blue circles.
- exact blocking and change-point oracles use orange.
- rolling conformal uses gold, ACI uses magenta, and generic split conformal uses neutral gray.
- infeasible oracle library members use black stars or dashed black lines.
- valid risk or coverage regions are shown by the same pale-green target band throughout.
- panel labels, titles, line weights, font sizes, and legend spacing are shared across all applications.

The palette is colorblind-safe and remains distinguishable in grayscale through marker and line-style redundancy. PDF fonts are embedded; PNGs are exported at 360 dpi.

## Files

`code/plot_style.py` contains the style constants and output helpers. `code/make_top_paper_figures.py` reconstructs:

- stationary-risk and width diagnostics under autoregressive dependence;
- the finite-range blocking comparison, certification-efficiency frontier, and oracle-gap experiment;
- the first-order quantile diagnostic;
- electricity certification, rolling coverage, conditional coverage, local scale reliability, and feature-dependent interval panels;
- carbon-dioxide episodic coverage, width, service, scale, and selection-dependence panels;
- the event-triggered threshold, exact-risk, rolling-coverage, and update-economy dashboard.

The older plotting statements inside experiment scripts are retained for stand-alone execution, but the final publication artifacts are overwritten by `make_top_paper_figures.py` at the end of `reproduce.sh`.
