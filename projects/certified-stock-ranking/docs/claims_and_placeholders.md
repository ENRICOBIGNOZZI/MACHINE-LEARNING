# Claims discipline

This repository currently supports methodological and software claims only.

## Supported now

- score-band separation induces a nested strict partial order;
- reliable breadth is monotone in the abstention threshold;
- the monotone envelope of selective pairwise error is a valid bounded monotone loss;
- the self-normalized selector is implemented exactly as specified in the companion methodology;
- the exact all-pair and top-versus-bottom loss algorithms agree with brute force in unit tests;
- the fixed-denominator edge map yields an exact certified/uncertified return decomposition;
- turnover is computed after passive return drift over the union of changing stock universes, including exits and entries;
- the forecast pipeline reads an explicit feature manifest and cannot silently use target or cost columns;
- the production fixed-grid and synthetic proposal-grid workflows run end to end.

## Not supported until licensed data are attached

- any numerical estimate of reliable breadth in U.S. equities;
- any claim about alpha, Sharpe ratio, turnover, costs, capacity, crises, or firm-level heterogeneity;
- any ranking of forecast models;
- any claim that certified portfolios outperform raw portfolios.

The manuscript marks all empirical result language as pending. Synthetic figures are explicitly labelled and must not be moved into the empirical results section.
