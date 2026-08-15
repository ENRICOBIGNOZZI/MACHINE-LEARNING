# Research design

## Economic question

The paper is not another comparison of forecasting algorithms. It asks how much of a machine-learned total stock ranking is reliable enough to trade, when that reliable subset expands or contracts, where reliable information is located in the cross-section, why it varies, and whether abstention improves net economic value.

## Pre-specified hypotheses

### H1: nominal breadth exceeds reliable breadth

Every forecast vector produces a total ranking, but only a subset of pairwise relations survives the certified score-band separation rule. Report the full distribution of reliable breadth, including all pairs and economically relevant tail pairs.

### H2: weak rank relations are implementation intensive

The uncertified edge component should account for a larger share of turnover and transaction costs than of persistent gross alpha. This is a hypothesis, not a result in the current archive.

### H3: reliable breadth is state dependent

Test whether reliable breadth falls when aggregate volatility, funding stress, market illiquidity, and common-return variation rise. Use predictive regressions and event studies, but do not identify a mechanism from correlation alone.

### H4: forecast magnitude is not reliability

Within predicted-spread bins, test heterogeneity by size, liquidity, idiosyncratic volatility, analyst coverage, institutional ownership, characteristic extremeness, and model disagreement.

### H5: abstention has economic value net of costs

Trace the certification frontier against gross return, turnover, costs, capacity, net Sharpe, and certainty equivalent. Report variable-exposure and equal-gross versions separately.

## Main empirical timeline

At each annual deployment epoch:

1. generate leakage-free monthly forecasts with a predetermined rolling-window algorithm;
2. estimate the positive score scale from residuals observable before each forecast;
3. evaluate a deterministic candidate grid fixed before the empirical analysis;
4. compute monthly pairwise decision losses over the trailing certification window;
5. select the smallest threshold passing the self-normalized risk certificate;
6. freeze the selected threshold for twelve deployment months while forecasts continue to update by the predetermined algorithm;
7. repeat.

A proposal-quantile grid is used only as a robustness check. Because rolling forecasts have finite memory, its proposal block is separated from certification by more than the forecast memory; the primary deterministic grid avoids this loss of sample entirely.

The production design uses a fixed grid, a 120-month certification block, and twelve-month deployment. The full frontier varies the certification horizon and uses a separately guarded proposal grid only in robustness.

## Primary loss

For candidate threshold `q`, let `e_t(q)` be the weighted error rate among active rank relations in month `t`. Since selected-pair error need not be monotone in `q`, the certified loss is

```text
L_t(q_k) = max_{l >= k} e_t(q_l).
```

Then `L_t(q_k)` is bounded and nonincreasing, and `e_t(q_k) <= L_t(q_k)`. The primary certificate targets the stationary mean of `L_t`; wrong-bet mass with a fixed denominator is a secondary exactly monotone target.

## Models

The model grid is deliberately standard:

- ridge;
- elastic net;
- extremely randomized trees;
- histogram gradient boosting;
- multilayer perceptron;
- an equal-weight ensemble.

The finance contribution must survive across model classes. Baseline hyperparameters are fixed ex ante; nested chronological selection is reported only as a robustness exercise and never uses the final deployment block.

## Portfolio maps

1. **Raw decile:** conventional equal- or value-weighted top-minus-bottom portfolio.
2. **Certified graph:** normalize out-degree minus in-degree of the certified relation graph.
3. **Variable-notional edge portfolio:** leaves capital uninvested when relations are removed.
4. **Equal-gross certified portfolio:** isolates signal quality from reduced exposure.
5. **Certified/uncertified attribution:** fixed-denominator edge representation gives an exact pre-rescaling decomposition.

## Costs and capacity

Report:

- linear one-way costs at 10, 25, and 50 basis points;
- spread-plus-square-root-impact costs using contemporaneous CRSP daily liquidity measures;
- AUM scenarios of USD 10 million, USD 100 million, and USD 1 billion;
- turnover, participation rate, and maximum position diagnostics.

## Required headline exhibits

1. probability a relation is certified against forecast spread and rank distance;
2. certified versus uncertified shares of gross return, turnover, and costs;
3. reliable breadth through time with recession/stress overlays;
4. net Sharpe and certainty-equivalent frontier against reliable breadth;
5. anatomy by size, liquidity, idiosyncratic volatility, and forecast disagreement;
6. comparison with forecast-confidence filtering.
