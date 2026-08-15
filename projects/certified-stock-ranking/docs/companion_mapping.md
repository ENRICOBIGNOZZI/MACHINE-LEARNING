# Boundary with the SN-RCPS companion paper

This repository is designed to produce a finance paper that is scientifically distinct from **Risk-Controlling Prediction Sets under Unknown Temporal Dependence**.

## Reused inferential infrastructure

The following ingredients are taken from, and should be cited to, the companion statistical paper:

- the impossibility of informative dependence-blind finite-sample certification without a dependence envelope;
- the self-normalized upper confidence bound for a bounded monotone loss trajectory;
- the ordered RCPS selector and its library-size-free post-selection argument;
- scheduled confidence spending and the modular event-triggered replacement principle.

The finance paper states only the specialization needed for stock-ranking decisions. Full proofs of the generic self-normalized calibration theorem remain in the companion paper. The finance appendix proves only finance-specific facts and verifies that its loss family satisfies the companion theorem's conditions.

## New finance objects and results

The following material belongs exclusively to the finance paper:

- a machine-learned stock ranking as a total order that mechanically suppresses abstention;
- score-band separation as a nested partial order over stocks;
- reliable breadth as the fraction of economically relevant rank relations retained;
- the selective pairwise error and its monotone envelope;
- all-pair and top-versus-bottom relation universes;
- relation-graph and fixed-denominator edge portfolios;
- the exact raw = certified + removed relation decomposition;
- the robust no-trade interpretation with ambiguity and transaction costs;
- the empirical anatomy of nominal versus reliably tradable cross-sectional information;
- turnover, cost, capacity, and implementable-frontier tests.

## Text and exhibit separation

Do not copy the companion paper's introduction, simulations, electricity application, carbon-dioxide application, or generic proof appendix into this manuscript. The finance paper should cite the companion result once, present the specialized corollary, and spend its pages on economic measurement and portfolio implications.

The statistical paper may contain at most a short sentence noting that bounded monotone deployment losses include selective ranking losses. It should not contain the CRSP design, reliable-breadth evidence, relation portfolios, transaction costs, or finance figures. Conversely, the finance paper should not market self-normalization itself as the primary novelty.

## Claim language

The preferred finance claim is:

> Modern prediction systems create more nominal cross-sectional differentiation than an investor can reliably deploy. Certified partial rankings measure the gap and reveal its consequences for positions, turnover, costs, and net performance.

The disallowed claim is:

> We apply our new SN-RCPS algorithm to stocks.

That wording would make the finance paper look like a recycled application rather than an independent economic contribution.
