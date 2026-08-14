# Algorithmic contract

This document specifies what the public implementation does and what its guarantees require.

## 1. Frozen feature-dependent prediction sets

Fit a location predictor $\widehat\mu$ and a strictly positive scale predictor $\widehat\sigma$ using data that precede proposal and certification. Define normalized scores

$$
S_t=\frac{|Y_t-\widehat\mu(X_t)|}{\widehat\sigma(X_t)}
$$

and nested prediction intervals

$$
C_q(x)=\left[\widehat\mu(x)-q\widehat\sigma(x),\widehat\mu(x)+q\widehat\sigma(x)\right].
$$

The software treats $\widehat\mu$ and $\widehat\sigma$ as frozen during proposal, certification, and one deployment episode.

## 2. Proposal and certification

A proposal block generates ordered thresholds $Q_1\leq\cdots\leq Q_K$. On a later certification block, candidate $k$ has loss

$$
L_{t,k}=\mathbf 1\{S_t>Q_k\}.
$$

For each candidate, `SNRCPS` computes the empirical risk, recursive self-normalizer, one-sided upper bound, and the RCPS suffix envelope. It returns the narrowest candidate whose complete wider suffix is certified below the target $\alpha$.

The fallback policy is explicit:

- `largest`: deploy the widest supplied candidate if none is certified;
- `infinite`: return an infinite threshold;
- `raise`: stop with an error.

The software never silently converts a failed certificate into a successful one.

## 3. Scheduled episodic deployment

For epochs $j=1,\ldots,J$, repeat the chronological fit-proposal-certification pipeline and deploy the selected rule frozen within epoch $j$. Choose confidence budgets satisfying

$$
\sum_{j=1}^{J}\delta_j\leq\delta.
$$

`equal_confidence_spending` implements $\delta_j=\delta/J$. `polynomial_confidence_spending` implements

$$
\delta_j=\frac{6\delta}{\pi^2j^2},
$$

whose infinite sum is $\delta$.

The paper's joint guarantee is a union bound over epoch-specific certificate failures; no independence between epochs is required.

## 4. Event-triggered activation

`SelfNormalizedRiskMonitor` observes losses of the currently deployed, frozen rule. At deterministic operational checkpoints it computes

$$
\widehat R\pm d\sqrt{\widehat V/m},
$$

where $\widehat V$ is the recursive self-normalizer on the monitoring window.

In the default `evidence` mode, the monitor alarms when the lower one-sided bound exceeds $\alpha+\kappa$, where $\kappa\geq0$ is a safety margin. `patience` requires repeated alarms; `cooldown_checkpoints` suppresses immediate repeated updates.

The alternative `support` mode is proactive: it alarms when recent data no longer support risk below a safety-adjusted target. It is an operational rule, not evidence that the deployed risk exceeds $\alpha$.

`EventTriggeredSNRCPS` enforces the state transition

```text
observe deployed loss
        ↓
deterministic checkpoint
        ↓
monitor alarm? ── no ──> keep current certified rule
        │
       yes
        ↓
obtain a chronologically separate replacement certificate
        ↓
deploy replacement and reset monitoring history
```

A trigger alone never certifies a replacement.

## 5. Why the trigger adds no selection penalty

At checkpoint $j$, let $A_j\in\{0,1\}$ be any activation decision measurable before the replacement certification data are used. Let $\widehat C_j$ be the potential replacement that would be certified at that checkpoint. The deployed recursion is

$$
\widetilde C_j=
\begin{cases}
\widehat C_j, & A_j=1,\\
\widetilde C_{j-1}, & A_j=0.
\end{cases}
$$

If every potential replacement satisfies an epoch-specific certificate with failure probability bounded by $\delta_j+\varepsilon_j$, then

$$
\Pr\left\{\exists j\leq J:R(\widetilde C_j)>\alpha\right\}
\leq
\sum_{j=1}^{J}(\delta_j+\varepsilon_j),
$$

up to the mixing-gap terms stated in the paper. The trigger only chooses among already certified service transitions; it does not create a new statistical failure event.

## 6. Scope limitations

The implementation and theorem do not establish:

- arbitrary-stopping validity at every observation;
- a confidence sequence for the deployed risk;
- validity after arbitrary distribution shift;
- a guarantee based only on the random number of alarms without spending over the potential checkpoints;
- a theorem that a monitoring alarm detects every change within a fixed delay.

The piecewise-stationary experiment measures update economy and detection delay under regime deterioration. It is deliberately labeled as a stress test rather than a consequence of the stationary theorem.
