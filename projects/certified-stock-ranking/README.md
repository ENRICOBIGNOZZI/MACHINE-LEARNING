# Certified Stock Ranking

Research repository for the working paper **When Should Investors Trust a Machine-Learned Stock Ranking?**

Machine-learning return forecasts mechanically produce a total ordering of the stock cross-section. This project asks a different economic question: which predicted rank relations are reliable enough to justify capital, turnover, and transaction costs?

The repository converts a total ranking into a nested family of **partial orders**. For stock `i` and stock `j`, the procedure asserts `i > j` at threshold `q` only when the lower score band of `i` exceeds the upper score band of `j`. Otherwise it abstains. A self-normalized risk certificate selects the least-abstaining threshold whose future pairwise decision loss is controlled using one temporally dependent financial history.

## Status

The statistical layer, exact all-pair and top-versus-bottom counting algorithms, matched portfolio maps, drift-aware turnover, explicit feature manifests, JKP/WRDS and CRSP adapters, tests, synthetic validation, paper source, and exhibit plan are complete. **No empirical stock-return result is fabricated or embedded.** Files under `results/synthetic/` are software validation only.

## Main data design

The production baseline is the prebuilt **Jensen-Kelly-Pedersen Global Factor Data** stock-month table on WRDS:

1. `contrib.global_factor` supplies identifiers, more than 400 available firm characteristics, and next-month stock returns in a single aligned panel;
2. the headline specification uses the published 153-characteristic JKP set pinned to an exact upstream release;
3. the US sample applies the standard JKP screens `common=1`, `exch_main=1`, `primary_sec=1`, and `obs_main=1`;
4. the prediction target is the already aligned `ret_exc_lead1m` field;
5. CRSP daily price and volume are an optional supplement for capacity-aware transaction-cost estimates.

This is cleaner than reconstructing the panel from Open Source Asset Pricing plus CRSP. The legacy OSAP pipeline remains only as an independent robustness path. Licensed data and credentials are excluded by `.gitignore`.

The JKP Common Task Framework tables are also supported conceptually as an external benchmark: `contrib_global_factor.ctff_chars`, `ctff_features`, and `ctff_daily_ret`. See `docs/jkp_data.md`.

## Core economic objects

For forecast `mu[i,t]`, positive scale `s[i,t]`, and threshold `q`, define

```text
lower[i,t](q) = mu[i,t] - q * s[i,t]
upper[i,t](q) = mu[i,t] + q * s[i,t]

i outranks j  <=>  lower[i,t](q) > upper[j,t](q)
```

This relation is a strict partial order and is nested as `q` increases.

- **Reliable breadth:** weighted share of raw pairwise rank relations retained after abstention.
- **Selective pairwise error:** share of active relations contradicted by next-month realized returns.
- **Monotone loss envelope:** the right-supremum of selective pairwise error across stricter thresholds.
- **Certified portfolio:** aggregate of active pair trades through graph out-degree minus in-degree.
- **Uncertified component:** exact removed-edge component under the fixed-denominator edge representation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_synthetic_demo.py
make paper
```

## Secure one-command WRDS extraction

Do not put WRDS credentials in the repository. Export them only in the current process:

```bash
export WRDS_USERNAME='your_username'
export WRDS_PASSWORD='your_password'
pip install -e '.[wrds,parquet]'
PYTHONPATH=src python scripts/download_jkp_wrds.py \
  --output-dir data/processed/jkp_usa_panel \
  --start 1963-01-01 \
  --end 2025-12-31
unset WRDS_PASSWORD
```

The downloader:

- pins and verifies the official JKP factor-details workbook;
- checks the live WRDS schema before querying;
- refuses to continue if any of the 153 characteristics is missing;
- keeps one persistent SSL connection to reduce repeated MFA prompts;
- downloads annual partitions to compressed Parquet;
- writes extraction and feature manifests without credentials.

Generate leakage-safe prequential forecasts and the prespecified ensemble:

```bash
PYTHONPATH=src python scripts/run_model_grid.py \
  --panel data/processed/jkp_usa_panel \
  --output-dir data/processed/predictions
```

Run episodic certification:

```bash
PYTHONPATH=src python scripts/run_paper_certification.py \
  --predictions data/processed/predictions/ensemble.parquet \
  --output results/paper/ensemble
```

## Repository map

```text
paper/                  Finance paper and bibliography
src/rankcert/snrcps.py  Self-normalized certification
src/rankcert/partial_order.py
                        Partial orders and exact pairwise losses
src/rankcert/portfolio.py
                        Raw/certified graph portfolios, turnover, and costs
src/rankcert/backtest.py
                        Episodic certification and deployment
src/rankcert/data/jkp.py
                        Secure JKP/WRDS extraction and canonical panel adapter
src/rankcert/data/      Legacy OSAP and direct CRSP robustness adapters
scripts/                Reproduction entrypoints
configs/                Prespecified paper and synthetic settings
docs/                   Data, research design, and claims discipline
results/synthetic/       Software validation only
tests/                   Unit tests
```

## Reproducibility rules

- Forecasts and scales are generated prequentially by a predetermined finite-memory algorithm.
- The primary candidate thresholds are fixed before deployment data are observed.
- Certification outcomes never tune or refit the forecasting algorithm.
- The exact JKP feature taxonomy, source commit, WRDS table, screens, dates, query columns, and partition counts are stored in manifests.
- Turnover uses passive post-return weights over the union of consecutive universes.
- Raw/licensed files, passwords, `.env`, `.pgpass`, and WRDS state are never committed.

## Citation

The code is MIT licensed. The paper remains a working paper; see `CITATION.cff`.
