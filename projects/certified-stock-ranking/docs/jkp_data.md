# JKP Global Factor Data

The production baseline uses the precomputed monthly stock-level panel in
`contrib.global_factor` on WRDS. It is the dataset associated with Jensen,
Kelly, and Pedersen (2023). The table contains identifiers, standardized firm
characteristics, and future stock returns in one panel, so the baseline does not
reconstruct hundreds of characteristics or merge an external signal file to
CRSP.

## Baseline sample

- Country: United States (`excntry='USA'`).
- Standard JKP screens: `common=1`, `exch_main=1`, `primary_sec=1`, and
  `obs_main=1`.
- Characteristics: the 153 published JKP signals listed in the pinned official
  `factor_details.xlsx` workbook.
- Target: `ret_exc_lead1m`, the next-month excess return already aligned to the
  formation month.
- Dates: January 1963 through December 2025 by default.
- Descriptive ranking results use the screened sample. Baseline investable
  portfolios exclude the JKP micro and nano size groups; those groups are
  restored in robustness exercises.

The downloader verifies the Git blob of the official factor-details workbook,
queries the WRDS schema before extraction, refuses a partial characteristic
set, and writes annual compressed Parquet partitions plus manifests. It never
writes credentials to disk.

## Secure extraction

Set credentials only in the current shell:

```bash
export WRDS_USERNAME='your_username'
export WRDS_PASSWORD='your_password'
```

Then run:

```bash
pip install -e '.[wrds,parquet]'
PYTHONPATH=src python scripts/download_jkp_wrds.py \
  --output-dir data/processed/jkp_usa_panel \
  --start 1963-01-01 \
  --end 2025-12-31
unset WRDS_PASSWORD
```

WRDS may require approval of a Duo request. The extraction uses a persistent
connection and annual chunks to control memory. Licensed stock-level data stay
under `data/`, which is ignored by Git.

## Alternative prebuilt WRDS tables

The JKP Common Task Framework tables are also available on WRDS:

- `contrib_global_factor.ctff_chars`;
- `contrib_global_factor.ctff_features`;
- `contrib_global_factor.ctff_daily_ret`.

They are useful as a standardized external benchmark and include a complete
model-input interface. Download them with:

```bash
PYTHONPATH=src python scripts/download_jkp_ctf_wrds.py \
  --output-dir data/processed/jkp_ctf
```

The paper baseline nevertheless uses `contrib.global_factor`
because it offers a transparent historical stock-month panel and the published
153-characteristic taxonomy in one table. The CTF dataset is retained as a
robustness and benchmarking design, not mixed into the headline specification.

## Transaction-cost inputs

A ready-made characteristic/return panel does not by itself solve transaction
cost measurement. The primary cost analysis therefore supplements JKP with
CRSP daily price and volume data when available. JKP liquidity and spread
characteristics are used for robustness, not silently interpreted as exact
realized trading costs.
