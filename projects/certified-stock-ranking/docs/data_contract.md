# Canonical panel contract

The production forecast panel is an asset-month table with the following required columns:

- `date`: formation month end;
- `asset_id`: stable security identifier, PERMNO for the US baseline;
- `ret_fwd`: return realized after the formation month and used only as the next-month target;
- `market_cap`: formation-month market equity;
- the exact characteristic columns listed in the adjacent feature manifest.

The baseline source is `contrib.global_factor` on WRDS. Its `eom` field becomes
`date`, `permno` becomes `asset_id`, and `ret_exc_lead1m` becomes `ret_fwd`.
The four standard JKP screens are applied inside the SQL query. The downloader
refuses to infer features from arbitrary numeric columns and instead uses the
153 names in a pinned official factor-details release.

## Timing

Every predictor in row `(asset_id, date)` must be available at `date`. The
forward return is never included in the feature manifest. Forecasts are fitted
only on formation months whose forward returns are already observable at the
refit date. Certification and deployment windows advance chronologically.

## Universe

Descriptive reliability estimates use all observations passing the JKP screens.
Headline portfolios use `tradable_baseline=True`, corresponding to the mega,
large, and small JKP size groups. Micro and nano stocks enter explicit
robustness exercises rather than disappearing through an undocumented filter.

## Manifest requirements

The production archive records:

- WRDS schema and table;
- JKP factor-details commit and Git blob;
- country and standard screens;
- requested and realized sample dates;
- selected columns and exact 153-feature list;
- annual partition row counts;
- extraction timestamp and file checksums;
- CRSP daily cost-data vintage when the capacity analysis is run.

No manifest may contain a username, password, authentication token, or `.pgpass`
content.
