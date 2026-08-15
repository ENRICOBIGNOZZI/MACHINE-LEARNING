# CRSP extraction notes

The default SQL targets the classic WRDS tables `crsp.msf`, `crsp.msenames`, `crsp.msedelist`, and `crsp.dsf`. Table names are command-line arguments because CRSP Stock v2 subscriptions use different schemas and column conventions.

## Volume units

The defaults are explicit:

- classic monthly `VOL`: hundreds of shares, multiplier `100`;
- classic daily `VOL`: shares, multiplier `1`.

Before the production run, verify these units against the data dictionary attached to the actual subscription. Record the table names and multipliers in the replication manifest. A wrong multiplier does not affect return prediction directly, but it materially distorts ADV and capacity costs.

## Return alignment

The monthly cleaner constructs

```text
ret_adjusted = (1 + ret) * (1 + dlret) - 1
ret_fwd      = next month's ret_adjusted for the same PERMNO
```

A missing ordinary return remains missing even if a delisting return is present; this is an explicit conservative data rule rather than silent imputation. The panel merger drops rows without a next-month target.

## Duplicate audit

The cleaner stops if the names or delisting joins produce duplicate `permno`-month observations. Do not resolve duplicates with `drop_duplicates`; inspect the join histories and event dates.
