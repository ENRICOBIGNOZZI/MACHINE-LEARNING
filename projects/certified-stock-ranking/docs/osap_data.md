# Legacy robustness path: Open Source Asset Pricing plus CRSP

This pipeline is retained as an independent robustness check. The production baseline is the prebuilt JKP panel documented in `docs/jkp_data.md`.


The production predictor panel is the official **signed wide firm-level file** from Open Source Asset Pricing. Download a named release from the provider's data page and retain the untouched original archive under `data/raw/`.

The current public wide panel contains the firm-level signals distributed by the project. Price, size, and short-term reversal are constructed directly from CRSP and are added during the merge rather than treated as columns supplied by the wide predictor file.

## Required identifiers

The loader expects:

- `permno`: CRSP permanent security identifier;
- `yyyymm`: integer `YYYYMM` or R `yearmon` month;
- one or more predictor columns.

## Release pinning

Before any production run:

```bash
PYTHONPATH=src python scripts/inspect_osap_release.py \
  --input data/raw/signed_predictors_dl_wide.csv \
  --output data/raw/osap_release_manifest.json
```

The manifest records the SHA256 checksum, file size, row count, number of PERMNOs, number and names of predictors, and sample dates. Record the release label and download date manually in `configs/paper.yaml`. Do not replace a pinned file with a provider's newer `latest` download without changing the manifest.

The repository intentionally does not contain an unversioned automatic downloader. This prevents a future replication from silently using a different data vintage.