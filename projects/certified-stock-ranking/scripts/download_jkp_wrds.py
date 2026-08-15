"""Download the prebuilt US JKP stock-characteristic and return panel from WRDS."""

from __future__ import annotations

import argparse
from pathlib import Path

from rankcert.data.jkp import download_jkp_usa_panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/jkp_usa_panel"),
    )
    parser.add_argument("--start", default="1963-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--factor-details", type=Path, default=None)
    parser.add_argument("--schema", default="contrib")
    parser.add_argument("--table", default="global_factor")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    extraction_manifest, feature_manifest = download_jkp_usa_panel(
        arguments.output_dir,
        start_date=arguments.start,
        end_date=arguments.end,
        factor_details_path=arguments.factor_details,
        schema=arguments.schema,
        table=arguments.table,
        overwrite=arguments.overwrite,
    )
    print(arguments.output_dir)
    print(extraction_manifest)
    print(feature_manifest)
