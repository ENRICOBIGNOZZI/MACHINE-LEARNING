"""Download the CRSP inputs used by the paper from WRDS."""

from __future__ import annotations

import argparse
from pathlib import Path

from rankcert.data.wrds_crsp import download_crsp_daily_liquidity, download_crsp_monthly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monthly-output",
        type=Path,
        default=Path("data/raw/crsp_monthly.parquet"),
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=Path("data/raw/crsp_daily_liquidity.parquet"),
    )
    parser.add_argument("--start", default="1957-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--username", default=None)
    parser.add_argument("--monthly-table", default="crsp.msf")
    parser.add_argument("--names-table", default="crsp.msenames")
    parser.add_argument("--delist-table", default="crsp.msedelist")
    parser.add_argument("--daily-table", default="crsp.dsf")
    parser.add_argument("--monthly-volume-multiplier", type=float, default=100.0)
    parser.add_argument("--daily-volume-multiplier", type=float, default=1.0)
    parser.add_argument("--with-daily", action="store_true")
    arguments = parser.parse_args()

    download_crsp_monthly(
        arguments.monthly_output,
        start_date=arguments.start,
        end_date=arguments.end,
        monthly_table=arguments.monthly_table,
        names_table=arguments.names_table,
        delist_table=arguments.delist_table,
        volume_multiplier=arguments.monthly_volume_multiplier,
        wrds_username=arguments.username,
    )
    if arguments.with_daily:
        download_crsp_daily_liquidity(
            arguments.daily_output,
            start_date=arguments.start,
            end_date=arguments.end,
            daily_table=arguments.daily_table,
            volume_multiplier=arguments.daily_volume_multiplier,
            wrds_username=arguments.username,
        )
