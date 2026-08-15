"""WRDS/CRSP extraction helpers with explicit schema and unit controls.

The defaults target the classic CRSP monthly/daily stock files on WRDS. CRSP
legacy monthly ``VOL`` is reported in hundreds of shares, whereas daily ``VOL``
is in shares. Users of CRSP Stock v2 should pass the appropriate table names and
unit multipliers after checking their subscription's data dictionary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


CRSP_MONTHLY_SQL = """
select
    a.permno,
    a.date,
    a.ret,
    a.retx,
    abs(a.prc) as prc,
    a.shrout,
    a.vol,
    n.shrcd,
    n.exchcd,
    d.dlret,
    d.dlstcd
from {monthly_table} as a
left join {names_table} as n
  on a.permno = n.permno
 and n.namedt <= a.date
 and a.date <= n.nameendt
left join {delist_table} as d
  on a.permno = d.permno
 and date_trunc('month', a.date) = date_trunc('month', d.dlstdt)
where a.date between %(start_date)s and %(end_date)s
  and n.shrcd in (10, 11)
  and n.exchcd in (1, 2, 3)
order by a.permno, a.date
"""

CRSP_DAILY_LIQUIDITY_SQL = """
select
    permno,
    date,
    ret,
    abs(prc) as prc,
    vol
from {daily_table}
where date between %(start_date)s and %(end_date)s
order by permno, date
"""


def _validated_table_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"invalid WRDS table name: {value!r}")
    return value


def clean_crsp_monthly(
    frame: pd.DataFrame,
    *,
    volume_multiplier: float = 100.0,
) -> pd.DataFrame:
    """Clean classic CRSP monthly data and create next-month total returns."""

    if volume_multiplier <= 0.0:
        raise ValueError("volume_multiplier must be positive")
    required = {"permno", "date", "ret", "prc", "shrout", "vol", "dlret"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CRSP monthly extract is missing columns: {sorted(missing)}")

    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"]) + pd.offsets.MonthEnd(0)
    for column in ["ret", "retx", "prc", "shrout", "vol", "dlret"]:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    duplicate_count = int(output.duplicated(["permno", "date"]).sum())
    if duplicate_count:
        raise ValueError(
            f"CRSP query returned {duplicate_count} duplicate permno-month rows; "
            "inspect names/delisting joins before continuing"
        )

    ordinary_return = output["ret"]
    delisting_return = output["dlret"].fillna(0.0)
    output["ret_adjusted"] = (1.0 + ordinary_return) * (1.0 + delisting_return) - 1.0
    # Do not invent an ordinary return when CRSP reports it missing.
    output.loc[ordinary_return.isna(), "ret_adjusted"] = pd.NA
    output["market_cap"] = output["prc"].abs() * output["shrout"] * 1_000.0
    output["share_volume"] = output["vol"] * volume_multiplier
    output["dollar_volume"] = output["prc"].abs() * output["share_volume"]

    output = output.sort_values(["permno", "date"]).reset_index(drop=True)
    output["ret_fwd"] = output.groupby("permno", sort=False)["ret_adjusted"].shift(-1)
    output["size"] = output["market_cap"]
    output["price"] = output["prc"].abs()
    output["st_reversal"] = output.groupby("permno", sort=False)["ret_adjusted"].shift(0)
    return output


def aggregate_daily_liquidity(
    frame: pd.DataFrame,
    *,
    min_days: int = 10,
    volume_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Aggregate daily CRSP data into formation-month liquidity inputs."""

    if min_days <= 0 or volume_multiplier <= 0.0:
        raise ValueError("min_days and volume_multiplier must be positive")
    required = {"permno", "date", "ret", "prc", "vol"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CRSP daily extract is missing columns: {sorted(missing)}")

    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"] + pd.offsets.MonthEnd(0)
    for column in ["ret", "prc", "vol"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["share_volume"] = daily["vol"] * volume_multiplier
    daily["dollar_volume"] = daily["prc"].abs() * daily["share_volume"]

    grouped = daily.groupby(["permno", "month"], sort=True)
    output = (
        grouped.agg(
            adv_dollars=("dollar_volume", "mean"),
            daily_volatility=("ret", "std"),
            trading_days=("ret", "count"),
        )
        .reset_index()
        .rename(columns={"month": "date"})
    )
    return output.loc[output["trading_days"] >= min_days].reset_index(drop=True)


def _connect(username: str | None = None):
    try:
        import wrds  # type: ignore
    except ImportError as exc:
        raise RuntimeError("install the 'wrds' optional dependency") from exc
    return wrds.Connection(wrds_username=username)


def _write_frame(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)
    return output


def download_crsp_monthly(
    output_path: str | Path,
    *,
    start_date: str = "1957-01-01",
    end_date: str = "2026-12-31",
    monthly_table: str = "crsp.msf",
    names_table: str = "crsp.msenames",
    delist_table: str = "crsp.msedelist",
    volume_multiplier: float = 100.0,
    wrds_username: str | None = None,
) -> Path:
    """Download and clean the common-share CRSP monthly panel."""

    monthly_table = _validated_table_name(monthly_table)
    names_table = _validated_table_name(names_table)
    delist_table = _validated_table_name(delist_table)
    sql = CRSP_MONTHLY_SQL.format(
        monthly_table=monthly_table,
        names_table=names_table,
        delist_table=delist_table,
    )
    connection = _connect(wrds_username)
    try:
        raw = connection.raw_sql(
            sql,
            params={"start_date": start_date, "end_date": end_date},
        )
    finally:
        connection.close()
    clean = clean_crsp_monthly(raw, volume_multiplier=volume_multiplier)
    return _write_frame(clean, output_path)


def download_crsp_daily_liquidity(
    output_path: str | Path,
    *,
    start_date: str = "1957-01-01",
    end_date: str = "2026-12-31",
    daily_table: str = "crsp.dsf",
    volume_multiplier: float = 1.0,
    wrds_username: str | None = None,
) -> Path:
    """Download daily CRSP inputs and aggregate them to month end."""

    daily_table = _validated_table_name(daily_table)
    sql = CRSP_DAILY_LIQUIDITY_SQL.format(daily_table=daily_table)
    connection = _connect(wrds_username)
    try:
        raw = connection.raw_sql(
            sql,
            params={"start_date": start_date, "end_date": end_date},
        )
    finally:
        connection.close()
    clean = aggregate_daily_liquidity(raw, volume_multiplier=volume_multiplier)
    return _write_frame(clean, output_path)
