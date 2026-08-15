"""JKP/WRDS stock-level data adapter.

The preferred production input is ``contrib.global_factor`` from Jensen,
Kelly, and Pedersen.  It already contains stock identifiers, monthly firm
characteristics, and the next-month excess return.  Credentials are read from
environment variables and are never written to disk by this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable, Sequence
from urllib.request import urlopen

import pandas as pd


JKP_SCHEMA = "contrib"
JKP_TABLE = "global_factor"
JKP_COUNTRY = "USA"
JKP_TARGET = "ret_exc_lead1m"
JKP_FACTOR_DETAILS_COMMIT = "6fb206b42b5778ba0a1d12e3fec0c23ac6b1c251"
JKP_FACTOR_DETAILS_GIT_BLOB = "a5402a84b62f9a3610e31d2054332e3c9dd20bd9"
JKP_FACTOR_DETAILS_URL = (
    "https://raw.githubusercontent.com/bkelly-lab/jkp-data/"
    f"{JKP_FACTOR_DETAILS_COMMIT}/src/jkp/data/resources/factor_details.xlsx"
)

IDENTIFIER_COLUMNS = [
    "id",
    "eom",
    "excntry",
    "gvkey",
    "permno",
    "size_grp",
    "me",
    JKP_TARGET,
]
SCREEN_COLUMNS = ["common", "exch_main", "primary_sec", "obs_main"]
OPTIONAL_IMPLEMENTATION_COLUMNS = [
    "prc",
    "ret_exc",
    "ret_1m",
    "dolvol_126d",
    "turnover_126d",
    "ami_126d",
    "bidaskhl_21d",
]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validated_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    return value


def _git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324 - Git object hash


def load_factor_details_bytes(path: str | Path | None = None) -> bytes:
    """Load the pinned official factor-details workbook and verify its Git blob.

    Passing a local path is useful on restricted machines.  Otherwise the file
    is fetched from the exact upstream commit, not from a moving ``main`` URL.
    """

    if path is None:
        with urlopen(JKP_FACTOR_DETAILS_URL, timeout=60) as response:  # noqa: S310
            content = response.read()
    else:
        content = Path(path).read_bytes()
    observed = _git_blob_sha(content)
    if observed != JKP_FACTOR_DETAILS_GIT_BLOB:
        raise ValueError(
            "factor-details workbook does not match the pinned JKP release: "
            f"expected {JKP_FACTOR_DETAILS_GIT_BLOB}, observed {observed}"
        )
    return content


def load_jkp_factor_names(path: str | Path | None = None) -> list[str]:
    """Return the 153 JKP characteristic mnemonics from the pinned workbook."""

    workbook = pd.read_excel(io.BytesIO(load_factor_details_bytes(path)))
    if "abr_jkp" not in workbook.columns:
        raise ValueError("factor-details workbook is missing the 'abr_jkp' column")
    names = [
        str(value).strip()
        for value in workbook["abr_jkp"].dropna().tolist()
        if str(value).strip()
    ]
    names = list(dict.fromkeys(names))
    for name in names:
        _validated_identifier(name)
    if len(names) != 153:
        raise ValueError(f"expected 153 JKP characteristics, found {len(names)}")
    return names


def create_wrds_engine(
    *,
    username: str | None = None,
    password: str | None = None,
):
    """Create an SSL WRDS SQLAlchemy engine without persisting credentials."""

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install the WRDS extra: pip install -e '.[wrds,parquet]'"
        ) from exc

    wrds_username = username or os.environ.get("WRDS_USERNAME")
    wrds_password = password or os.environ.get("WRDS_PASSWORD")
    if not wrds_username or not wrds_password:
        raise RuntimeError(
            "set WRDS_USERNAME and WRDS_PASSWORD in the process environment; "
            "credentials are intentionally not read from project files"
        )
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=wrds_username,
        password=wrds_password,
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        database="wrds",
        query={"sslmode": "require"},
    )
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 30})


def list_table_columns(engine, *, schema: str = JKP_SCHEMA, table: str = JKP_TABLE) -> list[str]:
    """Return table columns in database order."""

    from sqlalchemy import text

    schema = _validated_identifier(schema)
    table = _validated_identifier(table)
    query = text(
        """
        select column_name
        from information_schema.columns
        where table_schema = :schema and table_name = :table
        order by ordinal_position
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(query, connection, params={"schema": schema, "table": table})
    columns = frame["column_name"].astype(str).tolist()
    if not columns:
        raise RuntimeError(f"WRDS table {schema}.{table} was not found or is not accessible")
    return columns


def select_jkp_columns(
    available_columns: Sequence[str],
    feature_names: Sequence[str],
    *,
    include_optional_implementation_columns: bool = True,
) -> list[str]:
    """Build a strict, leakage-safe column list for the US ranking panel."""

    available = set(available_columns)
    required = set(IDENTIFIER_COLUMNS + SCREEN_COLUMNS)
    missing_required = sorted(required.difference(available))
    if missing_required:
        raise ValueError(f"JKP table is missing required columns: {missing_required}")

    missing_features = sorted(set(feature_names).difference(available))
    if missing_features:
        raise ValueError(
            "JKP table does not contain all pinned characteristics; first missing columns: "
            f"{missing_features[:20]}"
        )

    selected = list(IDENTIFIER_COLUMNS + SCREEN_COLUMNS)
    if include_optional_implementation_columns:
        selected.extend(
            column for column in OPTIONAL_IMPLEMENTATION_COLUMNS if column in available
        )
    selected.extend(feature_names)
    return list(dict.fromkeys(selected))


def build_jkp_sql(
    columns: Sequence[str],
    *,
    schema: str = JKP_SCHEMA,
    table: str = JKP_TABLE,
) -> str:
    """Return the parameterized annual-chunk query used by the downloader."""

    schema = _validated_identifier(schema)
    table = _validated_identifier(table)
    quoted = ",\n    ".join(f'"{_validated_identifier(column)}"' for column in columns)
    return f"""
select
    {quoted}
from "{schema}"."{table}"
where common = 1
  and exch_main = 1
  and primary_sec = 1
  and obs_main = 1
  and excntry = :country
  and eom >= :chunk_start
  and eom <= :chunk_end
order by eom, id
""".strip()


def prepare_jkp_panel(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Convert a raw JKP extract to the canonical panel used by ``rankcert``."""

    required = {"id", "eom", "permno", "me", JKP_TARGET, *feature_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"raw JKP extract is missing columns: {missing[:20]}")

    output = frame.copy()
    output["date"] = pd.to_datetime(output["eom"]) + pd.offsets.MonthEnd(0)
    output["asset_id"] = pd.to_numeric(output["permno"], errors="coerce").astype("Int64")
    if output["asset_id"].isna().any():
        raise ValueError("US JKP extract contains rows without PERMNO after standard screens")
    output["ret_fwd"] = pd.to_numeric(output[JKP_TARGET], errors="coerce")
    output["market_cap"] = pd.to_numeric(output["me"], errors="coerce")
    output["tradable_baseline"] = output.get("size_grp", "").isin(
        ["mega", "large", "small"]
    )
    for column in feature_names:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    duplicate_count = int(output.duplicated(["asset_id", "date"]).sum())
    if duplicate_count:
        raise ValueError(f"JKP panel contains {duplicate_count} duplicate asset-month rows")

    leading = [
        "date",
        "asset_id",
        "ret_fwd",
        "market_cap",
        "tradable_baseline",
        "size_grp",
        "id",
        "permno",
        "gvkey",
        "excntry",
    ]
    implementation = [
        column
        for column in OPTIONAL_IMPLEMENTATION_COLUMNS
        if column in output.columns and column not in leading
    ]
    ordered = [column for column in leading if column in output.columns]
    ordered += implementation
    ordered += [column for column in feature_names if column not in ordered]
    return output[ordered].sort_values(["date", "asset_id"]).reset_index(drop=True)


def _year_chunks(start_date: str | date, end_date: str | date) -> Iterable[tuple[date, date]]:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start > end:
        raise ValueError("start_date must not be after end_date")
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, date(year, 1, 1))
        chunk_end = min(end, date(year, 12, 31))
        yield chunk_start, chunk_end


def download_jkp_usa_panel(
    output_dir: str | Path,
    *,
    start_date: str = "1963-01-01",
    end_date: str = "2025-12-31",
    factor_details_path: str | Path | None = None,
    schema: str = JKP_SCHEMA,
    table: str = JKP_TABLE,
    username: str | None = None,
    password: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Download the screened US JKP panel in annual Parquet partitions.

    The function holds one persistent SQLAlchemy engine for the full extraction,
    which minimizes repeated MFA challenges.  No credential value is written to
    the extraction manifest.
    """

    from sqlalchemy import text

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    factor_names = load_jkp_factor_names(factor_details_path)
    engine = create_wrds_engine(username=username, password=password)
    try:
        available = list_table_columns(engine, schema=schema, table=table)
        selected = select_jkp_columns(available, factor_names)
        sql = text(build_jkp_sql(selected, schema=schema, table=table))
        partition_records: list[dict[str, object]] = []
        with engine.connect() as connection:
            for chunk_start, chunk_end in _year_chunks(start_date, end_date):
                output = destination / f"year={chunk_start.year}" / "part-000.parquet"
                if output.exists() and not overwrite:
                    existing = pd.read_parquet(
                        output, columns=["date", "asset_id", "ret_fwd"]
                    )
                    partition_records.append(
                        {
                            "year": chunk_start.year,
                            "path": str(output.relative_to(destination)),
                            "rows": int(len(existing)),
                            "status": "retained",
                        }
                    )
                    continue
                raw = pd.read_sql_query(
                    sql,
                    connection,
                    params={
                        "country": JKP_COUNTRY,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    },
                )
                clean = prepare_jkp_panel(raw, factor_names)
                output.parent.mkdir(parents=True, exist_ok=True)
                clean.to_parquet(output, index=False, compression="zstd")
                partition_records.append(
                    {
                        "year": chunk_start.year,
                        "path": str(output.relative_to(destination)),
                        "rows": int(len(clean)),
                        "status": "downloaded",
                    }
                )
    finally:
        engine.dispose()

    manifest = {
        "dataset": "Jensen-Kelly-Pedersen Global Factor Data",
        "wrds_table": f"{schema}.{table}",
        "country": JKP_COUNTRY,
        "screens": {column: 1 for column in SCREEN_COLUMNS},
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "target": JKP_TARGET,
        "n_features": len(factor_names),
        "features": factor_names,
        "selected_columns": selected,
        "factor_details_commit": JKP_FACTOR_DETAILS_COMMIT,
        "factor_details_git_blob": JKP_FACTOR_DETAILS_GIT_BLOB,
        "extracted_at_utc": datetime.now(UTC).isoformat(),
        "partitions": partition_records,
        "credentials_persisted": False,
    }
    manifest_path = destination / "jkp_extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    feature_manifest = {
        "source": "JKP Global Factor Data",
        "release_commit": JKP_FACTOR_DETAILS_COMMIT,
        "n_total_features": len(factor_names),
        "features": factor_names,
    }
    feature_manifest_path = Path(str(destination) + ".features.json")
    feature_manifest_path.write_text(
        json.dumps(feature_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path, feature_manifest_path

CTF_SCHEMA = "contrib_global_factor"
CTF_TABLES = ("ctff_features", "ctff_chars", "ctff_daily_ret")


def download_jkp_ctf_tables(
    output_dir: str | Path,
    *,
    chunksize: int = 500_000,
    username: str | None = None,
    password: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download the three prebuilt JKP Common Task Framework WRDS tables."""

    from sqlalchemy import text

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    engine = create_wrds_engine(username=username, password=password)
    table_records: list[dict[str, object]] = []
    try:
        with engine.connect() as connection:
            for table in CTF_TABLES:
                table_dir = destination / table
                existing = sorted(table_dir.glob("part-*.parquet"))
                if existing and not overwrite:
                    table_records.append(
                        {
                            "table": f"{CTF_SCHEMA}.{table}",
                            "parts": len(existing),
                            "status": "retained",
                        }
                    )
                    continue
                if overwrite and table_dir.exists():
                    for path in table_dir.glob("part-*.parquet"):
                        path.unlink()
                table_dir.mkdir(parents=True, exist_ok=True)
                query = text(f'SELECT * FROM "{CTF_SCHEMA}"."{_validated_identifier(table)}"')
                rows = 0
                columns: list[str] = []
                parts = 0
                iterator = pd.read_sql_query(query, connection, chunksize=chunksize)
                for part_number, chunk in enumerate(iterator):
                    if not columns:
                        columns = chunk.columns.astype(str).tolist()
                    output = table_dir / f"part-{part_number:05d}.parquet"
                    chunk.to_parquet(output, index=False, compression="zstd")
                    rows += len(chunk)
                    parts += 1
                table_records.append(
                    {
                        "table": f"{CTF_SCHEMA}.{table}",
                        "parts": parts,
                        "rows": int(rows),
                        "columns": columns,
                        "status": "downloaded",
                    }
                )
    finally:
        engine.dispose()

    manifest = {
        "dataset": "JKP Common Task Framework",
        "schema": CTF_SCHEMA,
        "tables": table_records,
        "chunksize": chunksize,
        "extracted_at_utc": datetime.now(UTC).isoformat(),
        "credentials_persisted": False,
    }
    manifest_path = destination / "ctf_extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
