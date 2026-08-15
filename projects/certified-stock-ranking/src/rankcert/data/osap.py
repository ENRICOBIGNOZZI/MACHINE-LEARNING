"""Loader and manifest utilities for Open Source Asset Pricing releases.

The project deliberately does not automate the website download. Production
research should download a named official release, retain the original archive,
and record its checksum. This avoids silently changing data vintages when the
provider updates the latest file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_yyyymm(values: pd.Series) -> pd.Series:
    """Parse integer ``YYYYMM`` or R ``yearmon`` values to month end."""

    raw = pd.to_numeric(values, errors="coerce")
    nonmissing = raw.dropna()
    if nonmissing.empty:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if float(nonmissing.median()) > 10_000.0:
        yyyymm = raw.round().astype("Int64")
        text = yyyymm.astype("string")
        parsed = pd.to_datetime(text, format="%Y%m", errors="coerce")
        return parsed + pd.offsets.MonthEnd(0)

    years = np.floor(raw).astype("Int64")
    months_float = np.rint((raw - np.floor(raw)) * 12.0 + 1.0)
    months = pd.Series(months_float, index=values.index).clip(1, 12).astype("Int64")
    text = years.astype("string") + months.astype("string").str.zfill(2)
    parsed = pd.to_datetime(text, format="%Y%m", errors="coerce")
    return parsed + pd.offsets.MonthEnd(0)


def load_osap_wide(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the official signed wide firm-characteristic panel.

    The expected identifiers are ``permno`` and ``yyyymm``. Compression is
    inferred by pandas, so CSV, gzip, and one-file ZIP archives are accepted.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, low_memory=False, compression="infer")
    required = {"permno", "yyyymm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OSAP file is missing columns: {sorted(missing)}")
    frame["date"] = parse_yyyymm(frame["yyyymm"])
    frame["permno"] = pd.to_numeric(frame["permno"], errors="coerce").astype("Int64")
    feature_names = [
        column for column in frame.columns if column not in {"permno", "yyyymm", "date"}
    ]
    if not feature_names:
        raise ValueError("OSAP file contains no predictor columns")
    frame = frame.dropna(subset=["permno", "date"])
    duplicate_count = int(frame.duplicated(["permno", "date"]).sum())
    if duplicate_count:
        raise ValueError(
            f"OSAP file contains {duplicate_count} duplicate permno-month rows"
        )
    return frame.sort_values(["date", "permno"]).reset_index(drop=True), feature_names


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute a streaming SHA256 checksum."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_osap_manifest(path: str | Path) -> dict[str, object]:
    """Inspect an official release and return a reproducibility manifest."""

    source = Path(path)
    frame, features = load_osap_wide(source)
    return {
        "file_name": source.name,
        "file_size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "n_rows": int(len(frame)),
        "n_permno": int(frame["permno"].nunique()),
        "n_predictors_in_file": int(len(features)),
        "first_month": frame["date"].min().date().isoformat(),
        "last_month": frame["date"].max().date().isoformat(),
        "predictor_columns": features,
    }


def write_osap_manifest(path: str | Path, output_path: str | Path) -> Path:
    """Write a JSON release manifest next to the licensed/raw data."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_osap_manifest(path), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output
