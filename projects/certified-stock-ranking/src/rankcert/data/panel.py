"""Merge, timing, and universe filters for the monthly research panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


CRSP_PREDICTOR_COLUMNS = ["Price", "Size", "STreversal"]


def merge_osap_crsp(
    osap: pd.DataFrame,
    crsp: pd.DataFrame,
    *,
    daily_liquidity: pd.DataFrame | None = None,
    min_price: float = 5.0,
    min_market_cap: float | None = None,
) -> pd.DataFrame:
    """Merge the signed OSAP panel with CRSP targets and formation variables."""

    left = osap.copy()
    right = crsp.copy()
    left["date"] = pd.to_datetime(left["date"]) + pd.offsets.MonthEnd(0)
    right["date"] = pd.to_datetime(right["date"]) + pd.offsets.MonthEnd(0)
    panel = left.merge(
        right,
        on=["permno", "date"],
        how="inner",
        validate="one_to_one",
    )
    panel = panel.loc[panel["prc"].abs() >= min_price]
    if min_market_cap is not None:
        panel = panel.loc[panel["market_cap"] >= min_market_cap]
    panel = panel.dropna(subset=["ret_fwd", "market_cap"])

    # Canonical names match the three CRSP-derived predictors excluded from the
    # downloadable OSAP wide file.
    panel["Price"] = panel["price"]
    panel["Size"] = panel["size"]
    panel["STreversal"] = panel["st_reversal"]

    if daily_liquidity is not None:
        liquidity = daily_liquidity.copy()
        liquidity["date"] = pd.to_datetime(liquidity["date"]) + pd.offsets.MonthEnd(0)
        panel = panel.merge(
            liquidity,
            on=["permno", "date"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_daily"),
        )

    panel = panel.rename(columns={"permno": "asset_id"})
    duplicate_count = int(panel.duplicated(["asset_id", "date"]).sum())
    if duplicate_count:
        raise ValueError(f"merged panel contains {duplicate_count} duplicate asset-month rows")
    return panel.sort_values(["date", "asset_id"]).reset_index(drop=True)


def build_feature_manifest(
    osap_feature_names: Iterable[str],
    *,
    osap_release: str | None = None,
) -> dict[str, object]:
    """Return the exact feature list used by the forecast models."""

    osap_features = list(dict.fromkeys(str(name) for name in osap_feature_names))
    features = list(dict.fromkeys(osap_features + CRSP_PREDICTOR_COLUMNS))
    return {
        "osap_release": osap_release,
        "n_osap_features": len(osap_features),
        "crsp_derived_features": CRSP_PREDICTOR_COLUMNS,
        "n_total_features": len(features),
        "features": features,
    }


def write_feature_manifest(
    output_path: str | Path,
    osap_feature_names: Iterable[str],
    *,
    osap_release: str | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            build_feature_manifest(osap_feature_names, osap_release=osap_release),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output


def read_feature_manifest(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list) or not features or not all(
        isinstance(name, str) for name in features
    ):
        raise ValueError("feature manifest does not contain a valid 'features' list")
    return features
