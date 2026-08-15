from pathlib import Path

import numpy as np
import pandas as pd

from rankcert.data.osap import build_osap_manifest, load_osap_wide, parse_yyyymm
from rankcert.data.panel import build_feature_manifest
from rankcert.data.wrds_crsp import aggregate_daily_liquidity, clean_crsp_monthly


def test_parse_yyyymm_integer_and_yearmon() -> None:
    integer = parse_yyyymm(pd.Series([202001, 202012]))
    yearmon = parse_yyyymm(pd.Series([2020.0, 2020.0 + 11.0 / 12.0]))
    assert integer.dt.strftime("%Y-%m").tolist() == ["2020-01", "2020-12"]
    assert yearmon.dt.strftime("%Y-%m").tolist() == ["2020-01", "2020-12"]


def test_osap_manifest_and_feature_manifest(tmp_path: Path) -> None:
    source = tmp_path / "signals.csv"
    pd.DataFrame(
        {
            "permno": [1, 2],
            "yyyymm": [202001, 202001],
            "SignalA": [0.1, -0.1],
            "SignalB": [1.0, 2.0],
        }
    ).to_csv(source, index=False)
    frame, features = load_osap_wide(source)
    manifest = build_osap_manifest(source)
    feature_manifest = build_feature_manifest(features, osap_release="test")
    assert len(frame) == 2
    assert manifest["n_predictors_in_file"] == 2
    assert feature_manifest["features"][-3:] == ["Price", "Size", "STreversal"]


def test_crsp_volume_units_and_forward_return() -> None:
    raw = pd.DataFrame(
        {
            "permno": [1, 1],
            "date": ["2020-01-31", "2020-02-29"],
            "ret": [0.10, 0.20],
            "retx": [0.10, 0.20],
            "prc": [10.0, 11.0],
            "shrout": [100.0, 100.0],
            "vol": [2.0, 3.0],
            "dlret": [np.nan, -0.5],
        }
    )
    clean = clean_crsp_monthly(raw, volume_multiplier=100.0)
    assert np.isclose(clean.loc[0, "share_volume"], 200.0)
    assert np.isclose(clean.loc[0, "ret_fwd"], (1.2 * 0.5) - 1.0)


def test_daily_volume_defaults_to_shares() -> None:
    raw = pd.DataFrame(
        {
            "permno": [1, 1],
            "date": ["2020-01-02", "2020-01-03"],
            "ret": [0.01, -0.01],
            "prc": [10.0, 10.0],
            "vol": [100.0, 200.0],
        }
    )
    aggregated = aggregate_daily_liquidity(raw, min_days=1, volume_multiplier=1.0)
    assert np.isclose(aggregated.loc[0, "adv_dollars"], 1500.0)
