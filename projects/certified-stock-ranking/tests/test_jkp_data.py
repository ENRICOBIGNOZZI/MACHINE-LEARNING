from __future__ import annotations

import pandas as pd
import pytest

from rankcert.data.jkp import (
    IDENTIFIER_COLUMNS,
    SCREEN_COLUMNS,
    build_jkp_sql,
    prepare_jkp_panel,
    select_jkp_columns,
)


def test_jkp_select_columns_and_sql_are_strict() -> None:
    features = ["feature_a", "feature_b"]
    available = IDENTIFIER_COLUMNS + SCREEN_COLUMNS + features + ["prc"]
    selected = select_jkp_columns(available, features)
    assert selected[-2:] == features
    assert "prc" in selected

    sql = build_jkp_sql(selected)
    assert 'from "contrib"."global_factor"' in sql
    assert "common = 1" in sql
    assert "exch_main = 1" in sql
    assert "primary_sec = 1" in sql
    assert "obs_main = 1" in sql
    assert "excntry = :country" in sql
    assert "eom >= :chunk_start" in sql


def test_jkp_select_columns_refuses_partial_feature_set() -> None:
    available = IDENTIFIER_COLUMNS + SCREEN_COLUMNS + ["feature_a"]
    with pytest.raises(ValueError, match="does not contain all pinned characteristics"):
        select_jkp_columns(available, ["feature_a", "feature_b"])


def test_prepare_jkp_panel_uses_permno_and_pre_aligned_target() -> None:
    raw = pd.DataFrame(
        {
            "id": [10001, 10002],
            "eom": ["2020-01-31", "2020-01-31"],
            "excntry": ["USA", "USA"],
            "gvkey": ["001", "002"],
            "permno": [10101, 10102],
            "size_grp": ["large", "micro"],
            "me": [1000.0, 20.0],
            "ret_exc_lead1m": [0.03, -0.02],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    )
    panel = prepare_jkp_panel(raw, ["feature_a", "feature_b"])
    assert panel["asset_id"].tolist() == [10101, 10102]
    assert panel["ret_fwd"].tolist() == [0.03, -0.02]
    assert panel["tradable_baseline"].tolist() == [True, False]
    assert panel["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-31",
        "2020-01-31",
    ]
