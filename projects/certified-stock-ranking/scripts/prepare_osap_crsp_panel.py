"""Merge a pinned OSAP release with cleaned CRSP targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rankcert.data.osap import load_osap_wide
from rankcert.data.panel import merge_osap_crsp, write_feature_manifest


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--osap", type=Path, required=True)
    parser.add_argument("--crsp", type=Path, required=True)
    parser.add_argument("--daily-liquidity", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/osap_crsp_panel.parquet"),
    )
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument("--osap-release", default=None)
    parser.add_argument("--min-price", type=float, default=5.0)
    arguments = parser.parse_args()

    osap, osap_features = load_osap_wide(arguments.osap)
    crsp = _read_frame(arguments.crsp)
    liquidity = _read_frame(arguments.daily_liquidity) if arguments.daily_liquidity else None
    panel = merge_osap_crsp(
        osap,
        crsp,
        daily_liquidity=liquidity,
        min_price=arguments.min_price,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.suffix.lower() == ".parquet":
        panel.to_parquet(arguments.output, index=False)
    else:
        panel.to_csv(arguments.output, index=False)

    manifest_path = arguments.feature_manifest or Path(str(arguments.output) + ".features.json")
    write_feature_manifest(
        manifest_path,
        osap_features,
        osap_release=arguments.osap_release,
    )
    print(arguments.output)
    print(manifest_path)
