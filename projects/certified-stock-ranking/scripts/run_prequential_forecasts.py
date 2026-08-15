"""Generate leakage-safe rolling stock-return predictions and scales."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rankcert.data.panel import read_feature_manifest
from rankcert.prequential import PrequentialConfig, generate_prequential_predictions


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_dir() or path.suffix.lower() == ".parquet" else pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["ridge", "elastic_net", "extra_trees", "hist_gb", "mlp"], default="ridge")
    parser.add_argument("--training-months", type=int, default=120)
    parser.add_argument("--refit-months", type=int, default=12)
    parser.add_argument("--scale-history-months", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260814)
    arguments = parser.parse_args()

    panel = _read_frame(arguments.panel)
    manifest_path = arguments.feature_manifest or Path(str(arguments.panel) + ".features.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"feature manifest not found: {manifest_path}. "
            "Run download_jkp_wrds.py or pass --feature-manifest explicitly."
        )
    features = read_feature_manifest(manifest_path)
    missing_features = sorted(set(features).difference(panel.columns))
    if missing_features:
        raise ValueError(f"panel is missing manifest features: {missing_features[:20]}")

    config = PrequentialConfig(
        train_months=arguments.training_months,
        refit_frequency_months=arguments.refit_months,
        scale_window_months=arguments.scale_history_months,
        model_name=arguments.model,
        seed=arguments.seed,
    )
    predictions = generate_prequential_predictions(
        panel,
        feature_names=features,
        config=config,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.suffix.lower() == ".parquet":
        predictions.to_parquet(arguments.output, index=False)
    else:
        predictions.to_csv(arguments.output, index=False)
    print(arguments.output)
