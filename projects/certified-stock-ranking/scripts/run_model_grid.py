"""Generate the prespecified model grid and an equal-weight ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rankcert.data.panel import read_feature_manifest
from rankcert.prequential import PrequentialConfig, generate_prequential_predictions


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_dir() or path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/predictions"))
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    arguments = parser.parse_args()

    configuration = yaml.safe_load(arguments.config.read_text())
    forecasting = configuration["forecasting"]
    panel = _read_frame(arguments.panel)
    manifest_path = arguments.feature_manifest or Path(str(arguments.panel) + ".features.json")
    features = read_feature_manifest(manifest_path)

    model_names = [name for name in forecasting["models"] if name != "ensemble"]
    prediction_frames: dict[str, pd.DataFrame] = {}
    for model_name in model_names:
        model_config = PrequentialConfig(
            train_months=int(forecasting.get("training_months", 120)),
            refit_frequency_months=int(forecasting.get("refit_frequency_months", 12)),
            scale_window_months=int(forecasting.get("scale_history_months", 60)),
            model_name=model_name,
            seed=int(configuration["project"]["seed"]),
        )
        predictions = generate_prequential_predictions(
            panel,
            feature_names=features,
            config=model_config,
        )
        predictions["model"] = model_name
        prediction_frames[model_name] = predictions
        _write_frame(
            predictions,
            arguments.output_dir / f"{model_name}.{arguments.format}",
        )

    if len(prediction_frames) >= 2:
        key_columns = ["date", "asset_id"]
        common = None
        prediction_columns = []
        scale_columns = []
        for model_name, frame in prediction_frames.items():
            selected = frame[key_columns + ["prediction", "scale", "ret_fwd"]].rename(
                columns={
                    "prediction": f"prediction_{model_name}",
                    "scale": f"scale_{model_name}",
                    "ret_fwd": f"ret_fwd_{model_name}",
                }
            )
            common = selected if common is None else common.merge(
                selected,
                on=key_columns,
                how="inner",
                validate="one_to_one",
            )
            prediction_columns.append(f"prediction_{model_name}")
            scale_columns.append(f"scale_{model_name}")
        assert common is not None
        return_columns = [column for column in common if column.startswith("ret_fwd_")]
        return_matrix = common[return_columns].to_numpy(dtype=float)
        if not np.allclose(return_matrix, return_matrix[:, [0]], equal_nan=True):
            raise ValueError("model panels disagree on next-month returns")
        prediction_matrix = common[prediction_columns].to_numpy(dtype=float)
        scale_matrix = common[scale_columns].to_numpy(dtype=float)
        ensemble = common[key_columns].copy()
        ensemble["prediction"] = prediction_matrix.mean(axis=1)
        model_dispersion = prediction_matrix.std(axis=1, ddof=0)
        ensemble["scale"] = np.sqrt(
            np.mean(scale_matrix * scale_matrix, axis=1) + model_dispersion * model_dispersion
        )
        ensemble["ret_fwd"] = return_matrix[:, 0]
        ensemble["model"] = "ensemble"
        _write_frame(
            ensemble,
            arguments.output_dir / f"ensemble.{arguments.format}",
        )
