"""Model-agnostic forecast and scale estimators for monthly equity panels."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor


@dataclass
class FittedForecastModel:
    model_name: str
    feature_names: list[str]
    estimator: object
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(frame[self.feature_names].to_numpy(dtype=float)), dtype=float)


def cross_sectional_rank_transform(frame: pd.DataFrame, feature_names: list[str], date_column: str = "date") -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby(date_column, sort=False)
    for col in feature_names:
        out[col] = (2 * grouped[col].rank(method="average", pct=True) - 1).fillna(0.0)
    return out


def fit_forecast_model(
    train: pd.DataFrame,
    feature_names: list[str],
    target: str,
    *,
    model_name: Literal["ridge", "elastic_net", "extra_trees", "hist_gb", "mlp"] = "ridge",
    random_state: int = 20260814,
) -> FittedForecastModel:
    x, y = train[feature_names].to_numpy(float), train[target].to_numpy(float)
    if model_name == "ridge": estimator = Ridge(alpha=10.0)
    elif model_name == "elastic_net": estimator = ElasticNet(alpha=1e-4, l1_ratio=.5, max_iter=10_000, random_state=random_state)
    elif model_name == "extra_trees": estimator = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=100, max_features=.5, n_jobs=-1, random_state=random_state)
    elif model_name == "hist_gb": estimator = HistGradientBoostingRegressor(learning_rate=.04, max_iter=300, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, random_state=random_state)
    elif model_name == "mlp": estimator = MLPRegressor(hidden_layer_sizes=(64,32), alpha=1e-3, batch_size=2048, max_iter=100, early_stopping=True, random_state=random_state)
    else: raise ValueError(f"unsupported model_name={model_name}")
    estimator.fit(x, y)
    return FittedForecastModel(model_name, feature_names, estimator)


def fit_residual_scale_model(validation: pd.DataFrame, feature_names: list[str], residual_column: str, *, random_state: int = 20260814) -> FittedForecastModel:
    data = validation.copy()
    target = "__log_abs_residual__"
    data[target] = np.log(np.abs(data[residual_column].to_numpy(float)) + 1e-4)
    est = HistGradientBoostingRegressor(learning_rate=.04, max_iter=250, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=2.0, random_state=random_state)
    est.fit(data[feature_names].to_numpy(dtype=float), data[target].to_numpy(dtype=float))
    return FittedForecastModel("residual_scale", feature_names, est)


def predict_positive_scale(model: FittedForecastModel, frame: pd.DataFrame, floor: float = 1e-4) -> np.ndarray:
    return np.maximum(np.exp(model.predict(frame)), floor)
