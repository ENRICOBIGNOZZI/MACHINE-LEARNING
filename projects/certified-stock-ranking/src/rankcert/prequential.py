"""Leakage-safe rolling forecasts for the production stock panel."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .models import cross_sectional_rank_transform, fit_forecast_model, fit_residual_scale_model, predict_positive_scale


@dataclass(frozen=True)
class PrequentialConfig:
    model_name: str = "ridge"
    train_months: int = 120
    refit_frequency_months: int = 12
    scale_window_months: int = 60
    min_scale_observations: int = 10_000
    seed: int = 20260814


def generate_prequential_predictions(
    panel: pd.DataFrame,
    feature_names: list[str],
    *,
    target: str = "ret_fwd",
    config: PrequentialConfig = PrequentialConfig(),
) -> pd.DataFrame:
    """Generate forecasts using only targets observed before each refit date.

    The input target at formation month t is the return from t to t+1. At a
    refit starting in month t, training formation dates end at t-1, whose
    forward return is observable at t. Predictions are held for the declared
    refit frequency. The algorithm therefore has a fixed finite memory.
    """
    required={"date","asset_id",target,*feature_names}
    missing=required.difference(panel.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    data=panel.copy(); data["date"]=pd.to_datetime(data.date)
    data=cross_sectional_rank_transform(data,feature_names,"date")
    data=data.sort_values(["date","asset_id"]).reset_index(drop=True)
    dates=np.array(sorted(data.date.unique()),dtype="datetime64[ns]")
    rows=[]; residual_history=[]
    start=config.train_months
    while start < len(dates):
        end=min(start+config.refit_frequency_months,len(dates))
        train_dates=pd.to_datetime(dates[start-config.train_months:start])
        predict_dates=pd.to_datetime(dates[start:end])
        train=data.loc[data.date.isin(train_dates)].dropna(subset=[target])
        future=data.loc[data.date.isin(predict_dates)].copy()
        if train.empty or future.empty: break
        model=fit_forecast_model(train,feature_names,target,model_name=config.model_name,random_state=config.seed+start)
        future["prediction"]=model.predict(future)

        if residual_history:
            residuals=pd.concat(residual_history,ignore_index=True)
            cutoff=predict_dates[0]
            valid_dates=np.array(sorted(residuals.loc[residuals.date<cutoff,"date"].unique()))
            valid_dates=valid_dates[-config.scale_window_months:]
            residuals=residuals.loc[residuals.date.isin(valid_dates)]
        else:
            residuals=pd.DataFrame()
        if len(residuals)>=config.min_scale_observations:
            scale_model=fit_residual_scale_model(residuals,feature_names,"residual",random_state=config.seed+50_000+start)
            future["scale"]=predict_positive_scale(scale_model,future)
        else:
            fallback=float(np.nanmedian(np.abs(train[target]-model.predict(train))))
            future["scale"]=max(fallback,1e-4)

        out_cols=["date","asset_id",target,"prediction","scale"]
        for optional in ["market_cap","adv_dollars","daily_volatility"]:
            if optional in future: out_cols.append(optional)
        rows.append(future[out_cols])

        observed=future.dropna(subset=[target]).copy()
        observed["residual"]=observed[target]-observed.prediction
        residual_history.append(observed[["date","asset_id",*feature_names,"residual"]])
        start=end
    if not rows: return pd.DataFrame(columns=["date","asset_id",target,"prediction","scale"])
    return pd.concat(rows,ignore_index=True).sort_values(["date","asset_id"]).reset_index(drop=True)
