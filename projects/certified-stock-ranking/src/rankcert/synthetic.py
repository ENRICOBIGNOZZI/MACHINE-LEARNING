"""Synthetic monthly stock panel used only for software validation.

The design intentionally creates a visible ranking signal so that the tests exercise
nontrivial abstention. It is not calibrated to the magnitude or Sharpe ratio of U.S.
equity strategies.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def generate_synthetic_panel(*, n_months: int = 240, n_assets: int = 250, n_features: int = 12, seed: int = 20260814) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    dates=pd.date_range("2000-01-31",periods=n_months,freq="ME")
    asset_ids=np.arange(10_000,10_000+n_assets)
    beta=rng.normal(0,.006,size=n_features)
    asset_loading=rng.normal(1,.25,size=n_assets)
    latent_x=rng.normal(size=(n_assets,n_features))
    common_forecast_error=0.; stress=0.; rows=[]
    for t,date in enumerate(dates):
        stress=.88*stress+rng.normal(0,.45)
        if t in {95,96,97,170,171,172,173}: stress+=2.0
        latent_x=.82*latent_x+rng.normal(0,.65,size=latent_x.shape)
        nonlinear=.003*np.tanh(latent_x[:,0]*latent_x[:,1])
        true_mu=latent_x@beta+nonlinear
        idio_scale=.025+.010*np.abs(latent_x[:,2])+.006*max(stress,0)
        common_forecast_error=.75*common_forecast_error+rng.normal(0,.0025+.0015*max(stress,0))
        forecast_noise=rng.normal(0,idio_scale*(.25+.12*max(stress,0)))
        prediction=true_mu+common_forecast_error*asset_loading+forecast_noise
        scale=np.maximum(idio_scale*(.55+.18*max(stress,0)),.005)
        market_shock=rng.normal(0,.025+.012*max(stress,0))
        realized=true_mu+asset_loading*market_shock+rng.normal(0,idio_scale)
        mcap=np.exp(8+.8*latent_x[:,3])
        frame=pd.DataFrame({"date":date,"asset_id":asset_ids,"prediction":prediction,"scale":scale,"ret_fwd":realized,"market_cap":mcap,"stress":stress})
        for j in range(n_features): frame[f"x{j+1}"]=latent_x[:,j]
        rows.append(frame)
    return pd.concat(rows,ignore_index=True)
