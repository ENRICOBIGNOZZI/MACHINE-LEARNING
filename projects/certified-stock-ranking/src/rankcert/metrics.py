"""Financial and ranking diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd


def annualized_summary(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    x = pd.Series(returns, dtype=float).dropna()
    if x.empty:
        return {k: float("nan") for k in ["mean", "volatility", "sharpe", "max_drawdown"]}
    mean = float(x.mean() * periods_per_year)
    vol = float(x.std(ddof=1) * np.sqrt(periods_per_year)) if x.size > 1 else float("nan")
    wealth = (1 + x).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {"mean": mean, "volatility": vol, "sharpe": mean / vol if vol > 0 else float("nan"), "max_drawdown": float(drawdown.min())}


def certainty_equivalent(returns: pd.Series, risk_aversion: float = 5.0, periods_per_year: int = 12) -> float:
    x = pd.Series(returns, dtype=float).dropna()
    return float("nan") if x.empty else float(periods_per_year * (x.mean() - .5 * risk_aversion * x.var(ddof=1)))


def newey_west_tstat(series: pd.Series, max_lags: int = 6) -> float:
    x = pd.Series(series, dtype=float).dropna().to_numpy()
    n = x.size
    if n < 3: return float("nan")
    centered = x - x.mean()
    long_run = float(centered @ centered / n)
    for lag in range(1, min(max_lags, n - 1) + 1):
        long_run += 2 * (1 - lag / (min(max_lags, n - 1) + 1)) * float(centered[lag:] @ centered[:-lag] / n)
    se = np.sqrt(max(long_run, 0) / n)
    return float(x.mean() / se) if se > 0 else float("nan")
