"""Shared helpers for episodic ranking certification backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .partial_order import (
    PairwiseCurve,
    pairwise_curve,
    proposal_grid_from_margins,
    sample_pair_margins,
)
from .portfolio import evaluate_portfolio_month_by_asset


@dataclass
class BacktestOutput:
    epochs: pd.DataFrame
    monthly: pd.DataFrame
    certification_curves: pd.DataFrame


def month_groups(panel: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> list[pd.DataFrame]:
    indexed = {pd.Timestamp(date): group for date, group in panel.groupby("date", sort=True)}
    return [indexed[pd.Timestamp(date)] for date in dates if pd.Timestamp(date) in indexed]


def candidate_grid(groups: list[pd.DataFrame], seed: int) -> np.ndarray:
    margins = [
        sample_pair_margins(
            group["prediction"].to_numpy(),
            group["scale"].to_numpy(),
            n_pairs=min(25_000, max(2_000, 20 * len(group))),
            seed=seed + offset,
        )
        for offset, group in enumerate(groups)
    ]
    if not margins:
        raise ValueError("proposal block contains no usable months")
    return proposal_grid_from_margins(np.concatenate(margins))


def loss_matrix(
    groups: list[pd.DataFrame],
    q_grid: np.ndarray,
    *,
    pair_universe: Literal["all", "tail"],
    tail_fraction: float,
) -> tuple[np.ndarray, list[PairwiseCurve]]:
    curves = [
        pairwise_curve(
            group["prediction"].to_numpy(),
            group["scale"].to_numpy(),
            group["ret_fwd"].to_numpy(),
            q_grid,
            universe=pair_universe,
            tail_fraction=tail_fraction,
        )
        for group in groups
    ]
    if not curves:
        raise ValueError("certification block contains no usable months")
    return np.vstack([curve.error_rate for curve in curves]), curves


def evaluate_method(
    asset_ids: np.ndarray,
    weights: np.ndarray,
    realized_returns: np.ndarray,
    previous_post_return: dict[object, float] | None,
    *,
    cost_bps: float,
):
    return evaluate_portfolio_month_by_asset(
        asset_ids,
        weights,
        realized_returns,
        previous_post_return,
        cost_bps=cost_bps,
    )
