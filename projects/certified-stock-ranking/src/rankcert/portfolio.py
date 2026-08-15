"""Portfolio maps induced by total and certified partial rankings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Mapping

import numpy as np

from .partial_order import dominance_scores, tail_dominance_scores, tail_indices


@dataclass(frozen=True)
class PortfolioReturn:
    """One-period portfolio result after formation-time trading costs."""

    gross_return: float
    turnover: float
    transaction_cost: float
    net_return: float


def _normalize_dollar_neutral(weights: np.ndarray, gross: float = 2.0) -> np.ndarray:
    if gross < 0.0:
        raise ValueError("gross must be nonnegative")
    w = np.asarray(weights, dtype=float).copy()
    w -= np.mean(w)
    l1 = float(np.sum(np.abs(w)))
    return np.zeros_like(w) if l1 <= np.finfo(float).eps else gross * w / l1


def raw_rank_weights(
    predicted: np.ndarray,
    *,
    top_fraction: float = 0.10,
    gross: float = 2.0,
    value_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Conventional equal- or value-weighted top-minus-bottom portfolio."""

    mu = np.asarray(predicted, dtype=float)
    if mu.ndim != 1 or mu.size < 2 or not np.all(np.isfinite(mu)):
        raise ValueError("predicted must be a finite vector")
    short_idx, long_idx = tail_indices(mu, top_fraction)
    base = np.ones_like(mu) if value_weights is None else np.asarray(value_weights, dtype=float)
    if base.shape != mu.shape or np.any(base < 0.0) or not np.all(np.isfinite(base)):
        raise ValueError("invalid value_weights")
    weights = np.zeros_like(mu)
    long_mass = float(base[long_idx].sum())
    short_mass = float(base[short_idx].sum())
    if long_mass > 0.0:
        weights[long_idx] = 0.5 * gross * base[long_idx] / long_mass
    if short_mass > 0.0:
        weights[short_idx] = -0.5 * gross * base[short_idx] / short_mass
    return weights


def relation_graph_degrees(
    predicted: np.ndarray,
    scale: np.ndarray,
    q: float,
    *,
    universe: Literal["all", "tail"] = "all",
    tail_fraction: float = 0.10,
    stock_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return relation-graph degrees for the selected pair universe."""

    if np.isinf(q):
        return np.zeros_like(np.asarray(predicted, dtype=float))
    if universe == "all":
        return dominance_scores(predicted, scale, q, weights=stock_weights)
    if universe == "tail":
        return tail_dominance_scores(
            predicted,
            scale,
            q,
            fraction=tail_fraction,
            weights=stock_weights,
        )
    raise ValueError("universe must be 'all' or 'tail'")


def certified_graph_weights(
    predicted: np.ndarray,
    scale: np.ndarray,
    q: float,
    *,
    gross: float = 2.0,
    universe: Literal["all", "tail"] = "all",
    tail_fraction: float = 0.10,
    stock_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Equal-gross portfolio obtained from certified relation-graph degrees."""

    degree = relation_graph_degrees(
        predicted,
        scale,
        q,
        universe=universe,
        tail_fraction=tail_fraction,
        stock_weights=stock_weights,
    )
    return _normalize_dollar_neutral(degree, gross)


def edge_notional_weights(
    predicted: np.ndarray,
    scale: np.ndarray,
    q: float,
    *,
    universe: Literal["all", "tail"] = "all",
    tail_fraction: float = 0.10,
    stock_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fixed-denominator edge portfolio for exact relation attribution.

    The denominator is independent of ``q``. Therefore the raw ``q=0`` edge
    portfolio equals the certified portfolio plus the removed-edge portfolio
    exactly before transaction costs and any equal-gross rescaling.
    """

    mu = np.asarray(predicted, dtype=float)
    weights = np.ones_like(mu) if stock_weights is None else np.asarray(stock_weights, dtype=float)
    if weights.shape != mu.shape or np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("invalid stock_weights")
    degree = relation_graph_degrees(
        mu,
        np.asarray(scale, dtype=float),
        q,
        universe=universe,
        tail_fraction=tail_fraction,
        stock_weights=weights,
    )
    if universe == "all":
        denominator = 0.5 * (float(weights.sum()) ** 2 - float(np.sum(weights * weights)))
    else:
        short_idx, long_idx = tail_indices(mu, tail_fraction)
        denominator = float(weights[short_idx].sum() * weights[long_idx].sum())
    return np.zeros_like(mu) if denominator <= 0.0 else degree / denominator


def one_way_turnover(current: np.ndarray, previous: np.ndarray | None = None) -> float:
    """Half-L1 turnover for already aligned vectors."""

    current_array = np.asarray(current, dtype=float)
    if previous is None:
        return 0.5 * float(np.sum(np.abs(current_array)))
    previous_array = np.asarray(previous, dtype=float)
    if previous_array.shape != current_array.shape:
        raise ValueError("previous and current weights must align")
    return 0.5 * float(np.sum(np.abs(current_array - previous_array)))


def turnover_from_mappings(
    current: Mapping[Hashable, float],
    previous_post_return: Mapping[Hashable, float] | None,
) -> float:
    """Half-L1 turnover over the union of changing stock universes."""

    previous = {} if previous_post_return is None else previous_post_return
    assets = set(current).union(previous)
    return 0.5 * float(
        sum(abs(float(current.get(asset, 0.0)) - float(previous.get(asset, 0.0))) for asset in assets)
    )


def linear_transaction_cost(turnover: float, cost_bps: float) -> float:
    if turnover < 0.0 or cost_bps < 0.0:
        raise ValueError("turnover and cost_bps must be nonnegative")
    return turnover * cost_bps / 10_000.0


def square_root_transaction_cost(
    trade_weights: np.ndarray,
    *,
    aum: float,
    adv_dollars: np.ndarray,
    daily_volatility: np.ndarray,
    half_spread_bps: np.ndarray | float = 5.0,
    impact_coefficient: float = 0.10,
) -> float:
    """Transparent spread-plus-square-root-impact cost model."""

    delta = np.abs(np.asarray(trade_weights, dtype=float))
    adv = np.asarray(adv_dollars, dtype=float)
    volatility = np.asarray(daily_volatility, dtype=float)
    spread = np.broadcast_to(np.asarray(half_spread_bps, dtype=float), delta.shape)
    if (
        delta.shape != adv.shape
        or delta.shape != volatility.shape
        or aum <= 0.0
        or np.any(adv <= 0.0)
        or np.any(volatility < 0.0)
        or impact_coefficient < 0.0
    ):
        raise ValueError("invalid liquidity inputs")
    participation = np.maximum(aum * delta / adv, 0.0)
    return float(
        np.sum(
            spread / 10_000.0 * delta
            + impact_coefficient * volatility * np.sqrt(participation) * delta
        )
    )


def drift_weights(
    asset_ids: np.ndarray,
    weights: np.ndarray,
    realized_returns: np.ndarray,
) -> dict[Hashable, float]:
    """Return post-return risky-asset weights used by the next rebalance.

    Cash is implicit. The denominator is portfolio wealth before the current
    rebalance's transaction cost, matching the standard passive-drift turnover
    convention. Securities that disappear next month remain in this mapping and
    are therefore liquidated explicitly by ``turnover_from_mappings``.
    """

    ids = np.asarray(asset_ids)
    w = np.asarray(weights, dtype=float)
    returns = np.asarray(realized_returns, dtype=float)
    if ids.ndim != 1 or ids.size != w.size or w.shape != returns.shape:
        raise ValueError("asset_ids, weights, and returns must align")
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("asset_ids must be unique within a formation month")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(returns)):
        raise ValueError("weights and returns must be finite")
    gross_return = float(np.dot(w, returns))
    denominator = 1.0 + gross_return
    if denominator <= np.finfo(float).eps:
        raise ValueError("portfolio wealth is nonpositive after realized returns")
    post = w * (1.0 + returns) / denominator
    return {
        asset_id.item() if hasattr(asset_id, "item") else asset_id: float(weight)
        for asset_id, weight in zip(ids, post, strict=True)
        if abs(float(weight)) > 1e-15
    }


def evaluate_portfolio_month(
    weights: np.ndarray,
    realized_returns: np.ndarray,
    previous_weights: np.ndarray | None = None,
    *,
    cost_bps: float,
) -> PortfolioReturn:
    """Evaluate one month for already aligned, fixed-universe vectors."""

    w = np.asarray(weights, dtype=float)
    returns = np.asarray(realized_returns, dtype=float)
    if w.shape != returns.shape:
        raise ValueError("weights and returns must align")
    gross_return = float(np.dot(w, returns))
    turnover = one_way_turnover(w, previous_weights)
    cost = linear_transaction_cost(turnover, cost_bps)
    return PortfolioReturn(gross_return, turnover, cost, gross_return - cost)


def evaluate_portfolio_month_by_asset(
    asset_ids: np.ndarray,
    weights: np.ndarray,
    realized_returns: np.ndarray,
    previous_post_return_weights: Mapping[Hashable, float] | None,
    *,
    cost_bps: float,
) -> tuple[PortfolioReturn, dict[Hashable, float]]:
    """Evaluate one month with universe changes and passive weight drift."""

    ids = np.asarray(asset_ids)
    w = np.asarray(weights, dtype=float)
    returns = np.asarray(realized_returns, dtype=float)
    if ids.ndim != 1 or ids.size != w.size or w.shape != returns.shape:
        raise ValueError("asset_ids, weights, and returns must align")
    current = {
        asset_id.item() if hasattr(asset_id, "item") else asset_id: float(weight)
        for asset_id, weight in zip(ids, w, strict=True)
        if abs(float(weight)) > 1e-15
    }
    turnover = turnover_from_mappings(current, previous_post_return_weights)
    gross_return = float(np.dot(w, returns))
    cost = linear_transaction_cost(turnover, cost_bps)
    post_return = drift_weights(ids, w, returns)
    return PortfolioReturn(gross_return, turnover, cost, gross_return - cost), post_return
