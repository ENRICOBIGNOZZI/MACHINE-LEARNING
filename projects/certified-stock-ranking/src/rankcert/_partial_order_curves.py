"""Exact pairwise ranking curves and proposal-grid helpers."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
from numba import njit

from ._partial_order_base import (
    PairwiseCurve,
    _validate_q_grid,
    _validate_vectors,
    tail_indices,
)

@njit(cache=True)
def _fenwick_add(tree: np.ndarray, index: int, value: float) -> None:
    i = index + 1
    while i < tree.shape[0]:
        tree[i] += value
        i += i & -i


@njit(cache=True)
def _fenwick_sum(tree: np.ndarray, index: int) -> float:
    if index < 0:
        return 0.0
    total = 0.0
    i = index + 1
    while i > 0:
        total += tree[i]
        i -= i & -i
    return total


@njit(cache=True)
def _compressed_ranks(values: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.int64)
    rank = 0
    ranks[order[0]] = rank
    for p in range(1, values.shape[0]):
        if values[order[p]] > values[order[p - 1]]:
            rank += 1
        ranks[order[p]] = rank
    return ranks, rank + 1


@njit(cache=True)
def _pair_counts_one_q(
    predicted: np.ndarray,
    scale: np.ndarray,
    realized: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> tuple[float, float]:
    n_assets = predicted.shape[0]
    lower = predicted - q * scale
    upper = predicted + q * scale
    upper_order = np.argsort(upper)
    lower_order = np.argsort(lower)
    ranks, n_ranks = _compressed_ranks(realized)

    tree = np.zeros(n_ranks + 1, dtype=np.float64)
    inserted_weight = 0.0
    active_weight = 0.0
    error_weight = 0.0
    cursor = 0
    for position in range(n_assets):
        i = lower_order[position]
        while cursor < n_assets and upper[upper_order[cursor]] < lower[i]:
            j = upper_order[cursor]
            _fenwick_add(tree, ranks[j], weights[j])
            inserted_weight += weights[j]
            cursor += 1
        active_weight += weights[i] * inserted_weight
        strictly_lower_realized_weight = _fenwick_sum(tree, ranks[i] - 1)
        # A claimed i > j is wrong when realized_i <= realized_j.
        error_weight += weights[i] * (inserted_weight - strictly_lower_realized_weight)
    return active_weight, error_weight


@njit(cache=True)
def _tail_pair_counts_one_q(
    predicted: np.ndarray,
    scale: np.ndarray,
    realized: np.ndarray,
    weights: np.ndarray,
    short_idx: np.ndarray,
    long_idx: np.ndarray,
    q: float,
) -> tuple[float, float]:
    n_short = short_idx.shape[0]
    n_long = long_idx.shape[0]
    upper_short = predicted[short_idx] + q * scale[short_idx]
    lower_long = predicted[long_idx] - q * scale[long_idx]
    short_order = np.argsort(upper_short)
    long_order = np.argsort(lower_long)

    selected_realized = np.empty(n_short + n_long, dtype=np.float64)
    selected_realized[:n_short] = realized[short_idx]
    selected_realized[n_short:] = realized[long_idx]
    selected_ranks, n_ranks = _compressed_ranks(selected_realized)
    short_ranks = selected_ranks[:n_short]
    long_ranks = selected_ranks[n_short:]

    tree = np.zeros(n_ranks + 1, dtype=np.float64)
    inserted_weight = 0.0
    active_weight = 0.0
    error_weight = 0.0
    cursor = 0
    for position in range(n_long):
        local_long = long_order[position]
        lower_value = lower_long[local_long]
        while cursor < n_short and upper_short[short_order[cursor]] < lower_value:
            local_short = short_order[cursor]
            j = short_idx[local_short]
            _fenwick_add(tree, short_ranks[local_short], weights[j])
            inserted_weight += weights[j]
            cursor += 1
        i = long_idx[local_long]
        active_weight += weights[i] * inserted_weight
        strictly_lower_realized_weight = _fenwick_sum(tree, long_ranks[local_long] - 1)
        error_weight += weights[i] * (inserted_weight - strictly_lower_realized_weight)
    return active_weight, error_weight


def pairwise_curve_all_pairs(
    predicted: np.ndarray,
    scale: np.ndarray,
    realized: np.ndarray,
    q_grid: Iterable[float],
    weights: np.ndarray | None = None,
) -> PairwiseCurve:
    """Exact all-pair ranking loss and breadth curves."""

    mu, s, y, w = _validate_vectors(predicted, scale, realized, weights)
    assert y is not None
    q = _validate_q_grid(q_grid)
    active = np.empty(q.size, dtype=float)
    errors = np.empty(q.size, dtype=float)
    for k, q_k in enumerate(q):
        active[k], errors[k] = _pair_counts_one_q(mu, s, y, w, float(q_k))
    total = 0.5 * (float(w.sum()) ** 2 - float(np.sum(w * w)))
    error_rate = np.divide(errors, active, out=np.zeros_like(errors), where=active > 0.0)
    breadth = active / total if total > 0.0 else np.zeros_like(active)
    wrong_bet_mass = errors / total if total > 0.0 else np.zeros_like(errors)
    return PairwiseCurve(q, active, errors, error_rate, wrong_bet_mass, breadth)


def pairwise_curve_tail_pairs(
    predicted: np.ndarray,
    scale: np.ndarray,
    realized: np.ndarray,
    q_grid: Iterable[float],
    *,
    fraction: float = 0.10,
    weights: np.ndarray | None = None,
) -> PairwiseCurve:
    """Exact raw top-versus-bottom ranking loss and breadth curves."""

    mu, s, y, w = _validate_vectors(predicted, scale, realized, weights)
    assert y is not None
    q = _validate_q_grid(q_grid)
    short_idx, long_idx = tail_indices(mu, fraction)
    short_idx = short_idx.astype(np.int64)
    long_idx = long_idx.astype(np.int64)
    active = np.empty(q.size, dtype=float)
    errors = np.empty(q.size, dtype=float)
    for k, q_k in enumerate(q):
        active[k], errors[k] = _tail_pair_counts_one_q(
            mu, s, y, w, short_idx, long_idx, float(q_k)
        )
    total = float(w[long_idx].sum() * w[short_idx].sum())
    error_rate = np.divide(errors, active, out=np.zeros_like(errors), where=active > 0.0)
    breadth = active / total if total > 0.0 else np.zeros_like(active)
    wrong_bet_mass = errors / total if total > 0.0 else np.zeros_like(errors)
    return PairwiseCurve(q, active, errors, error_rate, wrong_bet_mass, breadth)


def pairwise_curve(
    predicted: np.ndarray,
    scale: np.ndarray,
    realized: np.ndarray,
    q_grid: Iterable[float],
    *,
    universe: Literal["all", "tail"] = "all",
    tail_fraction: float = 0.10,
    weights: np.ndarray | None = None,
) -> PairwiseCurve:
    """Dispatch to the all-pair or top-versus-bottom curve."""

    if universe == "all":
        return pairwise_curve_all_pairs(predicted, scale, realized, q_grid, weights)
    if universe == "tail":
        return pairwise_curve_tail_pairs(
            predicted,
            scale,
            realized,
            q_grid,
            fraction=tail_fraction,
            weights=weights,
        )
    raise ValueError("universe must be 'all' or 'tail'")


def sample_pair_margins(
    predicted: np.ndarray,
    scale: np.ndarray,
    *,
    n_pairs: int,
    seed: int,
) -> np.ndarray:
    """Sample forecast-separation margins before outcomes are observed."""

    mu, s, _, _ = _validate_vectors(predicted, scale)
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    rng = np.random.default_rng(seed)
    n_assets = mu.size
    i = rng.integers(0, n_assets, size=n_pairs)
    j = rng.integers(0, n_assets - 1, size=n_pairs)
    j = j + (j >= i)
    return np.abs(mu[i] - mu[j]) / (s[i] + s[j])


def proposal_grid_from_margins(
    margins: np.ndarray,
    target_breadths: Iterable[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
) -> np.ndarray:
    """Map proposal-period pair margins to a nested threshold library."""

    valid_margins = np.asarray(margins, dtype=float)
    valid_margins = valid_margins[
        np.isfinite(valid_margins) & (valid_margins >= 0.0)
    ]
    breadths = np.asarray(list(target_breadths), dtype=float)
    if valid_margins.size == 0 or np.any((breadths <= 0.0) | (breadths > 1.0)):
        raise ValueError("invalid proposal margins or target breadths")
    q = np.quantile(valid_margins, np.clip(1.0 - breadths, 0.0, 1.0))
    q = np.unique(np.concatenate(([0.0], q)))
    q.sort()
    return q
