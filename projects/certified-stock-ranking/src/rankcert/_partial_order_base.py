"""Nested partial orders induced by machine-learned return forecasts.

The module exposes exact all-pair and top-versus-bottom pairwise loss curves.
Both implementations run in :math:`O(KN\\log N)` for ``K`` thresholds by using
Fenwick trees rather than materializing an :math:`N\times N` relation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


@dataclass(frozen=True)
class PairwiseCurve:
    """Pairwise ranking diagnostics over an ordered threshold grid."""

    q: np.ndarray
    active_weight: np.ndarray
    error_weight: np.ndarray
    error_rate: np.ndarray
    wrong_bet_mass: np.ndarray
    breadth: np.ndarray


def _validate_vectors(
    predicted: np.ndarray,
    scale: np.ndarray,
    realized: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    mu = np.asarray(predicted, dtype=float)
    s = np.asarray(scale, dtype=float)
    if mu.ndim != 1 or s.ndim != 1 or mu.size != s.size or mu.size < 2:
        raise ValueError("predicted and scale must be same-length vectors")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("forecasts must be finite and scales strictly positive")
    y = None if realized is None else np.asarray(realized, dtype=float)
    if y is not None and (y.shape != mu.shape or not np.all(np.isfinite(y))):
        raise ValueError("realized must be finite and match predicted")
    w = np.ones_like(mu) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != mu.shape or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite, nonnegative, and match predicted")
    if np.sum(w > 0.0) < 2:
        raise ValueError("at least two assets must receive positive weight")
    return mu, s, y, w


def _validate_q_grid(q_grid: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(q_grid), dtype=float)
    if (
        q.ndim != 1
        or q.size == 0
        or np.any(~np.isfinite(q))
        or np.any(q < 0.0)
        or np.any(np.diff(q) < 0.0)
    ):
        raise ValueError("q_grid must be a finite, nonnegative, increasing vector")
    return q


def score_bands(
    predicted: np.ndarray,
    scale: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper score bands at threshold ``q``."""

    mu, s, _, _ = _validate_vectors(predicted, scale)
    if q < 0.0 or not np.isfinite(q):
        raise ValueError("q must be finite and nonnegative")
    return mu - q * s, mu + q * s


def relation_matrix(predicted: np.ndarray, scale: np.ndarray, q: float) -> np.ndarray:
    """Materialize the strict partial-order matrix for small universes/tests."""

    lower, upper = score_bands(predicted, scale, q)
    relation = lower[:, None] > upper[None, :]
    np.fill_diagonal(relation, False)
    return relation


def check_strict_partial_order(relation: np.ndarray) -> bool:
    """Check irreflexivity and transitivity of a Boolean relation."""

    r = np.asarray(relation, dtype=bool)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("relation must be square")
    if np.any(np.diag(r)):
        return False
    two_step = (r.astype(np.int64) @ r.astype(np.int64)) > 0
    return bool(np.all(~two_step | r))


def tail_indices(predicted: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint raw bottom- and top-tail indices."""

    mu = np.asarray(predicted, dtype=float)
    if mu.ndim != 1 or mu.size < 2 or not np.all(np.isfinite(mu)):
        raise ValueError("predicted must be a finite vector")
    if not 0.0 < fraction <= 0.5:
        raise ValueError("fraction must lie in (0, 0.5]")
    n_tail = max(1, int(np.floor(fraction * mu.size)))
    order = np.argsort(mu, kind="mergesort")
    return order[:n_tail], order[-n_tail:]


def dominance_scores(
    predicted: np.ndarray,
    scale: np.ndarray,
    q: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted out-degree minus in-degree for the all-pair relation graph."""

    mu, s, _, w = _validate_vectors(predicted, scale, weights=weights)
    lower, upper = mu - q * s, mu + q * s

    upper_order = np.argsort(upper, kind="mergesort")
    sorted_upper = upper[upper_order]
    cumulative_upper_weights = np.concatenate(([0.0], np.cumsum(w[upper_order])))
    out_weight = cumulative_upper_weights[
        np.searchsorted(sorted_upper, lower, side="left")
    ]

    lower_order = np.argsort(lower, kind="mergesort")
    sorted_lower = lower[lower_order]
    cumulative_lower_weights = np.concatenate(([0.0], np.cumsum(w[lower_order])))
    n_at_most = np.searchsorted(sorted_lower, upper, side="right")
    in_weight = cumulative_lower_weights[-1] - cumulative_lower_weights[n_at_most]

    return w * (out_weight - in_weight)


def tail_dominance_scores(
    predicted: np.ndarray,
    scale: np.ndarray,
    q: float,
    *,
    fraction: float = 0.10,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Graph degrees using only raw top-versus-bottom relations."""

    mu, s, _, w = _validate_vectors(predicted, scale, weights=weights)
    short_idx, long_idx = tail_indices(mu, fraction)
    lower_long = mu[long_idx] - q * s[long_idx]
    upper_short = mu[short_idx] + q * s[short_idx]

    short_order = np.argsort(upper_short, kind="mergesort")
    sorted_upper_short = upper_short[short_order]
    short_weights_sorted = w[short_idx][short_order]
    cumulative_short = np.concatenate(([0.0], np.cumsum(short_weights_sorted)))
    long_out = cumulative_short[
        np.searchsorted(sorted_upper_short, lower_long, side="left")
    ]

    long_order = np.argsort(lower_long, kind="mergesort")
    sorted_lower_long = lower_long[long_order]
    long_weights_sorted = w[long_idx][long_order]
    cumulative_long = np.concatenate(([0.0], np.cumsum(long_weights_sorted)))
    n_at_most = np.searchsorted(sorted_lower_long, upper_short, side="right")
    short_in = cumulative_long[-1] - cumulative_long[n_at_most]

    degree = np.zeros_like(mu)
    degree[long_idx] = w[long_idx] * long_out
    degree[short_idx] = -w[short_idx] * short_in
    return degree


def reliable_breadth(
    predicted: np.ndarray,
    scale: np.ndarray,
    q: float,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted fraction of all raw rank relations retained at threshold ``q``."""

    mu, s, _, w = _validate_vectors(predicted, scale, weights=weights)
    lower, upper = mu - q * s, mu + q * s
    order = np.argsort(upper, kind="mergesort")
    sorted_upper = upper[order]
    cumulative_weights = np.concatenate(([0.0], np.cumsum(w[order])))
    active = float(
        np.sum(w * cumulative_weights[np.searchsorted(sorted_upper, lower, side="left")])
    )
    total = 0.5 * (float(w.sum()) ** 2 - float(np.sum(w * w)))
    return 0.0 if total <= 0.0 else active / total


def tail_reliable_breadth(
    predicted: np.ndarray,
    scale: np.ndarray,
    q: float,
    *,
    fraction: float = 0.10,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted fraction of raw top-versus-bottom relations retained."""

    mu, s, _, w = _validate_vectors(predicted, scale, weights=weights)
    short_idx, long_idx = tail_indices(mu, fraction)
    lower_long = mu[long_idx] - q * s[long_idx]
    upper_short = mu[short_idx] + q * s[short_idx]
    order = np.argsort(upper_short, kind="mergesort")
    sorted_upper = upper_short[order]
    cumulative_short = np.concatenate(([0.0], np.cumsum(w[short_idx][order])))
    active = float(
        np.sum(
            w[long_idx]
            * cumulative_short[np.searchsorted(sorted_upper, lower_long, side="left")]
        )
    )
    total = float(w[long_idx].sum() * w[short_idx].sum())
    return 0.0 if total <= 0.0 else active / total


