"""Self-normalized risk certification for ordered candidate libraries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CertificationResult:
    selected_index: int | None
    selected_value: float
    empirical_risk: np.ndarray
    upper_bound: np.ndarray
    self_normalizer: np.ndarray
    radius: np.ndarray
    critical_value: float
    alpha: float
    delta: float
    fallback: bool


def monotone_envelope(losses: np.ndarray) -> np.ndarray:
    """Return out[t,k] = max(losses[t,k:]), the least right-monotone majorant."""
    arr = np.asarray(losses, dtype=float)
    if arr.ndim != 2 or min(arr.shape) == 0:
        raise ValueError("losses must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(arr)) or arr.min() < -1e-12 or arr.max() > 1 + 1e-12:
        raise ValueError("losses must be finite and lie in [0,1]")
    arr = np.clip(arr, 0.0, 1.0)
    return np.maximum.accumulate(arr[:, ::-1], axis=1)[:, ::-1]


def self_normalizer(losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(losses, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("at least two time periods are required")
    m = arr.shape[0]
    means = arr.mean(axis=0)
    recursive = np.cumsum(arr - means[None, :], axis=0)
    vhat = np.sum(recursive * recursive, axis=0) / float(m * m)
    return means, vhat


def brownian_pivot_draws(
    n_paths: int = 100_000,
    n_grid: int = 1_500,
    seed: int = 20260814,
    chunk_size: int = 2_000,
) -> np.ndarray:
    if n_paths <= 0 or n_grid < 10 or chunk_size <= 0:
        raise ValueError("invalid Monte Carlo dimensions")
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_grid
    r = np.arange(1, n_grid + 1, dtype=float) / n_grid
    draws: list[np.ndarray] = []
    left = n_paths
    while left:
        n = min(chunk_size, left)
        increments = rng.normal(0.0, np.sqrt(dt), size=(n, n_grid))
        brownian = np.cumsum(increments, axis=1)
        endpoint = brownian[:, -1]
        bridge = brownian - endpoint[:, None] * r[None, :]
        denom = np.sqrt(np.mean(bridge * bridge, axis=1))
        draws.append(endpoint[denom > np.finfo(float).eps] / denom[denom > np.finfo(float).eps])
        left -= n
    return np.concatenate(draws)


@lru_cache(maxsize=32)
def _critical(delta: float, n_paths: int, n_grid: int, seed: int) -> float:
    return float(np.quantile(brownian_pivot_draws(n_paths, n_grid, seed), 1.0 - delta))


def brownian_critical_value(
    delta: float,
    *,
    n_paths: int = 100_000,
    n_grid: int = 1_500,
    seed: int = 20260814,
) -> float:
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    return _critical(float(delta), int(n_paths), int(n_grid), int(seed))


def ordered_selector(upper_bounds: Iterable[float], alpha: float) -> int | None:
    u = np.asarray(list(upper_bounds), dtype=float)
    if u.ndim != 1 or u.size == 0 or not 0 < alpha < 1:
        raise ValueError("invalid ordered-selector arguments")
    suffix_max = np.maximum.accumulate(u[::-1])[::-1]
    passing = np.flatnonzero(suffix_max <= alpha)
    return int(passing[0]) if passing.size else None


def certify_monotone_losses(
    losses: np.ndarray,
    candidate_values: Iterable[float],
    *,
    alpha: float,
    delta: float,
    critical_value: float | None = None,
    mc_paths: int = 100_000,
    mc_grid: int = 1_500,
    mc_seed: int = 20260814,
    enforce_monotone: bool = True,
    zero_denominator_policy: str = "paper",
) -> CertificationResult:
    values = np.asarray(list(candidate_values), dtype=float)
    arr = np.asarray(losses, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != values.size:
        raise ValueError("candidate_values must match loss columns")
    if np.any(np.diff(values) < 0):
        raise ValueError("candidate_values must be increasing")
    if enforce_monotone:
        arr = monotone_envelope(arr)
    elif np.any(np.diff(arr, axis=1) > 1e-10):
        raise ValueError("loss rows are not nonincreasing")
    means, vhat = self_normalizer(arr)
    m = arr.shape[0]
    c = brownian_critical_value(delta, n_paths=mc_paths, n_grid=mc_grid, seed=mc_seed) \
        if critical_value is None else float(critical_value)
    radius = c * np.sqrt(vhat / m)
    upper = np.minimum(1.0, means + radius)
    if zero_denominator_policy == "conservative":
        upper = np.where(vhat <= np.finfo(float).eps, 1.0, upper)
    elif zero_denominator_policy != "paper":
        raise ValueError("zero_denominator_policy must be 'paper' or 'conservative'")
    selected = ordered_selector(upper, alpha)
    return CertificationResult(
        selected_index=selected,
        selected_value=float("inf") if selected is None else float(values[selected]),
        empirical_risk=means,
        upper_bound=upper,
        self_normalizer=vhat,
        radius=radius,
        critical_value=c,
        alpha=float(alpha),
        delta=float(delta),
        fallback=selected is None,
    )


def polynomial_confidence_spending(delta: float, n_epochs: int) -> np.ndarray:
    if not 0 < delta < 1 or n_epochs <= 0:
        raise ValueError("invalid confidence-spending arguments")
    j = np.arange(1, n_epochs + 1, dtype=float)
    weights = 1.0 / (j * j)
    return delta * weights / weights.sum()
