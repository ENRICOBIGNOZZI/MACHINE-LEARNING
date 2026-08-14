"""Brownian self-normalized critical values and simulation utilities."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def load_critical_value_records() -> list[dict[str, float | int | str]]:
    """Load the critical-value records shipped with the replication package."""

    path = files("snrcps.data").joinpath("self_normalized_criticals.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["records"])


def critical_value(
    *,
    confidence: float | None = None,
    delta: float | None = None,
    name: str | None = None,
    tolerance: float = 5e-10,
) -> float:
    """Return a stored Brownian critical value.

    Select by record ``name`` or by exactly one of ``confidence`` and ``delta``.
    The stored values are Monte Carlo approximations; the record metadata expose
    path count, grid size, and seed.
    """

    selectors = sum(value is not None for value in (confidence, delta, name))
    if selectors != 1:
        raise ValueError("provide exactly one of confidence, delta, or name")
    target = 1.0 - float(delta) if delta is not None else confidence
    records = load_critical_value_records()
    if name is not None:
        matches = [record for record in records if record["name"] == name]
    else:
        if target is None or not 0 < target < 1:
            raise ValueError("confidence must lie strictly between zero and one")
        matches = [
            record
            for record in records
            if abs(float(record["confidence"]) - float(target)) <= tolerance
            and record.get("recommended", True)
        ]
    if not matches:
        raise KeyError("no stored critical value matches the requested selector")
    if len(matches) > 1:
        raise KeyError("multiple records match; select by name")
    return float(matches[0]["critical_value"])


def simulate_critical_values(
    confidences: Iterable[float],
    *,
    paths: int = 100_000,
    grid_size: int = 800,
    batch_size: int = 2_000,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Simulate quantiles of ``B(1) / ||B(r)-rB(1)||_L2``.

    This function is intended for reproducibility and sensitivity checks. Large
    path counts are needed for extreme confidence levels.
    """

    probabilities = np.asarray(list(confidences), dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("confidences must be a nonempty one-dimensional collection")
    if np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("confidences must lie strictly between zero and one")
    if paths < 1 or grid_size < 2 or batch_size < 1:
        raise ValueError("paths, grid_size, and batch_size must be positive")

    rng = np.random.default_rng(seed)
    time_grid = np.arange(1, grid_size + 1, dtype=float) / grid_size
    values = np.empty(paths, dtype=float)
    position = 0
    while position < paths:
        size = min(batch_size, paths - position)
        increments = rng.normal(scale=grid_size ** (-0.5), size=(size, grid_size))
        brownian = np.cumsum(increments, axis=1)
        endpoint = brownian[:, -1]
        bridge = brownian - time_grid[None, :] * endpoint[:, None]
        denominator = np.sqrt(np.mean(bridge * bridge, axis=1))
        values[position : position + size] = endpoint / denominator
        position += size
    return np.quantile(values, probabilities, method="higher").astype(float)
