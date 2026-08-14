"""Evaluation metrics for prediction intervals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def interval_metrics(
    y: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    alpha: float,
) -> dict[str, float]:
    """Compute coverage, width, and the Winkler interval score."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    y_array = np.asarray(y, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if y_array.shape != lower_array.shape or y_array.shape != upper_array.shape:
        raise ValueError("y, lower, and upper must have the same shape")
    if np.any(lower_array > upper_array):
        raise ValueError("lower bounds cannot exceed upper bounds")
    covered = (y_array >= lower_array) & (y_array <= upper_array)
    width = upper_array - lower_array
    score = (
        width
        + (2.0 / alpha) * (lower_array - y_array) * (y_array < lower_array)
        + (2.0 / alpha) * (y_array - upper_array) * (y_array > upper_array)
    )
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "interval_score": float(np.mean(score)),
    }
