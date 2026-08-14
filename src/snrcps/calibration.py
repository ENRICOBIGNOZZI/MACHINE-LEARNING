"""Core self-normalized risk certification for nested prediction sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_finite_vector(values: ArrayLike, *, name: str, minimum_size: int = 1) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} value(s)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float64, copy=False)


def candidate_thresholds(
    proposal_scores: ArrayLike,
    levels: Iterable[float],
    *,
    safe_threshold: float | None = None,
) -> FloatArray:
    """Construct an ordered candidate library from proposal-score quantiles.

    Parameters
    ----------
    proposal_scores:
        Nonnegative proposal scores.
    levels:
        Quantile levels in the open interval ``(0, 1)``.
    safe_threshold:
        Optional deterministic final candidate. It must be no smaller than the
        largest proposal quantile.
    """

    scores = _as_finite_vector(proposal_scores, name="proposal_scores")
    if np.any(scores < 0):
        raise ValueError("proposal_scores must be nonnegative")
    level_array = _as_finite_vector(list(levels), name="levels")
    if np.any((level_array <= 0) | (level_array >= 1)):
        raise ValueError("levels must lie strictly between zero and one")
    level_array = np.unique(level_array)
    thresholds = np.array(
        [np.quantile(scores, level, method="higher") for level in level_array],
        dtype=float,
    )
    thresholds = np.unique(thresholds)
    if safe_threshold is not None:
        safe = float(safe_threshold)
        if not np.isfinite(safe):
            raise ValueError("safe_threshold must be finite")
        if safe < thresholds[-1]:
            raise ValueError("safe_threshold must be at least the largest proposal quantile")
        thresholds = np.unique(np.append(thresholds, safe))
    return thresholds


def loss_matrix(scores: ArrayLike, thresholds: ArrayLike) -> FloatArray:
    """Return binary losses ``1{score > threshold}`` for ordered candidates."""

    score_array = _as_finite_vector(scores, name="scores", minimum_size=2)
    threshold_array = _as_finite_vector(thresholds, name="thresholds")
    if np.any(np.diff(threshold_array) < 0):
        raise ValueError("thresholds must be ordered from narrowest to widest")
    return (score_array[:, None] > threshold_array[None, :]).astype(np.float64)


def self_normalized_statistics(losses: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Compute empirical risks and recursive self-normalizers.

    If ``L`` has ``m`` rows, the returned normalizer for each candidate is

    ``m^{-2} sum_{t=1}^m [sum_{s=1}^t (L_s - L_bar)]^2``.
    """

    array = np.asarray(losses, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("losses must be one- or two-dimensional")
    if array.shape[0] < 2:
        raise ValueError("losses must contain at least two observations")
    if not np.all(np.isfinite(array)):
        raise ValueError("losses must contain only finite values")
    empirical_risk = array.mean(axis=0)
    partial_sums = np.cumsum(array - empirical_risk[None, :], axis=0)
    sample_size = array.shape[0]
    normalizer = np.sum(partial_sums * partial_sums, axis=0) / (sample_size * sample_size)
    return empirical_risk.astype(float), normalizer.astype(float)


def ordered_upper_envelope(pointwise_upper: ArrayLike) -> FloatArray:
    """Enforce the RCPS suffix condition for an ordered nested library."""

    upper = _as_finite_vector(pointwise_upper, name="pointwise_upper")
    return np.maximum.accumulate(upper[::-1])[::-1]


def ordered_select(pointwise_upper: ArrayLike, risk_target: float) -> int | None:
    """Select the narrowest candidate whose complete wider suffix is certified."""

    if not 0 < risk_target < 1:
        raise ValueError("risk_target must lie strictly between zero and one")
    envelope = ordered_upper_envelope(pointwise_upper)
    feasible = np.flatnonzero(envelope <= risk_target)
    return int(feasible[0]) if feasible.size else None


@dataclass(frozen=True)
class CertificationResult:
    """Complete output of one SN-RCPS certification call."""

    thresholds: FloatArray
    empirical_risk: FloatArray
    normalizer: FloatArray
    radius: FloatArray
    pointwise_upper: FloatArray
    ordered_upper: FloatArray
    selected_index: int | None
    selected_threshold: float
    fallback_used: bool
    risk_target: float
    critical_value: float

    @property
    def selected_candidate(self) -> int | None:
        """One-based candidate index, or ``None`` when fallback was used."""

        return None if self.selected_index is None else self.selected_index + 1


class SNRCPS:
    """Self-normalized risk certification for an ordered candidate library.

    The class deliberately requires a supplied critical value. The paper's
    Brownian critical values are available through :mod:`snrcps.critical_values`;
    users may also provide values generated with their own discretization.
    """

    def __init__(
        self,
        *,
        risk_target: float = 0.10,
        critical_value: float,
        fallback: Literal["largest", "infinite", "raise"] = "largest",
    ) -> None:
        if not 0 < risk_target < 1:
            raise ValueError("risk_target must lie strictly between zero and one")
        if not np.isfinite(critical_value) or critical_value < 0:
            raise ValueError("critical_value must be finite and nonnegative")
        if fallback not in {"largest", "infinite", "raise"}:
            raise ValueError("fallback must be 'largest', 'infinite', or 'raise'")
        self.risk_target = float(risk_target)
        self.critical_value = float(critical_value)
        self.fallback = fallback
        self.result_: CertificationResult | None = None

    def certify(self, certification_scores: ArrayLike, thresholds: ArrayLike) -> CertificationResult:
        """Certify an already constructed ordered threshold library."""

        threshold_array = _as_finite_vector(thresholds, name="thresholds")
        if np.any(np.diff(threshold_array) < 0):
            raise ValueError("thresholds must be ordered from narrowest to widest")
        losses = loss_matrix(certification_scores, threshold_array)
        empirical_risk, normalizer = self_normalized_statistics(losses)
        sample_size = losses.shape[0]
        radius = self.critical_value * np.sqrt(np.maximum(normalizer, 0.0) / sample_size)
        pointwise_upper = np.minimum(1.0, empirical_risk + radius)
        ordered_upper = ordered_upper_envelope(pointwise_upper)
        selected_index = ordered_select(pointwise_upper, self.risk_target)

        fallback_used = selected_index is None
        if selected_index is None:
            if self.fallback == "raise":
                raise RuntimeError("no candidate is certified at the requested risk target")
            selected_threshold = (
                float(threshold_array[-1]) if self.fallback == "largest" else float("inf")
            )
        else:
            selected_threshold = float(threshold_array[selected_index])

        result = CertificationResult(
            thresholds=threshold_array.copy(),
            empirical_risk=empirical_risk,
            normalizer=normalizer,
            radius=radius,
            pointwise_upper=pointwise_upper,
            ordered_upper=ordered_upper,
            selected_index=selected_index,
            selected_threshold=selected_threshold,
            fallback_used=fallback_used,
            risk_target=self.risk_target,
            critical_value=self.critical_value,
        )
        self.result_ = result
        return result

    def fit(
        self,
        proposal_scores: ArrayLike,
        certification_scores: ArrayLike,
        *,
        levels: Iterable[float],
        safe_threshold: float | None = None,
    ) -> CertificationResult:
        """Construct proposal quantiles and certify them on a later block."""

        thresholds = candidate_thresholds(
            proposal_scores, levels, safe_threshold=safe_threshold
        )
        return self.certify(certification_scores, thresholds)

    def predict_interval(self, location: ArrayLike, scale: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Apply the selected feature-dependent interval to point predictions."""

        if self.result_ is None:
            raise RuntimeError("call fit or certify before predict_interval")
        return prediction_interval(location, scale, self.result_.selected_threshold)


def prediction_interval(
    location: ArrayLike,
    scale: ArrayLike,
    threshold: float | ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Build feature-dependent intervals ``location +/- threshold * scale``."""

    location_array = np.asarray(location, dtype=float)
    scale_array = np.asarray(scale, dtype=float)
    threshold_array = np.asarray(threshold, dtype=float)
    if location_array.shape != scale_array.shape:
        raise ValueError("location and scale must have the same shape")
    if not np.all(np.isfinite(location_array)):
        raise ValueError("location must contain only finite values")
    if not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0):
        raise ValueError("scale must be finite and strictly positive")
    if np.any(threshold_array < 0) or np.any(np.isnan(threshold_array)):
        raise ValueError("threshold must be nonnegative and not NaN")
    radius = threshold_array * scale_array
    return (location_array - radius).astype(float), (location_array + radius).astype(float)
