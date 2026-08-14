from __future__ import annotations

import numpy as np
import pytest

from snrcps import (
    SNRCPS,
    candidate_thresholds,
    ordered_select,
    ordered_upper_envelope,
    prediction_interval,
)


def test_candidate_thresholds_are_ordered_and_include_safe_candidate() -> None:
    scores = np.array([0.1, 0.4, 0.2, 0.9, 0.5])
    thresholds = candidate_thresholds(scores, [0.8, 0.5, 0.8], safe_threshold=1.5)
    assert np.all(np.diff(thresholds) >= 0)
    assert thresholds[-1] == pytest.approx(1.5)


def test_ordered_selector_enforces_complete_suffix() -> None:
    pointwise = np.array([0.08, 0.12, 0.07, 0.05])
    envelope = ordered_upper_envelope(pointwise)
    np.testing.assert_allclose(envelope, [0.12, 0.12, 0.07, 0.05])
    assert ordered_select(pointwise, 0.10) == 2


def test_snrcps_certifies_and_predicts_feature_dependent_intervals() -> None:
    proposal = np.linspace(0.0, 3.0, 400)
    certification = np.concatenate([np.full(360, 0.5), np.full(40, 2.0)])
    model = SNRCPS(risk_target=0.20, critical_value=0.0)
    result = model.fit(proposal, certification, levels=[0.80, 0.90, 0.95])
    assert result.selected_index is not None
    location = np.array([10.0, 10.0])
    scale = np.array([1.0, 2.0])
    lower, upper = model.predict_interval(location, scale)
    assert (upper[1] - lower[1]) == pytest.approx(2.0 * (upper[0] - lower[0]))


def test_prediction_interval_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        prediction_interval([0.0], [0.0], 1.0)


def test_fallback_modes() -> None:
    scores = np.ones(50)
    thresholds = np.array([0.1, 0.2])
    largest = SNRCPS(risk_target=0.1, critical_value=0.0, fallback="largest")
    result = largest.certify(scores, thresholds)
    assert result.fallback_used
    assert result.selected_threshold == pytest.approx(0.2)

    infinite = SNRCPS(risk_target=0.1, critical_value=0.0, fallback="infinite")
    assert np.isinf(infinite.certify(scores, thresholds).selected_threshold)

    raising = SNRCPS(risk_target=0.1, critical_value=0.0, fallback="raise")
    with pytest.raises(RuntimeError):
        raising.certify(scores, thresholds)
