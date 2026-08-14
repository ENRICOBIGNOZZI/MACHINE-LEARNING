from __future__ import annotations

import numpy as np
import pytest

from snrcps import (
    EventTriggeredSNRCPS,
    SNRCPS,
    SelfNormalizedRiskMonitor,
    equal_confidence_spending,
    polynomial_confidence_spending,
)


def fitted_model(threshold: float) -> SNRCPS:
    model = SNRCPS(risk_target=0.10, critical_value=0.0, fallback="largest")
    model.certify(np.array([0.10, 0.20, 0.30, 0.40]), np.array([threshold]))
    return model


def test_equal_spending_uses_complete_budget() -> None:
    spending = equal_confidence_spending(0.10, 9)
    assert spending.shape == (9,)
    assert np.isclose(spending.sum(), 0.10)


def test_polynomial_spending_is_summable_and_decreasing() -> None:
    spending = polynomial_confidence_spending(0.10, 10_000)
    assert np.all(np.diff(spending) < 0)
    assert spending.sum() < 0.10
    assert spending.sum() > 0.09999


def test_monitor_does_not_trigger_on_low_risk_window() -> None:
    monitor = SelfNormalizedRiskMonitor(
        risk_target=0.10,
        critical_value=1.0,
        window=100,
        minimum_observations=50,
    )
    monitor.extend([0.0] * 95 + [1.0] * 5)
    decision = monitor.checkpoint()
    assert not decision.triggered
    assert decision.empirical_risk == 0.05


def test_monitor_triggers_after_required_patience() -> None:
    monitor = SelfNormalizedRiskMonitor(
        risk_target=0.10,
        critical_value=0.0,
        window=100,
        minimum_observations=50,
        patience=2,
    )
    monitor.extend(np.tile([1.0, 0.0, 0.0, 0.0], 25))
    first = monitor.checkpoint()
    second = monitor.checkpoint()
    assert not first.triggered
    assert second.triggered
    assert second.reason == "risk"


def test_monitor_can_trigger_for_excess_conservatism() -> None:
    monitor = SelfNormalizedRiskMonitor(
        risk_target=0.10,
        critical_value=0.5,
        window=100,
        minimum_observations=50,
        efficiency_margin=0.03,
    )
    monitor.extend([0.0] * 100)
    decision = monitor.checkpoint()
    assert decision.triggered
    assert decision.reason == "efficiency"


def test_event_trigger_replaces_certified_rule() -> None:
    initial = fitted_model(0.50)
    monitor = SelfNormalizedRiskMonitor(
        risk_target=0.10,
        critical_value=0.0,
        window=6,
        minimum_observations=6,
    )
    online = EventTriggeredSNRCPS(current=initial, monitor=monitor)
    for score in [1.0, 1.1, 0.9, 1.2, 1.3, 0.8]:
        online.observe_score(score)
    replacement = fitted_model(2.0)
    event = online.checkpoint(checkpoint=1, replacement=replacement)
    assert event.triggered
    assert online.selected_threshold_ == 2.0
    assert event.old_threshold == 0.50
    assert event.new_threshold == 2.0
    assert monitor.observations == 0


def test_trigger_requires_replacement_inputs() -> None:
    initial = fitted_model(0.50)
    monitor = SelfNormalizedRiskMonitor(
        risk_target=0.10,
        critical_value=0.0,
        window=4,
        minimum_observations=4,
    )
    online = EventTriggeredSNRCPS(current=initial, monitor=monitor)
    for score in [1.0, 1.1, 0.9, 1.2]:
        online.observe_score(score)
    with pytest.raises(ValueError, match="requires"):
        online.checkpoint(checkpoint=1)
