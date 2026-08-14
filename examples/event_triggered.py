"""Minimal event-triggered SN-RCPS service example."""

from __future__ import annotations

import numpy as np

from snrcps import EventTriggeredSNRCPS, SNRCPS, SelfNormalizedRiskMonitor


def certify(scores: np.ndarray, thresholds: np.ndarray) -> SNRCPS:
    """Create a small deterministic certificate for the example."""

    model = SNRCPS(risk_target=0.10, critical_value=0.0, fallback="largest")
    model.certify(scores, thresholds)
    return model


def main() -> None:
    thresholds = np.array([0.8, 1.2, 1.6, 2.0])
    initial = certify(
        np.array([0.1, 0.4, 0.7, 0.9, 1.0, 1.1, 0.5, 0.6, 0.3, 0.2]),
        thresholds,
    )
    monitor = SelfNormalizedRiskMonitor(
        risk_target=0.10,
        critical_value=0.0,
        window=12,
        minimum_observations=12,
        patience=1,
        mode="evidence",
    )
    service = EventTriggeredSNRCPS(current=initial, monitor=monitor)

    # The deployed score distribution deteriorates.
    deployment_scores = np.array(
        [2.2, 1.9, 2.4, 1.8, 2.1, 2.3, 1.7, 2.0, 2.5, 1.9, 2.2, 2.4]
    )
    for score in deployment_scores:
        service.observe_score(float(score))

    replacement = certify(
        np.array([0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 1.9, 2.0, 2.1, 2.2]),
        thresholds,
    )
    event = service.checkpoint(checkpoint=1, replacement=replacement)

    print(f"triggered: {event.triggered}")
    print(f"old threshold: {event.old_threshold:.2f}")
    print(f"new threshold: {event.new_threshold:.2f}")
    print(f"monitor empirical risk: {event.monitor.empirical_risk:.3f}")


if __name__ == "__main__":
    main()
