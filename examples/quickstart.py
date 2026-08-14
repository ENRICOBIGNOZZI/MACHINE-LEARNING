"""Minimal SN-RCPS example with feature-dependent intervals."""

from __future__ import annotations

import numpy as np

from snrcps import SNRCPS, critical_value, interval_metrics


def main() -> None:
    rng = np.random.default_rng(20260814)
    proposal_scores = np.abs(rng.normal(size=2_000))
    certification_scores = np.abs(0.6 * rng.normal(size=2_000) + 0.4 * rng.normal())

    certifier = SNRCPS(
        risk_target=0.10,
        critical_value=critical_value(delta=0.10),
        fallback="largest",
    )
    result = certifier.fit(
        proposal_scores,
        certification_scores,
        levels=[0.90, 0.91, 0.92, 0.94, 0.96, 0.98, 0.995],
        safe_threshold=4.5,
    )

    x = np.linspace(-2.0, 2.0, 500)
    location = np.sin(x)
    scale = 0.25 + 0.20 * np.abs(x)
    y = location + scale * rng.normal(size=x.size)
    lower, upper = certifier.predict_interval(location, scale)

    print(f"selected candidate: {result.selected_candidate}")
    print(f"selected threshold: {result.selected_threshold:.4f}")
    print(interval_metrics(y, lower, upper, alpha=0.10))


if __name__ == "__main__":
    main()
