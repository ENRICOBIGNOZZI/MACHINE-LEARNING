"""Scheduled monitoring and event-triggered episodic recertification.

The objects in this module deliberately separate monitoring from certification.
A monitoring alarm may decide *when* a fresh rule is proposed, but every rule
that is eventually deployed must still be certified on its own chronological
certification block.  The resulting workflow is event-triggered at
pre-specified checkpoints; it is not advertised as arbitrary-stopping or
per-observation anytime validity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike

from .calibration import CertificationResult, SNRCPS, self_normalized_statistics

MonitorMode = Literal["evidence", "support"]


@dataclass(frozen=True)
class TriggerDecision:
    """Diagnostics returned at one deterministic monitoring checkpoint."""

    checkpoint: int
    observations: int
    empirical_risk: float
    normalizer: float
    standard_error_scale: float
    lower_bound: float
    upper_bound: float
    trigger_statistic: float
    triggered: bool
    reason: str | None
    consecutive_alerts: int
    degenerate: bool
    mode: MonitorMode


@dataclass(frozen=True)
class RecertificationEvent:
    """State transition recorded by :class:`EventTriggeredSNRCPS`."""

    checkpoint: int
    triggered: bool
    old_threshold: float
    new_threshold: float
    monitor: TriggerDecision
    candidate_count: int
    fallback_used: bool


@dataclass
class SelfNormalizedRiskMonitor:
    """Monitor a frozen prediction rule at deterministic checkpoints.

    Parameters
    ----------
    risk_target:
        Target mean bounded loss, usually the miscoverage level ``alpha``.
    critical_value:
        One-sided self-normalized critical value used on the fixed monitoring
        window.  A checkpoint-specific value can be supplied by replacing this
        attribute before calling :meth:`checkpoint`.
    window:
        Maximum number of most recent losses retained by the monitor.
    minimum_observations:
        Number of observations required before a decision is returned.
    safety_margin:
        Evidence mode alarms only when the lower bound exceeds
        ``risk_target + safety_margin``.
    efficiency_margin:
        Optional operational trigger for a rule that appears unnecessarily
        conservative.  This is retained for backwards compatibility and maps
        to ``mode='support'`` when it is the active reason.
    patience:
        Number of consecutive alerts required before recertification.
    cooldown_checkpoints:
        Number of checkpoints suppressed immediately after :meth:`reset`.
    mode:
        ``'evidence'`` gives the clean one-sided risk alarm. ``'support'`` is a
        proactive operational mode that alarms when the upper bound no longer
        establishes risk below ``risk_target - safety_margin``.  The latter is
        not a test that the deployed risk exceeds the target.
    """

    risk_target: float = 0.10
    critical_value: float = 2.0
    window: int = 256
    minimum_observations: int = 64
    safety_margin: float = 0.0
    efficiency_margin: float | None = None
    patience: int = 1
    cooldown_checkpoints: int = 0
    mode: MonitorMode = "evidence"

    def __post_init__(self) -> None:
        if not 0.0 < self.risk_target < 1.0:
            raise ValueError("risk_target must lie strictly between zero and one")
        if not np.isfinite(self.critical_value) or self.critical_value < 0.0:
            raise ValueError("critical_value must be finite and nonnegative")
        if self.window < 2:
            raise ValueError("window must be at least two")
        if not 2 <= self.minimum_observations <= self.window:
            raise ValueError("minimum_observations must lie between two and window")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be nonnegative")
        if self.efficiency_margin is not None and self.efficiency_margin < 0.0:
            raise ValueError("efficiency_margin must be nonnegative or None")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.cooldown_checkpoints < 0:
            raise ValueError("cooldown_checkpoints must be nonnegative")
        if self.mode not in {"evidence", "support"}:
            raise ValueError("mode must be 'evidence' or 'support'")

        self._losses: deque[float] = deque(maxlen=self.window)
        self._checkpoint = 0
        self._consecutive_alerts = 0
        self._cooldown = 0

    @property
    def observations(self) -> int:
        """Number of losses currently retained."""

        return len(self._losses)

    def update(self, loss: float | bool) -> None:
        """Append one bounded deployment loss."""

        value = float(loss)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("loss must lie in [0, 1]")
        self._losses.append(value)

    def extend(self, losses: Iterable[float | bool]) -> None:
        """Append a sequence of bounded deployment losses."""

        for loss in losses:
            self.update(loss)

    def reset(self, *, keep_history: bool = False) -> None:
        """Reset alert state after a deployment update.

        By default the monitoring window is cleared, so the replacement rule is
        judged only on losses realized after deployment.
        """

        if not keep_history:
            self._losses.clear()
        self._consecutive_alerts = 0
        self._cooldown = self.cooldown_checkpoints

    def checkpoint(self, *, critical_value: float | None = None) -> TriggerDecision:
        """Evaluate the current fixed monitoring window."""

        self._checkpoint += 1
        critical = self.critical_value if critical_value is None else float(critical_value)
        if not np.isfinite(critical) or critical < 0.0:
            raise ValueError("critical_value must be finite and nonnegative")

        observations = len(self._losses)
        if observations < self.minimum_observations:
            return TriggerDecision(
                checkpoint=self._checkpoint,
                observations=observations,
                empirical_risk=float("nan"),
                normalizer=float("nan"),
                standard_error_scale=float("nan"),
                lower_bound=float("nan"),
                upper_bound=float("nan"),
                trigger_statistic=float("nan"),
                triggered=False,
                reason=None,
                consecutive_alerts=self._consecutive_alerts,
                degenerate=False,
                mode=self.mode,
            )

        losses = np.asarray(self._losses, dtype=float)
        empirical, normalizer_array = self_normalized_statistics(losses)
        risk = float(empirical[0])
        normalizer = max(float(normalizer_array[0]), 0.0)
        scale = float(np.sqrt(normalizer / observations))
        degenerate = normalizer <= 1e-14
        radius = critical * scale
        lower = max(0.0, risk - radius)
        upper = min(1.0, risk + radius)

        if self.mode == "evidence":
            boundary = self.risk_target + self.safety_margin
            raw_alert = lower > boundary
            reason = "risk" if raw_alert else None
            if scale == 0.0:
                statistic = (
                    float("inf") if risk > boundary else
                    float("-inf") if risk < boundary else 0.0
                )
            else:
                statistic = (risk - boundary) / scale
        else:
            boundary = max(0.0, self.risk_target - self.safety_margin)
            raw_alert = upper > boundary
            reason = "support" if raw_alert else None
            if scale == 0.0:
                statistic = (
                    float("inf") if risk > boundary else
                    float("-inf") if risk < boundary else 0.0
                )
            else:
                statistic = (risk - boundary) / scale

        # Optional two-sided operational alarm retained from the original API.
        if self.efficiency_margin is not None and not raw_alert:
            efficiency_boundary = max(0.0, self.risk_target - self.efficiency_margin)
            if upper < efficiency_boundary:
                raw_alert = True
                reason = "efficiency"

        if self._cooldown > 0:
            self._cooldown -= 1
            raw_alert = False
            reason = None

        self._consecutive_alerts = self._consecutive_alerts + 1 if raw_alert else 0
        triggered = self._consecutive_alerts >= self.patience
        if not triggered and reason is not None:
            reason = None

        return TriggerDecision(
            checkpoint=self._checkpoint,
            observations=observations,
            empirical_risk=risk,
            normalizer=normalizer,
            standard_error_scale=scale,
            lower_bound=lower,
            upper_bound=upper,
            trigger_statistic=float(statistic),
            triggered=bool(triggered),
            reason=reason,
            consecutive_alerts=self._consecutive_alerts,
            degenerate=degenerate,
            mode=self.mode,
        )


class EventTriggeredSNRCPS:
    """Operational wrapper for a frozen SN-RCPS rule and risk monitor.

    Monitoring checkpoints are deterministic.  An alarm changes the deployed
    rule only after a separately fitted and certified replacement is supplied,
    or after chronological proposal and certification scores are supplied to
    this method.  Monitoring therefore changes update timing, not the validity
    standard applied to replacement rules.
    """

    def __init__(self, *, current: SNRCPS, monitor: SelfNormalizedRiskMonitor) -> None:
        if current.result_ is None:
            raise ValueError("current must already be fitted or certified")
        self.current = current
        self.monitor = monitor
        self.events_: list[RecertificationEvent] = []
        self.observations_: int = 0

    @property
    def selected_threshold_(self) -> float:
        """Threshold of the currently deployed rule."""

        if self.current.result_ is None:  # pragma: no cover - guarded at construction
            raise RuntimeError("current rule is not certified")
        return float(self.current.result_.selected_threshold)

    def observe_score(self, score: float) -> float:
        """Record one normalized score and return its miscoverage loss."""

        value = float(score)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("score must be finite and nonnegative")
        loss = float(value > self.selected_threshold_)
        self.monitor.update(loss)
        self.observations_ += 1
        return loss

    def predict_interval(self, location: ArrayLike, scale: ArrayLike):
        """Apply the currently deployed feature-dependent interval."""

        return self.current.predict_interval(location, scale)

    def checkpoint(
        self,
        *,
        checkpoint: int,
        replacement: SNRCPS | None = None,
        proposal_scores: ArrayLike | None = None,
        certification_scores: ArrayLike | None = None,
        levels: Iterable[float] | None = None,
        safe_threshold: float | None = None,
        critical_value: float | None = None,
    ) -> RecertificationEvent:
        """Evaluate the monitor and optionally deploy a certified replacement.

        When no alarm fires, supplied replacement data are ignored.  When an
        alarm fires, provide either a pre-fitted ``replacement`` or the proposal
        and certification inputs needed to fit one with the current certifier's
        risk target, critical value, and fallback policy.
        """

        decision = self.monitor.checkpoint(critical_value=critical_value)
        old_threshold = self.selected_threshold_

        if decision.triggered:
            if replacement is None:
                if proposal_scores is None or certification_scores is None or levels is None:
                    raise ValueError(
                        "a triggered checkpoint requires a certified replacement or "
                        "proposal_scores, certification_scores, and levels"
                    )
                replacement = SNRCPS(
                    risk_target=self.current.risk_target,
                    critical_value=self.current.critical_value,
                    fallback=self.current.fallback,
                )
                replacement.fit(
                    proposal_scores,
                    certification_scores,
                    levels=levels,
                    safe_threshold=safe_threshold,
                )
            elif replacement.result_ is None:
                raise ValueError("replacement must already be fitted or certified")
            self.current = replacement
            self.monitor.reset(keep_history=False)

        result: CertificationResult | None = self.current.result_
        if result is None:  # pragma: no cover - guarded above
            raise RuntimeError("deployed rule is not certified")
        event = RecertificationEvent(
            checkpoint=int(checkpoint),
            triggered=bool(decision.triggered),
            old_threshold=float(old_threshold),
            new_threshold=float(result.selected_threshold),
            monitor=decision,
            candidate_count=int(result.thresholds.size),
            fallback_used=bool(result.fallback_used),
        )
        self.events_.append(event)
        return event


def polynomial_confidence_spending(total_delta: float, count: int) -> np.ndarray:
    """Return the first ``count`` terms of ``6 delta / (pi^2 j^2)``."""

    if not 0.0 < total_delta < 1.0:
        raise ValueError("total_delta must lie strictly between zero and one")
    if count < 1:
        raise ValueError("count must be positive")
    indices = np.arange(1, count + 1, dtype=float)
    return 6.0 * total_delta / (np.pi * np.pi * indices * indices)


def equal_confidence_spending(total_delta: float, count: int) -> np.ndarray:
    """Split a finite global confidence budget equally across deployments."""

    if not 0.0 < total_delta < 1.0:
        raise ValueError("total_delta must lie strictly between zero and one")
    if count < 1:
        raise ValueError("count must be positive")
    return np.full(count, total_delta / count, dtype=float)


# Concise aliases for notebook users.
equal_spending = equal_confidence_spending
polynomial_spending = polynomial_confidence_spending
