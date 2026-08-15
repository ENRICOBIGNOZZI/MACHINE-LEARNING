"""Episodic certification and portfolio deployment on prediction panels."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

from ._backtest_deploy import evaluate_deployment_month
from ._backtest_support import BacktestOutput, candidate_grid as build_candidate_grid, loss_matrix, month_groups
from .snrcps import CertificationResult, certify_monotone_losses


def run_episodic_backtest(
    panel: pd.DataFrame,
    *,
    proposal_months: int = 36,
    certification_months: int = 72,
    deployment_months: int = 12,
    initial_history_months: int | None = None,
    alpha: float = 0.30,
    delta: float = 0.10,
    top_fraction: float = 0.10,
    cost_bps: float = 25.0,
    gross: float = 2.0,
    critical_value: float | None = None,
    candidate_grid: Sequence[float] | None = None,
    pair_universe: Literal["all", "tail"] = "all",
    tail_fraction: float = 0.10,
    seed: int = 20260814,
) -> BacktestOutput:
    """Run frozen-threshold deployment on a precomputed prediction panel.

    The raw and certified headline portfolios use the same graph construction,
    at ``q=0`` and ``q=q_selected`` respectively. A conventional decile
    portfolio is reported separately. Turnover is computed over the union of
    changing stock universes after passive return drift, so delisted or filtered
    securities are explicitly liquidated.
    """

    required = {"date", "asset_id", "prediction", "scale", "ret_fwd"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")
    if pair_universe not in {"all", "tail"}:
        raise ValueError("pair_universe must be 'all' or 'tail'")
    if not 0.0 < tail_fraction <= 0.5:
        raise ValueError("tail_fraction must lie in (0, 0.5]")

    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.dropna(subset=["prediction", "scale", "ret_fwd"])
    data = data.sort_values(["date", "asset_id"]).reset_index(drop=True)
    dates = np.array(sorted(data["date"].unique()), dtype="datetime64[ns]")

    proposal_history = proposal_months if candidate_grid is None else 0
    required_history = certification_months + proposal_history
    if initial_history_months is None:
        initial_history_months = required_history
    if initial_history_months < required_history:
        raise ValueError(
            "initial_history_months is shorter than the required certification/proposal history"
        )
    if len(dates) < initial_history_months + deployment_months:
        raise ValueError("insufficient monthly history")

    epoch_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    previous_post: dict[str, dict[object, float] | None] = {
        "raw_graph": None,
        "certified_graph": None,
        "decile": None,
        "edge_raw": None,
        "edge_certified": None,
        "edge_removed": None,
    }

    epoch = 0
    deploy_start_index = initial_history_months
    while deploy_start_index < len(dates):
        certification_end = deploy_start_index
        certification_start = certification_end - certification_months
        if certification_start < 0:
            break
        deployment_end = min(deploy_start_index + deployment_months, len(dates))
        certification_dates = pd.to_datetime(dates[certification_start:certification_end])
        deployment_dates = pd.to_datetime(dates[deploy_start_index:deployment_end])
        certification_groups = month_groups(data, certification_dates)

        if candidate_grid is None:
            proposal_end = certification_start
            proposal_start = proposal_end - proposal_months
            if proposal_start < 0:
                break
            proposal_dates = pd.to_datetime(dates[proposal_start:proposal_end])
            proposal_groups = month_groups(data, proposal_dates)
            q_grid = build_candidate_grid(proposal_groups, seed + 10_000 * epoch)
        else:
            proposal_groups = []
            q_grid = np.asarray(list(candidate_grid), dtype=float)
            if (
                q_grid.ndim != 1
                or q_grid.size == 0
                or np.any(~np.isfinite(q_grid))
                or np.any(q_grid < 0.0)
                or np.any(np.diff(q_grid) < 0.0)
            ):
                raise ValueError(
                    "candidate_grid must be a finite, nonnegative, increasing sequence"
                )

        losses, pair_curves = loss_matrix(
            certification_groups,
            q_grid,
            pair_universe=pair_universe,
            tail_fraction=tail_fraction,
        )
        certification: CertificationResult = certify_monotone_losses(
            losses,
            q_grid,
            alpha=alpha,
            delta=delta,
            critical_value=critical_value,
            mc_paths=50_000,
            mc_grid=1_000,
            mc_seed=seed,
        )
        mean_breadth = np.mean(np.vstack([curve.breadth for curve in pair_curves]), axis=0)
        mean_error = np.mean(np.vstack([curve.error_rate for curve in pair_curves]), axis=0)
        mean_wrong_mass = np.mean(
            np.vstack([curve.wrong_bet_mass for curve in pair_curves]), axis=0
        )
        for k, q_k in enumerate(q_grid):
            curve_rows.append(
                {
                    "epoch": epoch,
                    "pair_universe": pair_universe,
                    "q": q_k,
                    "empirical_error": mean_error[k],
                    "empirical_wrong_bet_mass": mean_wrong_mass[k],
                    "monotone_risk": certification.empirical_risk[k],
                    "upper_bound": certification.upper_bound[k],
                    "breadth": mean_breadth[k],
                    "selected": certification.selected_index == k,
                }
            )
        epoch_rows.append(
            {
                "epoch": epoch,
                "pair_universe": pair_universe,
                "deployment_start": deployment_dates[0],
                "deployment_end": deployment_dates[-1],
                "q_selected": certification.selected_value,
                "fallback": certification.fallback,
                "critical_value": certification.critical_value,
                "n_proposal_months": len(proposal_groups),
                "n_certification_months": len(certification_groups),
            }
        )

        for date in deployment_dates:
            group = data.loc[data["date"] == date]
            monthly_rows.append(
                evaluate_deployment_month(
                    group,
                    date=date,
                    epoch=epoch,
                    q_selected=certification.selected_value,
                    pair_universe=pair_universe,
                    tail_fraction=tail_fraction,
                    top_fraction=top_fraction,
                    cost_bps=cost_bps,
                    gross=gross,
                    previous_post=previous_post,
                )
            )

        epoch += 1
        deploy_start_index = deployment_end

    return BacktestOutput(
        epochs=pd.DataFrame(epoch_rows),
        monthly=pd.DataFrame(monthly_rows),
        certification_curves=pd.DataFrame(curve_rows),
    )
