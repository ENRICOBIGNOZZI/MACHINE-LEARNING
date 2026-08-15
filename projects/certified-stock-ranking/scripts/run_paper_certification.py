"""Run the prespecified certification and portfolio experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rankcert.backtest import run_episodic_backtest
from rankcert.metrics import annualized_summary, certainty_equivalent, newey_west_tstat


def _read_panel(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    parser.add_argument("--critical-value", type=float, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    predictions = _read_panel(args.predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    certification_config = config["certification"]
    portfolio_config = config["portfolio"]
    pair_universes = certification_config.get("pair_universes", ["all"])
    tail_fraction = float(certification_config.get("tail_fraction", 0.10))

    all_monthly: list[pd.DataFrame] = []
    all_epochs: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for pair_universe in pair_universes:
        for alpha in certification_config["alpha_grid"]:
            for cost_bps in portfolio_config["linear_cost_bps"]:
                output = run_episodic_backtest(
                    predictions,
                    proposal_months=0,
                    certification_months=int(certification_config["certification_months"]),
                    deployment_months=int(certification_config["deployment_months"]),
                    alpha=float(alpha),
                    delta=float(certification_config["delta"]),
                    top_fraction=float(portfolio_config["raw_top_fraction"]),
                    cost_bps=float(cost_bps),
                    gross=float(portfolio_config["gross_leverage"]),
                    critical_value=args.critical_value,
                    candidate_grid=certification_config["candidate_grid"],
                    pair_universe=pair_universe,
                    tail_fraction=tail_fraction,
                    seed=int(config["project"]["seed"]),
                )
                monthly = output.monthly.assign(alpha=alpha, cost_bps=cost_bps)
                epochs = output.epochs.assign(alpha=alpha, cost_bps=cost_bps)
                curves = output.certification_curves.assign(alpha=alpha, cost_bps=cost_bps)
                all_monthly.append(monthly)
                all_epochs.append(epochs)
                all_curves.append(curves)

                return_columns = {
                    "raw_graph": "raw_graph_net_return",
                    "certified_graph": "certified_graph_net_return",
                    "decile": "decile_net_return",
                    "edge_raw": "edge_raw_net_return",
                    "edge_certified": "edge_certified_net_return",
                    "edge_removed": "edge_removed_net_return",
                }
                for method, column in return_columns.items():
                    statistics = annualized_summary(monthly[column])
                    statistics.update(
                        {
                            "pair_universe": pair_universe,
                            "alpha": alpha,
                            "cost_bps": cost_bps,
                            "method": method,
                            "certainty_equivalent": certainty_equivalent(monthly[column]),
                            "nw_tstat": newey_west_tstat(monthly[column]),
                            "mean_turnover": float(monthly[f"{method}_turnover"].mean())
                            if f"{method}_turnover" in monthly
                            else float("nan"),
                        }
                    )
                    summaries.append(statistics)

    pd.concat(all_monthly, ignore_index=True).to_csv(
        args.output / "monthly_results.csv", index=False
    )
    pd.concat(all_epochs, ignore_index=True).to_csv(args.output / "epochs.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(
        args.output / "certification_curves.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(args.output / "performance_summary.csv", index=False)
