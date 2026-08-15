"""Run the end-to-end synthetic software validation."""

from pathlib import Path

import yaml

from rankcert.backtest import run_episodic_backtest
from rankcert.reporting import write_synthetic_report
from rankcert.synthetic import generate_synthetic_panel


if __name__ == "__main__":
    configuration = yaml.safe_load(Path("configs/synthetic.yaml").read_text())
    panel = generate_synthetic_panel(
        n_months=int(configuration["n_months"]),
        n_assets=int(configuration["n_assets"]),
        n_features=int(configuration["n_features"]),
        seed=int(configuration["seed"]),
    )
    # Fixed only for the fast software demo. Production runs simulate and store
    # the Brownian critical value with the requested Monte Carlo budget.
    output = run_episodic_backtest(
        panel,
        proposal_months=int(configuration["proposal_months"]),
        certification_months=int(configuration["certification_months"]),
        deployment_months=int(configuration["deployment_months"]),
        alpha=float(configuration["alpha"]),
        delta=float(configuration["delta"]),
        cost_bps=float(configuration["cost_bps"]),
        critical_value=3.2,
        pair_universe="all",
        seed=int(configuration["seed"]),
    )
    write_synthetic_report(panel, output, Path("results/synthetic"))
