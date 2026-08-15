"""Publication-style outputs for the synthetic software validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import BacktestOutput
from .metrics import annualized_summary
from .partial_order import relation_matrix


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def conceptual_partial_order(path: Path) -> None:
    labels = np.array(list("ABCDEF"))
    predicted = np.array([0.085, 0.081, 0.079, 0.058, 0.051, 0.047])
    scale = np.array([0.010, 0.012, 0.011, 0.009, 0.013, 0.010])
    q = 0.55
    relation = relation_matrix(predicted, scale, q)
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.errorbar(x, predicted, yerr=q * scale, fmt="o", capsize=5)
    for i, label in enumerate(labels):
        axis.text(i, predicted[i] + q * scale[i] + 0.002, label, ha="center")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if relation[i, j]:
                y = min(predicted[i] - q * scale[i], predicted[j] + q * scale[j]) - 0.004
                axis.annotate(
                    "",
                    xy=(j, y),
                    xytext=(i, y),
                    arrowprops={"arrowstyle": "->"},
                )
    axis.set_xticks([])
    axis.set_ylabel("Predicted monthly return and score band")
    axis.set_title("A certified ranking is a partial order: overlap implies abstention")
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    _save(figure, path)


def write_synthetic_report(
    panel: pd.DataFrame,
    output: BacktestOutput,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    panel.head(5_000).to_csv(directory / "synthetic_panel_sample.csv", index=False)
    output.epochs.to_csv(directory / "epochs.csv", index=False)
    output.monthly.to_csv(directory / "monthly_results.csv", index=False)
    output.certification_curves.to_csv(directory / "certification_curves.csv", index=False)
    conceptual_partial_order(directory / "concept_partial_order.pdf")

    curves = output.certification_curves
    if not curves.empty:
        epoch = int(curves["epoch"].iloc[len(curves) // 2])
        selected_epoch = curves.loc[curves["epoch"] == epoch]
        figure, axis = plt.subplots(figsize=(7.6, 4.6))
        axis.plot(
            selected_epoch["breadth"],
            selected_epoch["empirical_error"],
            marker="o",
            label="Pairwise error",
        )
        axis.plot(
            selected_epoch["breadth"],
            selected_epoch["upper_bound"],
            marker="s",
            label="SN upper bound",
        )
        selected = selected_epoch.loc[selected_epoch["selected"]]
        if not selected.empty:
            axis.scatter(
                selected["breadth"],
                selected["upper_bound"],
                s=90,
                zorder=4,
                label="Selected",
            )
        axis.axhline(0.30, linestyle="--", linewidth=1, label="Risk budget")
        axis.set_xlabel("Reliable breadth")
        axis.set_ylabel("Risk")
        axis.set_title("Synthetic certification frontier")
        axis.invert_xaxis()
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        _save(figure, directory / "certification_frontier.pdf")

    monthly = output.monthly.copy()
    if monthly.empty:
        return

    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    axis.plot(monthly["date"], monthly["reliable_breadth"])
    axis.set_ylabel("Reliable breadth")
    axis.set_title("Synthetic reliable breadth is state dependent")
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, directory / "reliable_breadth_time_series.pdf")

    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    axis.plot(
        monthly["date"],
        (1.0 + monthly["raw_graph_net_return"]).cumprod(),
        label="Raw equal-gross graph",
    )
    axis.plot(
        monthly["date"],
        (1.0 + monthly["certified_graph_net_return"]).cumprod(),
        label="Certified equal-gross graph",
    )
    axis.set_ylabel("Growth of one unit")
    axis.set_title("Synthetic software validation -- not an empirical result")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, directory / "synthetic_cumulative_net_returns.pdf")

    turnover = monthly[
        [
            "raw_graph_turnover",
            "certified_graph_turnover",
            "edge_raw_turnover",
            "edge_certified_turnover",
        ]
    ].mean()
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    labels = ["Raw graph", "Certified graph", "Raw edge", "Certified edge"]
    positions = np.arange(len(labels))
    axis.bar(positions, turnover.to_numpy())
    axis.set_xticks(positions, labels, rotation=15)
    axis.set_ylabel("Mean one-way turnover")
    axis.set_title("Abstention and equal-gross rescaling are distinct")
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, directory / "synthetic_turnover_costs.pdf")

    rows = []
    for name, column in [
        ("Raw equal-gross graph", "raw_graph_net_return"),
        ("Certified equal-gross graph", "certified_graph_net_return"),
        ("Raw variable-notional edge", "edge_raw_net_return"),
        ("Certified variable-notional edge", "edge_certified_net_return"),
    ]:
        statistics = annualized_summary(monthly[column])
        statistics["method"] = name
        rows.append(statistics)
    pd.DataFrame(rows).to_csv(
        directory / "synthetic_performance_summary.csv",
        index=False,
    )

    diagnostics = pd.DataFrame(
        {
            "metric": [
                "raw_pair_error",
                "certified_pair_error",
                "reliable_breadth",
                "raw_graph_turnover",
                "certified_graph_turnover",
                "edge_raw_turnover",
                "edge_certified_turnover",
                "max_abs_edge_return_decomposition_error",
            ],
            "value": [
                monthly["raw_pair_error"].mean(),
                monthly["certified_pair_error"].mean(),
                monthly["reliable_breadth"].mean(),
                monthly["raw_graph_turnover"].mean(),
                monthly["certified_graph_turnover"].mean(),
                monthly["edge_raw_turnover"].mean(),
                monthly["edge_certified_turnover"].mean(),
                monthly["edge_return_decomposition_error"].abs().max(),
            ],
        }
    )
    diagnostics.to_csv(directory / "synthetic_diagnostics.csv", index=False)
