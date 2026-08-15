"""One-month deployment evaluation for certified stock-ranking portfolios."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ._backtest_support import evaluate_method
from .partial_order import pairwise_curve_all_pairs, pairwise_curve_tail_pairs
from .portfolio import certified_graph_weights, edge_notional_weights, raw_rank_weights


def evaluate_deployment_month(
    group: pd.DataFrame,
    *,
    date: pd.Timestamp,
    epoch: int,
    q_selected: float,
    pair_universe: Literal["all", "tail"],
    tail_fraction: float,
    top_fraction: float,
    cost_bps: float,
    gross: float,
    previous_post: dict[str, dict[object, float] | None],
) -> dict[str, object]:
    asset_ids = group["asset_id"].to_numpy()
    predicted = group["prediction"].to_numpy(dtype=float)
    scale = group["scale"].to_numpy(dtype=float)
    realized = group["ret_fwd"].to_numpy(dtype=float)

    raw_graph_weights = certified_graph_weights(
        predicted,
        scale,
        0.0,
        gross=gross,
        universe=pair_universe,
        tail_fraction=tail_fraction,
    )
    certified_weights = certified_graph_weights(
        predicted,
        scale,
        q_selected,
        gross=gross,
        universe=pair_universe,
        tail_fraction=tail_fraction,
    )
    decile_weights = raw_rank_weights(predicted, top_fraction=top_fraction, gross=gross)
    edge_raw_weights = edge_notional_weights(
        predicted,
        scale,
        0.0,
        universe=pair_universe,
        tail_fraction=tail_fraction,
    )
    edge_certified_weights = edge_notional_weights(
        predicted,
        scale,
        q_selected,
        universe=pair_universe,
        tail_fraction=tail_fraction,
    )
    edge_removed_weights = edge_raw_weights - edge_certified_weights

    method_weights = {
        "raw_graph": raw_graph_weights,
        "certified_graph": certified_weights,
        "decile": decile_weights,
        "edge_raw": edge_raw_weights,
        "edge_certified": edge_certified_weights,
        "edge_removed": edge_removed_weights,
    }
    method_results = {}
    for method, weights in method_weights.items():
        result, post_return = evaluate_method(
            asset_ids,
            weights,
            realized,
            previous_post[method],
            cost_bps=cost_bps,
        )
        method_results[method] = result
        previous_post[method] = post_return

    evaluation_q = q_selected if np.isfinite(q_selected) else 1e9
    all_curve = pairwise_curve_all_pairs(predicted, scale, realized, [0.0, evaluation_q])
    tail_curve = pairwise_curve_tail_pairs(
        predicted,
        scale,
        realized,
        [0.0, evaluation_q],
        fraction=tail_fraction,
    )
    selected_curve = all_curve if pair_universe == "all" else tail_curve

    raw_result = method_results["raw_graph"]
    certified_result = method_results["certified_graph"]
    decile_result = method_results["decile"]
    edge_raw_result = method_results["edge_raw"]
    edge_certified_result = method_results["edge_certified"]
    edge_removed_result = method_results["edge_removed"]

    return {
        "date": date,
        "epoch": epoch,
        "pair_universe": pair_universe,
        "q_selected": q_selected,
        "raw_gross_return": raw_result.gross_return,
        "raw_turnover": raw_result.turnover,
        "raw_cost": raw_result.transaction_cost,
        "raw_net_return": raw_result.net_return,
        "certified_gross_return": certified_result.gross_return,
        "certified_turnover": certified_result.turnover,
        "certified_cost": certified_result.transaction_cost,
        "certified_net_return": certified_result.net_return,
        "raw_graph_gross_return": raw_result.gross_return,
        "raw_graph_turnover": raw_result.turnover,
        "raw_graph_cost": raw_result.transaction_cost,
        "raw_graph_net_return": raw_result.net_return,
        "certified_graph_gross_return": certified_result.gross_return,
        "certified_graph_turnover": certified_result.turnover,
        "certified_graph_cost": certified_result.transaction_cost,
        "certified_graph_net_return": certified_result.net_return,
        "decile_gross_return": decile_result.gross_return,
        "decile_turnover": decile_result.turnover,
        "decile_cost": decile_result.transaction_cost,
        "decile_net_return": decile_result.net_return,
        "edge_raw_gross_return": edge_raw_result.gross_return,
        "edge_raw_turnover": edge_raw_result.turnover,
        "edge_raw_cost": edge_raw_result.transaction_cost,
        "edge_raw_net_return": edge_raw_result.net_return,
        "edge_certified_gross_return": edge_certified_result.gross_return,
        "edge_certified_turnover": edge_certified_result.turnover,
        "edge_certified_cost": edge_certified_result.transaction_cost,
        "edge_certified_net_return": edge_certified_result.net_return,
        "edge_removed_gross_return": edge_removed_result.gross_return,
        "edge_removed_turnover": edge_removed_result.turnover,
        "edge_removed_cost": edge_removed_result.transaction_cost,
        "edge_removed_net_return": edge_removed_result.net_return,
        "edge_return_decomposition_error": (
            edge_raw_result.gross_return
            - edge_certified_result.gross_return
            - edge_removed_result.gross_return
        ),
        "raw_pair_error": selected_curve.error_rate[0],
        "certified_pair_error": selected_curve.error_rate[1],
        "reliable_breadth": selected_curve.breadth[1],
        "all_pair_error_raw": all_curve.error_rate[0],
        "all_pair_error_certified": all_curve.error_rate[1],
        "all_pair_reliable_breadth": all_curve.breadth[1],
        "tail_pair_error_raw": tail_curve.error_rate[0],
        "tail_pair_error_certified": tail_curve.error_rate[1],
        "tail_pair_reliable_breadth": tail_curve.breadth[1],
        "stress": float(group["stress"].iloc[0]) if "stress" in group else np.nan,
    }
