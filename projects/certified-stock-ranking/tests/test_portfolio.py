import numpy as np

from rankcert.portfolio import (
    certified_graph_weights,
    edge_notional_weights,
    evaluate_portfolio_month_by_asset,
    raw_rank_weights,
    turnover_from_mappings,
)


def test_portfolios_are_dollar_neutral() -> None:
    predicted = np.array([0.08, 0.04, 0.01, -0.02])
    scale = np.full(4, 0.01)
    raw = raw_rank_weights(predicted, top_fraction=0.25)
    certified = certified_graph_weights(predicted, scale, 0.5)
    tail_certified = certified_graph_weights(
        predicted,
        scale,
        0.5,
        universe="tail",
        tail_fraction=0.25,
    )
    assert np.isclose(raw.sum(), 0.0)
    assert np.isclose(certified.sum(), 0.0)
    assert np.isclose(tail_certified.sum(), 0.0)


def test_edge_decomposition_is_exact() -> None:
    predicted = np.array([0.08, 0.04, 0.01, -0.02])
    scale = np.array([0.01, 0.02, 0.01, 0.03])
    for universe in ["all", "tail"]:
        raw = edge_notional_weights(
            predicted,
            scale,
            0.0,
            universe=universe,
            tail_fraction=0.25,
        )
        certified = edge_notional_weights(
            predicted,
            scale,
            0.75,
            universe=universe,
            tail_fraction=0.25,
        )
        removed = raw - certified
        assert np.allclose(raw, certified + removed)


def test_turnover_includes_universe_exits_and_entries() -> None:
    previous = {"A": 0.5, "B": -0.5}
    current = {"A": 0.5, "C": -0.5}
    assert np.isclose(turnover_from_mappings(current, previous), 0.5)


def test_month_evaluation_returns_drifted_mapping() -> None:
    asset_ids = np.array([1, 2])
    weights = np.array([0.5, -0.5])
    returns = np.array([0.10, -0.10])
    result, post = evaluate_portfolio_month_by_asset(
        asset_ids,
        weights,
        returns,
        previous_post_return_weights=None,
        cost_bps=10.0,
    )
    assert np.isclose(result.gross_return, 0.10)
    assert np.isclose(result.turnover, 0.5)
    assert set(post) == {1, 2}
    assert not np.isclose(post[2], weights[1])
