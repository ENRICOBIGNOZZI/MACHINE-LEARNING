import numpy as np

from rankcert.partial_order import (
    check_strict_partial_order,
    pairwise_curve_all_pairs,
    pairwise_curve_tail_pairs,
    relation_matrix,
    reliable_breadth,
    tail_indices,
    tail_reliable_breadth,
)


def test_band_relation_is_partial_order() -> None:
    predicted = np.array([0.08, 0.06, 0.051, 0.02])
    scale = np.array([0.01, 0.01, 0.02, 0.01])
    relation = relation_matrix(predicted, scale, 0.5)
    assert check_strict_partial_order(relation)


def test_breadth_decreases() -> None:
    predicted = np.array([0.09, 0.07, 0.05, 0.01])
    scale = np.full(4, 0.01)
    breadth = [reliable_breadth(predicted, scale, q) for q in [0.0, 0.5, 1.0, 2.0]]
    tail_breadth = [
        tail_reliable_breadth(predicted, scale, q, fraction=0.25)
        for q in [0.0, 0.5, 1.0, 2.0]
    ]
    assert np.all(np.diff(breadth) <= 1e-12)
    assert np.all(np.diff(tail_breadth) <= 1e-12)


def test_exact_all_pair_curve_matches_bruteforce() -> None:
    predicted = np.array([0.08, 0.04, 0.06, 0.01])
    scale = np.array([0.01, 0.02, 0.01, 0.01])
    realized = np.array([0.01, 0.03, -0.02, 0.00])
    q_grid = np.array([0.0, 0.5, 1.0])
    curve = pairwise_curve_all_pairs(predicted, scale, realized, q_grid)
    for k, q in enumerate(q_grid):
        relation = relation_matrix(predicted, scale, q)
        active = relation.sum()
        errors = np.sum(relation & (realized[:, None] <= realized[None, :]))
        expected = 0.0 if active == 0 else errors / active
        assert np.isclose(curve.error_rate[k], expected)


def test_exact_tail_curve_matches_bruteforce() -> None:
    predicted = np.array([0.09, 0.07, 0.04, 0.01, -0.02, -0.04])
    scale = np.array([0.02, 0.01, 0.02, 0.01, 0.02, 0.01])
    realized = np.array([0.01, -0.02, 0.04, 0.03, -0.01, 0.00])
    q_grid = np.array([0.0, 0.5, 1.0])
    fraction = 1.0 / 3.0
    curve = pairwise_curve_tail_pairs(
        predicted,
        scale,
        realized,
        q_grid,
        fraction=fraction,
    )
    short_idx, long_idx = tail_indices(predicted, fraction)
    for k, q in enumerate(q_grid):
        lower = predicted[long_idx] - q * scale[long_idx]
        upper = predicted[short_idx] + q * scale[short_idx]
        relation = lower[:, None] > upper[None, :]
        errors = relation & (realized[long_idx, None] <= realized[None, short_idx])
        active_count = relation.sum()
        expected_error = 0.0 if active_count == 0 else errors.sum() / active_count
        assert np.isclose(curve.error_rate[k], expected_error)
        assert np.isclose(curve.breadth[k], active_count / (len(long_idx) * len(short_idx)))
