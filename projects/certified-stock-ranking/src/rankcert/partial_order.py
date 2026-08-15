"""Public interface for certified stock-ranking partial orders."""

from ._partial_order_base import (
    PairwiseCurve,
    check_strict_partial_order,
    dominance_scores,
    relation_matrix,
    reliable_breadth,
    score_bands,
    tail_dominance_scores,
    tail_indices,
    tail_reliable_breadth,
)
from ._partial_order_curves import (
    pairwise_curve,
    pairwise_curve_all_pairs,
    pairwise_curve_tail_pairs,
    proposal_grid_from_margins,
    sample_pair_margins,
)

__all__ = [
    "PairwiseCurve",
    "check_strict_partial_order",
    "dominance_scores",
    "pairwise_curve",
    "pairwise_curve_all_pairs",
    "pairwise_curve_tail_pairs",
    "proposal_grid_from_margins",
    "relation_matrix",
    "reliable_breadth",
    "sample_pair_margins",
    "score_bands",
    "tail_dominance_scores",
    "tail_indices",
    "tail_reliable_breadth",
]
