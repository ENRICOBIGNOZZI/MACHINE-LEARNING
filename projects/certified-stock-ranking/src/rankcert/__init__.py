"""Certified stock-ranking toolkit."""

from .partial_order import (
    PairwiseCurve,
    dominance_scores,
    pairwise_curve,
    pairwise_curve_all_pairs,
    pairwise_curve_tail_pairs,
    reliable_breadth,
    tail_reliable_breadth,
)
from .snrcps import CertificationResult, certify_monotone_losses

__all__ = [
    "CertificationResult",
    "PairwiseCurve",
    "certify_monotone_losses",
    "dominance_scores",
    "pairwise_curve",
    "pairwise_curve_all_pairs",
    "pairwise_curve_tail_pairs",
    "reliable_breadth",
    "tail_reliable_breadth",
]

__version__ = "0.1.0"
