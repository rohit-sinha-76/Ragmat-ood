"""Adaptive gating module for RAGMat-OOD.

Decides whether to use retrieval-augmented prediction or fall back to
the base model, based on two signals:
1. OOD score (from Mahalanobis detector) — how far the query is from training.
2. Neighbour coherence (variance of neighbour property values) — how
   informative the retrieved neighbours are.

Retrieval is used when: ood_score < ood_threshold AND variance < coherence_threshold.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveGate:
    """Dual-threshold gate controlling retrieval fusion.

    Args:
        ood_threshold: If ood_score >= this value, skip retrieval (query is
            too far OOD for retrieved neighbours to be meaningful).
        coherence_threshold: If neighbour property variance >= this value,
            skip retrieval (retrieved neighbours are too incoherent).
    """

    def __init__(
        self,
        ood_threshold: float = 0.7,
        coherence_threshold: float = 0.5,
    ) -> None:
        self.ood_threshold = ood_threshold
        self.coherence_threshold = coherence_threshold

    def should_retrieve(
        self,
        ood_score: float,
        neighbor_property_variance: float,
    ) -> bool:
        """Decide whether to use retrieval for a single query.

        Args:
            ood_score: Normalised Mahalanobis OOD score in [0, 1].
            neighbor_property_variance: Variance of the target property
                values among the retrieved neighbours.

        Returns:
            ``True`` → use retrieval-augmented prediction.
            ``False`` → use base model prediction only.
        """
        return (
            ood_score < self.ood_threshold
            and neighbor_property_variance < self.coherence_threshold
        )

    def batch_gate(
        self,
        ood_scores: np.ndarray,
        neighbor_variances: np.ndarray,
    ) -> np.ndarray:
        """Compute gate decisions for a batch of queries.

        Args:
            ood_scores: OOD scores ``(N,)`` in [0, 1].
            neighbor_variances: Neighbour property variances ``(N,)``.

        Returns:
            Boolean array ``(N,)`` — True = use retrieval.
        """
        return (ood_scores < self.ood_threshold) & (
            neighbor_variances < self.coherence_threshold
        )

    def log_stats(self, gate_decisions: np.ndarray) -> None:
        """Log statistics about gate decisions."""
        n_total = len(gate_decisions)
        n_retrieve = gate_decisions.sum()
        logger.info(
            "AdaptiveGate: %d/%d queries use retrieval (%.1f%%)",
            n_retrieve,
            n_total,
            100.0 * n_retrieve / max(n_total, 1),
        )
