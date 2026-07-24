"""Mahalanobis OOD detector for RAGMat-OOD.

Fits a Gaussian model on training embeddings and scores test embeddings
by their Mahalanobis distance to the training distribution mean.
Scores are normalised to [0, 1] by the maximum training distance.

Higher score = more OOD.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


class MahalanobisDetector:
    """Mahalanobis distance-based OOD detector.

    Fits on training embeddings, scores test embeddings.
    Output scores are in [0, 1] — higher means more OOD.
    """

    def __init__(self, threshold_percentile: float = 95.0) -> None:
        self._mean: np.ndarray | None = None
        self._precision: np.ndarray | None = None
        self._threshold: float = 1.0
        self._threshold_percentile = threshold_percentile
        self._fitted: bool = False

    def fit(self, train_embeddings: np.ndarray) -> None:
        """Fit the detector on training embeddings.

        Computes the mean and precision matrix (inverse covariance) of the
        training distribution. Uses ``np.linalg.pinv`` for numerical stability.

        Also computes the maximum Mahalanobis distance on the training set,
        which is used to normalise scores to [0, 1].

        Args:
            train_embeddings: Training embeddings ``(N, D)``.
        """
        # L2-normalize embeddings as per spec
        norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        train_embeddings = train_embeddings / np.clip(norms, 1e-12, None)

        self._mean = train_embeddings.mean(axis=0)
        cov = np.cov(train_embeddings, rowvar=False)
        
        # Add regularization to handle singular covariance matrix
        # This addresses OD3: singular covariance from zero-variance columns
        # Use stronger regularization (1e-5) for high-dimensional features
        reg = 1e-5
        cov = cov + reg * np.eye(cov.shape[0])
        
        self._precision = np.linalg.pinv(cov)

        # Compute threshold based on percentile of training distances
        # This addresses OD2: threshold calibration for OOD detection
        train_dists = self._compute_distances(train_embeddings)
        self._threshold = float(np.percentile(train_dists, self._threshold_percentile))
        self._max_train_dist = float(train_dists.max()) if train_dists.max() > 0 else 1.0
        self._fitted = True

        logger.info(
            "MahalanobisDetector fitted: n_train=%d, dim=%d, threshold=%.4f (p%d), max_train_dist=%.4f, reg=%.2e",
            len(train_embeddings),
            train_embeddings.shape[1],
            self._threshold,
            int(self._threshold_percentile),
            self._max_train_dist,
            reg,
        )

    @property
    def normalized_threshold(self) -> float:
        """The 95th percentile threshold in the normalised [0, 1] score space."""
        if not self._fitted:
            raise RuntimeError("Detector not fitted.")
        return self._threshold / self._max_train_dist

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute normalised OOD scores for a batch of embeddings.

        Args:
            embeddings: Test embeddings ``(M, D)``.

        Returns:
            OOD scores ``(M,)`` in [0, 1] (relative to training max).
            Higher = more OOD.

        Raises:
            RuntimeError: If the detector has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("MahalanobisDetector not fitted. Call fit() first.")

        # L2-normalize embeddings as per spec
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        dists = self._compute_distances(embeddings)
        # Normalise by maximum training distance per spec
        scores = dists / self._max_train_dist
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compute_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Vectorised Mahalanobis distance computation.

        Args:
            embeddings: ``(M, D)`` array.

        Returns:
            Distance array ``(M,)``.
        """
        diff = embeddings - self._mean  # (M, D)
        # Mahalanobis: sqrt( diff @ precision @ diff.T )
        # Vectorised: (M, D) @ (D, D) @ (D, M) → diagonal
        temp = diff @ self._precision  # (M, D)
        sq_dists = (temp * diff).sum(axis=1)  # (M,)
        # Clamp negatives from numerical noise
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.sqrt(sq_dists)
