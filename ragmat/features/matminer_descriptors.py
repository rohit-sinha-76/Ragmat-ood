"""Matminer composition + structure feature extraction for RAGMat-OOD (Tier 0).

Implements ``MatminerFeaturizer`` which:
- Computes composition features with ElementProperty (magpie preset), dim=132.
- Computes structure features via CrystalNNFingerprint → SiteStatsFingerprint.
- Uses ``error_mode='return_nan'`` and drops/logs failed materials.
- Fits ``StandardScaler`` on train partition ONLY.

Critical rules:
- ``fit_scaler`` is NEVER called on the full dataset.
- NaN rows are dropped and their IDs recorded.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

# Composition feature dimension (ElementProperty magpie)
COMPOSITION_DIM = 132
# Structure feature dimension varies but is consistent within a run
STRUCTURE_DIM_APPROX = 150
TOTAL_FEATURE_DIM_APPROX = COMPOSITION_DIM + STRUCTURE_DIM_APPROX


class MatminerFeaturizer:
    """Composition + structure feature extractor using matminer.

    Args:
        n_jobs: Number of parallel featurization workers.
    """

    def __init__(self, n_jobs: int = -1) -> None:
        self.n_jobs = n_jobs
        self._comp_featurizer = None
        self._struct_featurizer = None
        self._feature_labels: Optional[list[str]] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def featurize_dataset(
        self,
        structures: list[Structure],
        ids: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Featurize a list of structures and return a feature matrix.

        Runs composition + structure featurizers, concatenates features,
        drops rows with NaN values, and logs skipped material IDs.

        Args:
            structures: List of pymatgen ``Structure`` objects.
            ids: Corresponding material IDs (same length as ``structures``).

        Returns:
            Tuple of ``(X, valid_ids)`` where ``X`` has shape ``(N, feature_dim)``
            and ``valid_ids`` lists the IDs corresponding to each row.
        """
        self._init_featurizers()
        n = len(structures)
        assert len(ids) == n, "structures and ids must have the same length"

        import hashlib, pickle
        import os
        from pathlib import Path
        cache_dir = Path(__file__).parent.parent.parent / "data" / "features"
        cache_dir.mkdir(parents=True, exist_ok=True)
        id_str = "\n".join(ids).encode('utf-8')
        cache_hash = hashlib.md5(id_str).hexdigest()
        cache_file = cache_dir / f"matminer_cache_{cache_hash}.pkl"
        
        if cache_file.exists():
            logger.info("Loading cached features from %s", cache_file)
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        logger.info("Featurizing %d structures ...", n)

        # ── Composition features ────────────────────────────────────────
        comp_rows = self._run_composition(structures, ids)

        # ── Structure features ──────────────────────────────────────────
        struct_rows = self._run_structure(structures, ids)

        # ── Concatenate and drop NaN rows ───────────────────────────────
        comp_df = pd.DataFrame(comp_rows, index=ids)
        struct_df = pd.DataFrame(struct_rows, index=ids)
        combined = pd.concat([comp_df, struct_df], axis=1)

        nan_mask = combined.isnull().any(axis=1)
        n_dropped = nan_mask.sum()
        if n_dropped:
            dropped_ids = combined.index[nan_mask].tolist()
            logger.warning(
                "Dropping %d/%d materials with NaN features: %s%s",
                n_dropped,
                n,
                dropped_ids[:10],
                " ..." if len(dropped_ids) > 10 else "",
            )
        valid_df = combined[~nan_mask]
        valid_ids = valid_df.index.tolist()
        X = valid_df.values.astype(np.float32)

        logger.info(
            "Featurized %d materials, feature dim=%d (dropped %d)",
            len(valid_ids),
            X.shape[1],
            n_dropped,
        )
        
        with open(cache_file, "wb") as f:
            pickle.dump((X, valid_ids), f)
            
        return X, valid_ids

    def fit_scaler(self, X_train: np.ndarray):
        """Fit a ``StandardScaler`` on the training partition ONLY.

        Args:
            X_train: Feature matrix of shape ``(N_train, feature_dim)``.

        Returns:
            Fitted ``sklearn.preprocessing.StandardScaler``.
        """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(X_train)
        logger.info(
            "Scaler fitted on %d train samples, feature_dim=%d",
            X_train.shape[0],
            X_train.shape[1],
        )
        return scaler

    @staticmethod
    def transform(X: np.ndarray, scaler) -> np.ndarray:
        """Apply a fitted scaler to a feature matrix.

        Args:
            X: Feature matrix of shape ``(N, feature_dim)``.
            scaler: Fitted ``StandardScaler``.

        Returns:
            Normalised feature matrix of the same shape.
        """
        return scaler.transform(X).astype(np.float32)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _init_featurizers(self) -> None:
        """Lazily initialise matminer featurizers (import cost is non-trivial)."""
        if self._comp_featurizer is not None:
            return

        from matminer.featurizers.composition import ElementProperty
        from matminer.featurizers.site import CrystalNNFingerprint
        from matminer.featurizers.structure import SiteStatsFingerprint

        self._comp_featurizer = ElementProperty.from_preset("magpie")
        self._comp_featurizer.set_n_jobs(self.n_jobs)

        cnn_fp = CrystalNNFingerprint.from_preset("ops")
        self._struct_featurizer = SiteStatsFingerprint(
            site_featurizer=cnn_fp,
            stats=["mean", "std_dev", "minimum", "maximum"],
        )
        self._struct_featurizer.set_n_jobs(self.n_jobs)

    def _run_composition(
        self, structures: list[Structure], ids: list[str]
    ) -> list[dict]:
        """Extract composition features in parallel."""
        from joblib import Parallel, delayed
        from pymatgen.core import Composition

        labels = self._comp_featurizer.feature_labels()

        def _feat_one(s, sid):
            try:
                comp = s.composition
                vals = self._comp_featurizer.featurize(comp)
                return dict(zip(labels, vals))
            except Exception as exc:
                logger.debug("Composition featurization failed for %s: %s", sid, exc)
                return {lbl: np.nan for lbl in labels}

        rows = Parallel(n_jobs=self.n_jobs)(
            delayed(_feat_one)(s, sid) for s, sid in zip(structures, ids)
        )
        return rows

    def _run_structure(
        self, structures: list[Structure], ids: list[str]
    ) -> list[dict]:
        """Extract structure features in parallel."""
        from joblib import Parallel, delayed

        labels = self._struct_featurizer.feature_labels()

        def _feat_one(s, sid):
            try:
                vals = self._struct_featurizer.featurize(s)
                return dict(zip(labels, vals))
            except Exception as exc:
                logger.debug("Structure featurization failed for %s: %s", sid, exc)
                return {lbl: np.nan for lbl in labels}

        rows = Parallel(n_jobs=self.n_jobs)(
            delayed(_feat_one)(s, sid) for s, sid in zip(structures, ids)
        )
        return rows
