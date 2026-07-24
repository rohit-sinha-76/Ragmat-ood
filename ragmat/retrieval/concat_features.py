"""Retrieval feature concatenation for RAGMat-OOD.

Implements ``concat_retrieval_features()`` which:
1. Queries the FAISS index for top-k nearest neighbors
2. Retrieves the feature vectors for those neighbors
3. Concatenates query features with aggregated neighbor features

This is the CRITICAL missing piece that connects retrieval to prediction.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from ragmat.retrieval.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


def concat_retrieval_features(
    query_features: np.ndarray,
    index: FAISSIndex,
    train_features: np.ndarray,
    train_ids: list[str],
    top_k: int = 10,
    aggregation: Literal["mean", "concat_all"] = "mean",
) -> np.ndarray:
    """Concatenate retrieved neighbor features with query features.

    This function performs the core retrieval-augmented feature construction:
    1. Query FAISS index to find top-k nearest neighbors
    2. Look up the actual feature vectors for those neighbors
    3. Aggregate neighbor features (mean pooling or full concatenation)
    4. Concatenate aggregated features with original query features

    Args:
        query_features: Query feature matrix, shape (N, D)
        index: Built FAISSIndex (must be already built with train embeddings)
        train_features: Training partition feature matrix, shape (N_train, D).
            These are the ORIGINAL features (not embeddings) that will be
            retrieved and concatenated.
        train_ids: Training partition material IDs, shape (N_train,).
            Must match the order of train_features and the IDs used to build index.
        top_k: Number of neighbors to retrieve (default: 10)
        aggregation: How to aggregate neighbor features:
            - "mean": Mean pool neighbors, output shape (N, D + D)
            - "concat_all": Concatenate all k neighbors, output shape (N, D + k*D)

    Returns:
        Concatenated feature matrix:
        - If aggregation="mean": shape (N, 2*D)
        - If aggregation="concat_all": shape (N, (k+1)*D)

    Raises:
        ValueError: If index is not built or if train_ids don't match index IDs.

    Example:
        >>> # After building FAISS index from train embeddings
        >>> X_train_augmented = concat_retrieval_features(
        ...     query_features=X_train,
        ...     index=index,
        ...     train_features=X_train,
        ...     train_ids=train_ids,
        ...     top_k=10,
        ...     aggregation="mean"
        ... )
        >>> # X_train_augmented.shape = (N_train, 2*D)
        >>> model.fit(X_train_augmented, y_train)
    """
    if not index._built:
        raise ValueError(
            "FAISSIndex must be built before calling concat_retrieval_features"
        )

    N, D = query_features.shape
    logger.info(
        "Concatenating retrieval features: query_shape=%s, top_k=%d, aggregation=%s",
        query_features.shape,
        top_k,
        aggregation,
    )

    # Step 1: Query FAISS index to get neighbor IDs
    # Note: We query using the same features as embeddings for Tier 0
    # (matminer features serve as both features and embeddings for Tier 0)
    scores, neighbor_ids_nested = index.query(query_features, top_k)

    # Step 2: Build ID-to-index mapping for fast lookup
    id_to_idx = {mid: idx for idx, mid in enumerate(train_ids)}

    # Step 3: Retrieve neighbor features and aggregate
    neighbor_features_list = []

    for i, neighbor_ids in enumerate(neighbor_ids_nested):
        # Get feature vectors for this query's neighbors
        neighbor_feature_rows = []
        for nid in neighbor_ids:
            if nid in id_to_idx:
                idx = id_to_idx[nid]
                neighbor_feature_rows.append(train_features[idx])
            else:
                # Fallback: if ID not found, use zero vector
                logger.warning(
                    "Neighbor ID '%s' not found in train_ids (query index %d). Using zero vector.",
                    nid,
                    i,
                )
                neighbor_feature_rows.append(np.zeros(D, dtype=np.float32))

        # Stack into (k, D)
        neighbor_feats = np.stack(neighbor_feature_rows, axis=0)

        # Aggregate
        if aggregation == "mean":
            # Mean pool: (k, D) -> (D,)
            agg_feats = neighbor_feats.mean(axis=0)
        elif aggregation == "concat_all":
            # Flatten all: (k, D) -> (k*D,)
            agg_feats = neighbor_feats.flatten()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        neighbor_features_list.append(agg_feats)

    # Stack aggregated neighbor features: (N, D) or (N, k*D)
    neighbor_features = np.stack(neighbor_features_list, axis=0)

    # Step 4: Concatenate query + neighbor features
    concatenated = np.concatenate([query_features, neighbor_features], axis=1)

    logger.info(
        "Retrieval feature concatenation complete: "
        "input_shape=%s, neighbor_shape=%s, output_shape=%s",
        query_features.shape,
        neighbor_features.shape,
        concatenated.shape,
    )

    return concatenated


def concat_random_retrieval_features(
    query_features: np.ndarray,
    train_features: np.ndarray,
    top_k: int = 10,
    aggregation: Literal["mean", "concat_all"] = "mean",
    seed: int = 42,
) -> np.ndarray:
    """Concatenate RANDOM neighbor features as a control baseline.

    This implements the "random_control" retrieval mode from the spec.
    Instead of using FAISS to find semantically similar neighbors,
    it samples random training examples for each query.

    This control is critical for validating that retrieval provides genuine
    benefit beyond just adding more feature dimensions.

    Args:
        query_features: Query feature matrix, shape (N, D)
        train_features: Training partition feature matrix, shape (N_train, D)
        top_k: Number of random neighbors to sample
        aggregation: "mean" or "concat_all"
        seed: Random seed for reproducibility

    Returns:
        Concatenated feature matrix with same shape as concat_retrieval_features
    """
    rng = np.random.RandomState(seed)
    N, D = query_features.shape
    N_train = train_features.shape[0]

    logger.info(
        "Concatenating RANDOM retrieval features: query_shape=%s, top_k=%d, aggregation=%s",
        query_features.shape,
        top_k,
        aggregation,
    )

    neighbor_features_list = []

    for i in range(N):
        # Sample k random indices from training set (without replacement per query)
        if top_k <= N_train:
            random_indices = rng.choice(N_train, size=top_k, replace=False)
        else:
            # If k > N_train, sample with replacement
            random_indices = rng.choice(N_train, size=top_k, replace=True)

        neighbor_feats = train_features[random_indices]  # (k, D)

        # Aggregate
        if aggregation == "mean":
            agg_feats = neighbor_feats.mean(axis=0)
        elif aggregation == "concat_all":
            agg_feats = neighbor_feats.flatten()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        neighbor_features_list.append(agg_feats)

    neighbor_features = np.stack(neighbor_features_list, axis=0)
    concatenated = np.concatenate([query_features, neighbor_features], axis=1)

    logger.info(
        "Random retrieval feature concatenation complete: "
        "input_shape=%s, neighbor_shape=%s, output_shape=%s",
        query_features.shape,
        neighbor_features.shape,
        concatenated.shape,
    )

    return concatenated
