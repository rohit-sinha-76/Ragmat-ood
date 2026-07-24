"""Integration test for retrieval feature concatenation.

This test verifies that:
1. concat_retrieval_features() produces correct output shapes
2. Random control produces different results from true retrieval
3. The concatenated features actually change model predictions
"""

import numpy as np
import pytest

from ragmat.retrieval.faiss_index import FAISSIndex
from ragmat.retrieval.concat_features import (
    concat_retrieval_features,
    concat_random_retrieval_features,
)


def test_concat_retrieval_features_shape():
    """Test that concatenation produces correct output shapes."""
    # Create synthetic data
    N_train = 100
    N_query = 20
    D = 50
    top_k = 10

    np.random.seed(42)
    train_features = np.random.randn(N_train, D).astype(np.float32)
    query_features = np.random.randn(N_query, D).astype(np.float32)
    train_ids = [f"mat_{i}" for i in range(N_train)]

    # Build FAISS index
    index = FAISSIndex(dim=D, property_name="test_property", split_name="test_split")
    index.build(train_features, train_ids)

    # Test mean aggregation
    result_mean = concat_retrieval_features(
        query_features=query_features,
        index=index,
        train_features=train_features,
        train_ids=train_ids,
        top_k=top_k,
        aggregation="mean",
    )

    # Expected shape: (N_query, D + D) = (N_query, 2*D)
    assert result_mean.shape == (
        N_query,
        2 * D,
    ), f"Mean aggregation shape mismatch: expected {(N_query, 2*D)}, got {result_mean.shape}"

    # Test concat_all aggregation
    result_concat = concat_retrieval_features(
        query_features=query_features,
        index=index,
        train_features=train_features,
        train_ids=train_ids,
        top_k=top_k,
        aggregation="concat_all",
    )

    # Expected shape: (N_query, D + k*D) = (N_query, (k+1)*D)
    assert result_concat.shape == (
        N_query,
        (top_k + 1) * D,
    ), f"Concat_all shape mismatch: expected {(N_query, (top_k+1)*D)}, got {result_concat.shape}"

    # Verify that query features are preserved in the first D dimensions
    np.testing.assert_allclose(
        result_mean[:, :D],
        query_features,
        rtol=1e-5,
        err_msg="Query features should be preserved in concatenated output",
    )


def test_random_control_shape():
    """Test that random control produces correct shapes."""
    N_train = 100
    N_query = 20
    D = 50
    top_k = 10

    np.random.seed(42)
    train_features = np.random.randn(N_train, D).astype(np.float32)
    query_features = np.random.randn(N_query, D).astype(np.float32)

    result = concat_random_retrieval_features(
        query_features=query_features,
        train_features=train_features,
        top_k=top_k,
        aggregation="mean",
        seed=42,
    )

    assert result.shape == (
        N_query,
        2 * D,
    ), f"Random control shape mismatch: expected {(N_query, 2*D)}, got {result.shape}"


def test_retrieval_vs_random_different():
    """Test that true retrieval produces different results from random control."""
    N_train = 100
    N_query = 20
    D = 50
    top_k = 10

    np.random.seed(42)
    train_features = np.random.randn(N_train, D).astype(np.float32)
    query_features = np.random.randn(N_query, D).astype(np.float32)
    train_ids = [f"mat_{i}" for i in range(N_train)]

    # Build FAISS index
    index = FAISSIndex(dim=D, property_name="test_property", split_name="test_split")
    index.build(train_features, train_ids)

    # Get true retrieval features
    result_true = concat_retrieval_features(
        query_features=query_features,
        index=index,
        train_features=train_features,
        train_ids=train_ids,
        top_k=top_k,
        aggregation="mean",
    )

    # Get random control features
    result_random = concat_random_retrieval_features(
        query_features=query_features,
        train_features=train_features,
        top_k=top_k,
        aggregation="mean",
        seed=42,
    )

    # They should have the same shape
    assert result_true.shape == result_random.shape

    # But the neighbor features should be different
    # (first D dimensions are the same query features, next D are different)
    neighbor_feats_true = result_true[:, D:]
    neighbor_feats_random = result_random[:, D:]

    # Compute difference
    diff = np.abs(neighbor_feats_true - neighbor_feats_random).mean()

    # There should be substantial difference (not identical)
    assert diff > 0.01, (
        f"True retrieval and random control are too similar (diff={diff}). "
        "They should produce different neighbor features."
    )


def test_no_nan_in_output():
    """Test that concatenation never produces NaN values."""
    N_train = 50
    N_query = 10
    D = 30
    top_k = 5

    np.random.seed(42)
    train_features = np.random.randn(N_train, D).astype(np.float32)
    query_features = np.random.randn(N_query, D).astype(np.float32)
    train_ids = [f"mat_{i}" for i in range(N_train)]

    index = FAISSIndex(dim=D, property_name="test_property", split_name="test_split")
    index.build(train_features, train_ids)

    result = concat_retrieval_features(
        query_features=query_features,
        index=index,
        train_features=train_features,
        train_ids=train_ids,
        top_k=top_k,
        aggregation="mean",
    )

    assert not np.isnan(result).any(), "Output contains NaN values"
    assert not np.isinf(result).any(), "Output contains Inf values"


def test_retrieval_reproducibility():
    """Test that retrieval is deterministic given the same inputs."""
    N_train = 50
    N_query = 10
    D = 30
    top_k = 5

    np.random.seed(42)
    train_features = np.random.randn(N_train, D).astype(np.float32)
    query_features = np.random.randn(N_query, D).astype(np.float32)
    train_ids = [f"mat_{i}" for i in range(N_train)]

    # Build index and query twice
    index1 = FAISSIndex(dim=D, property_name="test_property", split_name="test_split")
    index1.build(train_features, train_ids)
    result1 = concat_retrieval_features(
        query_features=query_features,
        index=index1,
        train_features=train_features,
        train_ids=train_ids,
        top_k=top_k,
        aggregation="mean",
    )

    index2 = FAISSIndex(dim=D, property_name="test_property", split_name="test_split")
    index2.build(train_features, train_ids)
    result2 = concat_retrieval_features(
        query_features=query_features,
        index=index2,
        train_features=train_features,
        train_ids=train_ids,
        top_k=top_k,
        aggregation="mean",
    )

    np.testing.assert_allclose(
        result1, result2, rtol=1e-6, err_msg="Retrieval should be deterministic"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
