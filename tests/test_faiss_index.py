"""Test FAISS index."""
import numpy as np
import pytest
from ragmat.retrieval.faiss_index import FAISSIndex

def test_faiss_index():
    index = FAISSIndex(dim=64, property_name="prop", split_name="iid")
    embs = np.random.randn(100, 64).astype(np.float32)
    ids = [f"id_{i}" for i in range(100)]
    
    index.build(embs, ids)
    assert index._built
    
    q_embs = np.random.randn(5, 64).astype(np.float32)
    scores, res_ids = index.query(q_embs, top_k=10)
    
    assert scores.shape == (5, 10)
    assert len(res_ids) == 5
    assert len(res_ids[0]) == 10
    for r in res_ids:
        assert all(isinstance(i, str) for i in r)

def test_faiss_computes_cosine_similarity():
    """Verify that FAISS query computes cosine similarity via L2 normalization."""
    index = FAISSIndex(dim=2, property_name="prop", split_name="iid")
    # Vectors with different magnitudes but same direction should have score=1.0
    embs = np.array([[3.0, 4.0], [-1.0, 0.0]], dtype=np.float32)
    ids = ["a", "b"]
    index.build(embs, ids)
    
    q_embs = np.array([[6.0, 8.0]], dtype=np.float32) # Same direction as "a"
    scores, res_ids = index.query(q_embs, top_k=1)
    
    # Cosine similarity between [3,4] and [6,8] is 1.0.
    # If L2 normalization was skipped, it would be dot product (18+32 = 50.0).
    np.testing.assert_allclose(scores[0][0], 1.0, rtol=1e-5, err_msg="Score is not cosine similarity (likely L2 norm is missing)")
    assert res_ids[0][0] == "a"
