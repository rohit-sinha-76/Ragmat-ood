"""Test ExplainabilityModule."""
import numpy as np
from ragmat.explain import ExplainabilityModule

def test_explainability_cosine():
    """Verify the physical relevance score is exact cosine similarity average."""
    ex = ExplainabilityModule(top_k=2)
    q = np.array([3.0, 4.0]) # norm 5
    n = np.array([[6.0, 8.0], [0.0, -1.0]]) # norms 10, 1
    # cos(q, n[0]) = 1.0. cos(q, n[1]) = (3*0 + 4*-1)/(5*1) = -0.8
    # Average = (1.0 - 0.8) / 2 = 0.1
    res = ex.explain(q, ["a", "b"], n, [1.0, 2.0], query_id="q1")
    np.testing.assert_allclose(res["physical_relevance_score"], 0.1, atol=1e-5)
