from __future__ import annotations
import logging
import numpy as np
logger = logging.getLogger(__name__)

class ExplainabilityModule:
    def __init__(self, top_k=10):
        self.top_k = top_k
    def explain(self, query_features, neighbor_ids, neighbor_features, neighbor_labels, query_id=None):
        q = query_features / (np.linalg.norm(query_features) + 1e-10)
        n = neighbor_features / (np.linalg.norm(neighbor_features, axis=1, keepdims=True) + 1e-10)
        cosine = (n @ q).astype(np.float32)
        return {
            "query_id": query_id,
            "top_k": self.top_k,
            "neighbor_ids": list(neighbor_ids[:self.top_k]),
            "neighbor_cosine_similarities": cosine[:self.top_k].tolist(),
            "neighbor_labels": list(neighbor_labels[:self.top_k]),
            "physical_relevance_score": float(np.mean(cosine[:self.top_k])),
        }
