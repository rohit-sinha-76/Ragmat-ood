"""Concatenation fusion head for RAGMat-OOD.

Mean-pools retrieved neighbour embeddings, concatenates with query embedding,
and passes through an MLP to produce a property prediction.
"""

from __future__ import annotations

import json
import os
import torch
import torch.nn as nn
import urllib.request
from torch import Tensor


def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict) -> None:
    """Best-effort debug reporting for the retrieval silent failure session."""
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        ".dbg",
        "retrieval-silent-failure.env",
    )
    url = "http://127.0.0.1:7777/event"
    session_id = "retrieval-silent-failure"
    try:
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line.startswith("DEBUG_SERVER_URL="):
                        url = line.split("=", 1)[1]
                    elif line.startswith("DEBUG_SESSION_ID="):
                        session_id = line.split("=", 1)[1]
        payload = {
            "sessionId": session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
        }
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
        ).read()
    except Exception:
        pass


class ConcatFusionHead(nn.Module):
    """Retrieval-augmented fusion via concatenation + MLP.

    Architecture:
        mean_pool(neighbor_embeddings) → concat with query → MLP → prediction

    Args:
        embedding_dim: Dimension of query and neighbour embeddings.
        hidden_dim: Hidden dimension of the MLP.
        dropout_rate: Dropout probability in the MLP.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_embedding: Tensor,
        neighbor_embeddings: Tensor,
        neighbor_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute property prediction from query + retrieved neighbours.

        Args:
            query_embedding: Query graph embeddings ``(B, D)``.
            neighbor_embeddings: Retrieved neighbour embeddings ``(B, K, D)``.
            neighbor_mask: Optional boolean mask ``(B, K)`` — True = valid
                neighbour, False = padding.  If None, all neighbours are used.

        Returns:
            Property predictions ``(B, 1)``.
        """
        # #region debug-point C:concat-input-shapes
        _debug_report(
            "C",
            "ragmat/fusion/concat.py:forward",
            "[DEBUG] concat forward received tensors",
            {
                "query_shape": list(query_embedding.shape),
                "neighbor_shape": list(neighbor_embeddings.shape),
                "mask_shape": list(neighbor_mask.shape) if neighbor_mask is not None else None,
            },
        )
        # #endregion
        if neighbor_mask is not None:
            # Zero out padding positions before mean-pooling
            mask = neighbor_mask.unsqueeze(-1).float()  # (B, K, 1)
            neighbor_embeddings = neighbor_embeddings * mask
            n_valid = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
            mean_neighbors = neighbor_embeddings.sum(dim=1) / n_valid
        else:
            mean_neighbors = neighbor_embeddings.mean(dim=1)  # (B, D)

        combined = torch.cat([query_embedding, mean_neighbors], dim=-1)  # (B, 2D)
        # #region debug-point C:concat-output-shapes
        _debug_report(
            "C",
            "ragmat/fusion/concat.py:forward",
            "[DEBUG] concat forward combined tensors",
            {
                "mean_neighbor_shape": list(mean_neighbors.shape),
                "combined_shape": list(combined.shape),
                "embedding_dim": int(self.embedding_dim),
            },
        )
        # #endregion
        return self.mlp(combined)  # (B, 1)
