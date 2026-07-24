"""Cross-attention fusion head for RAGMat-OOD.

Uses multi-head attention with query as Q and retrieved neighbours as K/V,
then concatenates with the query and passes through an MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CrossAttentionFusionHead(nn.Module):
    """Retrieval-augmented fusion via cross-attention + MLP.

    Architecture:
        Q = query_embedding.unsqueeze(1)   → (B, 1, D)
        K = V = neighbor_embeddings         → (B, K, D)
        attn_out = MHA(Q, K, V)             → (B, 1, D)
        concat(query, attn_out.squeeze(1))  → (B, 2D)
        MLP → (B, 1)

    Args:
        embedding_dim: Dimension of query and neighbour embeddings.
        n_heads: Number of attention heads (must divide embedding_dim).
        hidden_dim: Hidden dimension of the final MLP.
        dropout_rate: Dropout probability in the MLP.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int = 1,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            batch_first=True,
        )
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
    ) -> Tensor:
        """Compute property prediction via cross-attention.

        Args:
            query_embedding: Query embeddings ``(B, D)``.
            neighbor_embeddings: Retrieved neighbour embeddings ``(B, K, D)``.

        Returns:
            Property predictions ``(B, 1)``.
        """
        # Q: (B, 1, D) — query as a single sequence element
        q = query_embedding.unsqueeze(1)

        # Cross-attention: Q attends over K/V = neighbour embeddings
        attn_out, _ = self.attention(
            query=q,
            key=neighbor_embeddings,
            value=neighbor_embeddings,
        )  # (B, 1, D)

        attn_out = attn_out.squeeze(1)  # (B, D)
        combined = torch.cat([query_embedding, attn_out], dim=-1)  # (B, 2D)
        return self.mlp(combined)  # (B, 1)
