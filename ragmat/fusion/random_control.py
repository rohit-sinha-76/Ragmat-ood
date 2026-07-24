"""Random retrieval control fusion head for RAGMat-OOD.

Wraps a base fusion head (concat or cross-attention) and substitutes
random train-partition samples for FAISS top-k neighbours — at BOTH
training and inference time.

CRITICAL: This must be a SEPARATELY TRAINED model. It is NOT the same
model as the true-neighbour model with inputs swapped at eval time.
Training a separate model ensures that the model parameters and the
input distribution are co-adapted to random retrieval, giving a fair
baseline that does not confound parameters with inputs.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class RandomRetrievalFusionHead(nn.Module):
    """Fusion head that uses random train samples instead of FAISS neighbours.

    This is the random-retrieval control model. It must be TRAINED from
    scratch with random neighbours — not reused from the true-neighbour model.

    Args:
        base_fusion_head: A ``ConcatFusionHead`` or ``CrossAttentionFusionHead``
            instance to wrap. Should be freshly initialised.
        train_embeddings_pool: All training embeddings ``(N_train, D)``
            to sample random neighbours from.
        top_k: Number of random neighbours to sample per query.
    """

    def __init__(
        self,
        base_fusion_head: nn.Module,
        train_embeddings_pool: np.ndarray,
        top_k: int = 10,
    ) -> None:
        super().__init__()
        self.base_fusion_head = base_fusion_head
        self.top_k = top_k
        # Register as a buffer so it moves to the right device automatically
        self.register_buffer(
            "_pool",
            torch.from_numpy(train_embeddings_pool.astype(np.float32)),
        )
        logger.info(
            "RandomRetrievalFusionHead: pool_size=%d, top_k=%d (RANDOM MODE ACTIVE)",
            len(train_embeddings_pool),
            top_k,
        )

    def forward(
        self,
        query_embedding: Tensor,
        neighbor_embeddings: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Sample random neighbours and pass to the base fusion head.

        The ``neighbor_embeddings`` argument is IGNORED — random samples
        from the training pool are always used instead.

        Args:
            query_embedding: Query embeddings ``(B, D)``.
            neighbor_embeddings: Ignored. Present for API compatibility.
            **kwargs: Forwarded to base fusion head (e.g. ``neighbor_mask``).

        Returns:
            Property predictions ``(B, 1)``.
        """
        B = query_embedding.shape[0]
        pool_size = self._pool.shape[0]

        # Sample k random indices (without replacement if pool is large enough)
        if pool_size >= self.top_k:
            idx = torch.randperm(pool_size, device=self._pool.device)[: self.top_k]
            rand_neighbors = self._pool[idx].unsqueeze(0).expand(B, -1, -1)
        else:
            # Fall back to sampling with replacement when pool is tiny
            idx = torch.randint(0, pool_size, (B, self.top_k), device=self._pool.device)
            rand_neighbors = self._pool[idx]  # (B, top_k, D)

        return self.base_fusion_head(query_embedding, rand_neighbors, **kwargs)
