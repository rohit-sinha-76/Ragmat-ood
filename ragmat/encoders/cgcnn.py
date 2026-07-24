"""CGCNN encoder for RAGMat-OOD (Tier 1).

Implements the Crystal Graph Convolutional Neural Network from:
    Xie & Grossman, Physical Review Letters 120, 145301 (2018).

CRITICAL: This class has NO from_pretrained() method and NO pretrained
weight loading logic. It can only be instantiated fresh. Attempting to
load any external checkpoint via this class is explicitly prevented.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, global_mean_pool

logger = logging.getLogger(__name__)


class CGCNNLayer(MessagePassing):
    """One crystal graph convolution layer (Xie & Grossman 2018).

    Implements the gated graph convolution:
        z_ij = Linear([h_i || h_j || e_ij]) → Softplus   (gate logits)
        h_ij  = sigmoid(z_ij[:D]) * tanh(z_ij[D:])       (gated update)
        h_i'  = BatchNorm(h_i + Σ_j h_ij)                (residual)

    Args:
        hidden_dim: Hidden node feature dimension.
        edge_dim: Edge feature dimension (Gaussian basis size).
    """

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # Message network: [h_i || h_j || e_ij] → 2*hidden_dim
        self.msg_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, 2 * hidden_dim),
            nn.Softplus(),
        )
        # Update: gated activation → hidden
        self.update_linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.use_bn = False

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Compute one layer of crystal graph convolution.

        Args:
            x: Node features ``(N, hidden_dim)``.
            edge_index: Edge connectivity ``(2, E)``.
            edge_attr: Edge features ``(E, edge_dim)``.

        Returns:
            Updated node features ``(N, hidden_dim)``.
        """
        # Residual connection
        # Pass x as tuple (x, x) for homogeneous graph (same source and target nodes)
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        norm_layer = self.bn if self.use_bn else self.ln
        return norm_layer(x + out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """Compute messages from neighbours.

        Args:
            x_i: Source node features ``(E, hidden_dim)``.
            x_j: Target node features ``(E, hidden_dim)``.
            edge_attr: Edge features ``(E, edge_dim)``.

        Returns:
            Gated messages ``(E, hidden_dim)``.
        """
        z = self.msg_net(torch.cat([x_i, x_j, edge_attr], dim=-1))
        # Gating: sigmoid * tanh
        gate = torch.sigmoid(z[:, : self.hidden_dim])
        content = torch.tanh(z[:, self.hidden_dim :])
        return gate * content


class CGCNNEncoder(nn.Module):
    """CGCNN graph encoder producing property predictions and embeddings.

    Architecture:
        1. Linear embedding: node_dim (92) → hidden_dim
        2. ``n_conv_layers`` × CGCNNLayer + BatchNorm
        3. global_mean_pool → graph embedding (hidden_dim,)
        4. Prediction head: MLP → scalar prediction

    CRITICAL: No ``from_pretrained()`` method exists. No external checkpoint
    is ever loaded. Always initialised from random weights.

    Args:
        node_dim: Input node feature dimension (92 for H→U one-hot).
        edge_dim: Edge feature dimension (Gaussian basis size, 40).
        hidden_dim: Hidden layer dimension.
        n_conv_layers: Number of CGCNNLayer blocks.
        dropout_rate: Dropout probability in the prediction head.
    """

    def __init__(
        self,
        node_dim: int = 92,
        edge_dim: int = 40,
        hidden_dim: int = 64,
        n_conv_layers: int = 3,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Node embedding
        self.embedding = nn.Linear(node_dim, hidden_dim)

        # Convolution layers
        self.conv_layers = nn.ModuleList(
            [CGCNNLayer(hidden_dim, edge_dim) for _ in range(n_conv_layers)]
        )

        # Prediction head: hidden_dim → hidden_dim//2 → 1
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ── No from_pretrained ───────────────────────────────────────────────────
    # This class intentionally has NO from_pretrained(), load_pretrained(),
    # or any method that loads weights from an external source.
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, data: Batch) -> tuple[Tensor, Tensor]:
        """Run a forward pass and return prediction + embedding.

        Args:
            data: PyG ``Batch`` containing ``x``, ``edge_index``, ``edge_attr``,
                ``batch`` tensors.

        Returns:
            Tuple ``(pred, embedding)`` where:
            - ``pred``: property predictions of shape ``(N_graphs, 1)``.
            - ``embedding``: graph-level embeddings of shape ``(N_graphs, hidden_dim)``
              (pooled BEFORE the prediction head — used for FAISS indexing).
        """
        x = self.embedding(data.x)  # (N_atoms, hidden_dim)

        for conv in self.conv_layers:
            x = conv(x, data.edge_index, data.edge_attr)

        # Global mean pool → graph embedding
        embedding = global_mean_pool(x, data.batch)  # (N_graphs, hidden_dim)

        # Prediction head
        pred = self.head(embedding)  # (N_graphs, 1)
        return pred, embedding

    def get_embedding(self, data: Batch) -> Tensor:
        """Extract graph-level embeddings BEFORE the prediction head.

        Used for FAISS indexing. Embeddings should be L2-normalised
        externally before adding to the index.

        Args:
            data: PyG ``Batch``.

        Returns:
            Embeddings of shape ``(N_graphs, hidden_dim)``.
        """
        with torch.no_grad():
            _, embedding = self.forward(data)
        return embedding

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Override to dynamically configure layer types (LN vs BN) based on state_dict keys."""
        # Detect if BN was actually used. A model trained with LN will have BN parameters
        # present in the checkpoint (due to module initialization) but they will be at 
        # default/initial values (running_mean all zeros, running_var all ones).
        has_bn = False
        running_mean = state_dict.get("conv_layers.0.bn.running_mean")
        if running_mean is not None:
            # If running_mean has non-zero elements or running_var has non-one elements, BN was active.
            running_var = state_dict.get("conv_layers.0.bn.running_var")
            is_default = torch.all(running_mean == 0.0) and (running_var is None or torch.all(running_var == 1.0))
            if not is_default:
                has_bn = True

        for conv in self.conv_layers:
            conv.use_bn = has_bn
        # Load keys. strict=False ignores keys of the inactive normalization layers
        return super().load_state_dict(state_dict, strict=False)
