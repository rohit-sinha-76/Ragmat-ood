"""MC-Dropout uncertainty quantification for RAGMat-OOD.

Enables dropout at inference time, runs N stochastic forward passes,
and returns the mean and variance of predictions as uncertainty estimates.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


class MCDropoutUQ:
    """Monte Carlo Dropout uncertainty estimator.

    Runs ``n_passes`` stochastic forward passes with dropout enabled and
    returns mean + variance of predictions.
    """

    @staticmethod
    def predict_with_uncertainty(
        model: nn.Module,
        forward_fn: callable,
        n_passes: int = 30,
    ) -> tuple[Tensor, Tensor]:
        """Run MC-Dropout inference and return mean + variance.

        Sets the model to train mode (activates dropout), runs ``n_passes``
        forward passes, then restores eval mode.

        Args:
            model: A model with dropout layers (e.g., ``CGCNNEncoder`` or ``FusionHead``).
            forward_fn: A callable that takes no arguments and returns a prediction tensor.
            n_passes: Number of stochastic forward passes.

        Returns:
            Tuple ``(mean_pred, var_pred)`` where each is shape ``(N, 1)``:
            - ``mean_pred``: Mean of N stochastic predictions.
            - ``var_pred``: Variance of N stochastic predictions.
        """
        original_mode_training = model.training
        model.train()  # Enable dropout

        predictions = []
        with torch.no_grad():
            for _ in range(n_passes):
                pred = forward_fn()
                if isinstance(pred, tuple):
                    pred = pred[0]  # Extract prediction if it returns a tuple (e.g., base model)
                predictions.append(pred)

        # Restore original mode
        if not original_mode_training:
            model.eval()

        # Stack: (n_passes, N, 1) → stats over first dim
        stacked = torch.stack(predictions, dim=0)  # (n_passes, N, 1)
        mean_pred = stacked.mean(dim=0)            # (N, 1)
        var_pred = stacked.var(dim=0)              # (N, 1)

        logger.debug(
            "MCDropout: n_passes=%d, mean_uncertainty=%.6f",
            n_passes,
            float(var_pred.mean()),
        )
        return mean_pred, var_pred
