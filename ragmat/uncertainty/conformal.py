"""Conformal prediction for regression in RAGMat-OOD (RAPS implementation).

Post-hoc conformal prediction calibrated on the validation set.
Provides marginal coverage guarantee at the specified level.

Method: Residual Adaptive Prediction Sets for regression.
Nonconformity score: |y_true - y_pred|
Calibrated interval half-width: (1-alpha)*(1+1/n_cal) quantile of scores.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """Post-hoc conformal prediction for regression.

    Calibrated on the validation set. Coverage is guaranteed at the
    marginal level (i.e., on average across test points).

    Args:
        None — state is set during ``calibrate()``.
    """

    def __init__(self) -> None:
        self._half_width: float | None = None
        self._n_cal: int = 0
        self._coverage: float = 0.9
        self._calibrated: bool = False

    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        coverage: float = 0.9,
    ) -> None:
        """Calibrate the conformal predictor on the validation set.

        Computes nonconformity scores |y_true - y_pred| on the validation
        set and stores the (1-coverage)*(1+1/n_cal) quantile as the
        prediction interval half-width.

        Args:
            model: Trained model (should be in eval mode).
            val_loader: DataLoader for the validation partition.
            coverage: Target marginal coverage level (e.g., 0.9 for 90%).
        """
        model.eval()
        self._coverage = coverage

        scores = []
        with torch.no_grad():
            for batch in val_loader:
                device = next(model.parameters()).device
                batch = batch.to(device)
                pred, _ = model(batch)
                y_true = batch.y.view(-1, 1)
                nonconformity = (y_true - pred).abs()  # (N, 1)
                scores.append(nonconformity.cpu())

        all_scores = torch.cat(scores, dim=0).squeeze()  # (n_cal,)
        self._n_cal = len(all_scores)

        # Conformal quantile: ceil((n+1)(1-alpha)) / n percentile
        alpha = 1.0 - coverage
        level = (1.0 - alpha) * (1.0 + 1.0 / self._n_cal)
        level = min(level, 1.0)
        self._half_width = float(torch.quantile(all_scores, level))
        self._calibrated = True

        logger.info(
            "ConformalPredictor calibrated: coverage=%.2f, n_cal=%d, "
            "half_width=%.6f",
            coverage,
            self._n_cal,
            self._half_width,
        )

    def predict_interval(
        self, predictions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return conformal prediction intervals around point predictions.

        Args:
            predictions: Point predictions ``(N, 1)`` or ``(N,)``.

        Returns:
            Tuple ``(lower_bound, upper_bound)`` each of shape ``(N, 1)``.

        Raises:
            RuntimeError: If ``calibrate()`` has not been called.
        """
        if not self._calibrated:
            raise RuntimeError(
                "ConformalPredictor not calibrated. Call calibrate() first."
            )
        pred = predictions.view(-1, 1)
        hw = torch.tensor(self._half_width, device=pred.device)
        return pred - hw, pred + hw

    @property
    def half_width(self) -> float | None:
        """Calibrated interval half-width."""
        return self._half_width


class SklearnConformalPredictor:
    """Post-hoc conformal prediction for scikit-learn/numpy models (Tier 0)."""
    
    def __init__(self) -> None:
        self._half_width: float | None = None
        self._n_cal: int = 0
        self._coverage: float = 0.9
        self._calibrated: bool = False

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.9) -> None:
        """Calibrate on the validation set numpy arrays."""
        self._coverage = coverage
        y_t = y_true.reshape(-1)
        y_p = y_pred.reshape(-1)
        scores = np.abs(y_t - y_p)
        self._n_cal = len(scores)
        
        alpha = 1.0 - coverage
        level = (1.0 - alpha) * (1.0 + 1.0 / self._n_cal)
        level = min(level, 1.0)
        self._half_width = float(np.quantile(scores, level))
        self._calibrated = True
        logger.info(
            "SklearnConformalPredictor calibrated: coverage=%.2f, n_cal=%d, half_width=%.6f",
            coverage, self._n_cal, self._half_width,
        )

    def predict_interval(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._calibrated:
            raise RuntimeError("SklearnConformalPredictor not calibrated.")
        pred = predictions.reshape(-1)
        hw = self._half_width
        return pred - hw, pred + hw

    @property
    def half_width(self) -> float | None:
        return self._half_width
