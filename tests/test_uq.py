"""Test Uncertainty Quantification (Spec UQ).

Verifies:
1. Conformal predictor calibration achieves target coverage.
2. MC-Dropout computes variance across multiple stochastic passes.
"""
import pytest
import torch
import torch.nn as nn
from ragmat.uncertainty.conformal import ConformalPredictor
from ragmat.uncertainty.mc_dropout import MCDropoutUQ

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.drop = nn.Dropout(0.5)
    def forward(self, batch):
        return self.drop(self.fc(batch.x)), None

class DummyBatch:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def to(self, device):
        return self

def test_conformal_coverage():
    """Verify conformal predictor hits ~90% coverage on calibration data."""
    torch.manual_seed(42)
    model = DummyModel()
    x = torch.randn(1000, 10)
    with torch.no_grad():
        model.eval()
        clean_pred = model(DummyBatch(x, None))[0]
    y_true = clean_pred + torch.randn_like(clean_pred) * 0.5
    cp = ConformalPredictor()
    cp.calibrate(model, [DummyBatch(x, y_true)], coverage=0.90)
    lower, upper = cp.predict_interval(clean_pred)
    covered = (y_true >= lower) & (y_true <= upper)
    emp_cov = covered.float().mean().item()
    assert 0.85 <= emp_cov <= 0.95, f"Expected ~0.90, got {emp_cov}"

def test_mc_dropout_variance():
    """Verify MC-Dropout produces non-zero variance by keeping dropout active."""
    torch.manual_seed(42)
    model = DummyModel()
    model.eval()
    batch = DummyBatch(torch.randn(5, 10), None)
    mean_pred, var_pred = MCDropoutUQ.predict_with_uncertainty(
        model, lambda: model(batch)[0], n_passes=10
    )
    assert var_pred.mean().item() > 0.0
    assert not model.training
