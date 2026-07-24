"""Test Metrics Severity Bins."""
import numpy as np
from eval.metrics import compute_all_metrics

def test_severity_bins():
    """Verify threshold binning (0.99 -> low, 1.00 -> high)."""
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([1.0, 1.0])
    ood = np.array([0.99, 1.01])
    
    res = compute_all_metrics(y_true, y_pred, ood)
    
    assert "low_ood" in res
    assert res["low_ood"]["n_samples"] == 1.0
    
    assert "high_ood" in res
    assert res["high_ood"]["n_samples"] == 1.0
    
    assert res["all"]["n_samples"] == 2.0
