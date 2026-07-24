import numpy as np

class SklearnConformalPredictor:
    def __init__(self, coverage_target=0.90):
        self.coverage_target = coverage_target
        self.q_hat = None
    def calibrate(self, y_val, val_preds):
        residuals = np.abs(y_val - val_preds)
        n = len(residuals)
        q_level = np.ceil((n+1) * self.coverage_target) / n
        self.q_hat = np.quantile(residuals, min(q_level, 1.0))
    def predict_intervals(self, test_preds):
        assert self.q_hat is not None, 'Call calibrate() first'
        return test_preds - self.q_hat, test_preds + self.q_hat
    def compute_coverage(self, y_test, lower, upper):
        return float(((y_test >= lower) & (y_test <= upper)).mean())
