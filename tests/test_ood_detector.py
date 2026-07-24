"""Test MahalanobisDetector (spec OD1-OD3 + 95th-pct threshold requirement).

Checks:
- Scores are non-zero with real embeddings (not stubbed)
- 95th-percentile default threshold
- In-distribution scores cluster below 1.0; OOD scores cluster above 1.0
- Fit-then-score pipeline does not leak test data into fit
"""
import numpy as np
import pytest
from ragmat.ood.mahalanobis import MahalanobisDetector


def test_default_threshold_is_95th_pct():
    """Spec mandates 95th percentile threshold, not 50th."""
    det = MahalanobisDetector()
    assert det._threshold_percentile == 95.0, (
        f"Default threshold_percentile must be 95.0, got {det._threshold_percentile}"
    )


def test_scores_nonzero_after_fit():
    """OD3: scores must not be all-zeros after fitting."""
    np.random.seed(42)
    train = np.random.randn(200, 30).astype(np.float32)
    test  = np.random.randn(50, 30).astype(np.float32)
    det = MahalanobisDetector()
    det.fit(train)
    scores = det.score(test)
    assert scores.std() > 0.0, "OOD scores are all identical -- detector appears stubbed."


def test_score_before_fit_raises():
    """Must raise RuntimeError if score() is called before fit()."""
    det = MahalanobisDetector()
    with pytest.raises(RuntimeError, match="not fitted"):
        det.score(np.random.randn(5, 10).astype(np.float32))


def test_indistrib_scores_below_threshold():
    """In-distribution test samples (same dist as train) should mostly score < 1.0."""
    np.random.seed(0)
    train = np.random.randn(500, 20).astype(np.float32)
    # Test from same distribution
    test_id = np.random.randn(200, 20).astype(np.float32)
    det = MahalanobisDetector()
    det.fit(train)
    scores = det.score(test_id)
    frac_below = (scores < det.normalized_threshold).mean()
    # At 95th-pct threshold: ~95% of TRAINING is below. Test (same dist) should be similar.
    assert frac_below > 0.80, (
        f"Only {frac_below:.1%} of in-dist test samples score < threshold. Threshold may be miscalibrated."
    )


def test_ood_scores_above_threshold():
    """Samples drawn from a very different distribution should mostly score > threshold."""
    np.random.seed(1)
    # Train on tight cluster around one pole
    train = np.random.randn(500, 20).astype(np.float32)
    train[:, 0] += 50.0
    # OOD samples: far from train pole
    ood = np.random.randn(100, 20).astype(np.float32)
    ood[:, 1] += 50.0
    det = MahalanobisDetector()
    det.fit(train)
    scores = det.score(ood)
    frac_above = (scores > det.normalized_threshold).mean()
    assert frac_above > 0.90, (
        f"Only {frac_above:.1%} of clearly OOD samples score > threshold. OOD detection is failing."
    )


def test_fit_uses_only_train_data():
    """Ensure test embeddings are never seen during fit (no data leakage into detector)."""
    np.random.seed(7)
    train = np.random.randn(100, 15).astype(np.float32)
    det = MahalanobisDetector()
    det.fit(train)
    # Mean must equal training mean, not include test data
    # Mean must equal L2-normalized training mean
    norms = np.linalg.norm(train, axis=1, keepdims=True)
    train_norm = train / np.clip(norms, 1e-12, None)
    expected_mean = train_norm.mean(axis=0)
    np.testing.assert_allclose(det._mean, expected_mean, rtol=1e-5,
        err_msg="Detector mean differs from training mean -- possible test data leakage into fit.")
