"""Evaluation metrics for RAGMat-OOD.

All metric functions operate on numpy arrays. Per-severity-bin versions
are computed by slicing on Mahalanobis OOD score quantiles.

Metrics:
- Regression: MAE, RMSE, R²
- Calibration: ECE, NLL
- OOD detection: AUROC, FPR95
- Retrieval: Recall@K (K=1,5,10), MRR
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

# OOD severity bins defined by Mahalanobis score quantile ranges
# OOD scores are normalised by the training max.
# threshold splits the distribution into low_ood and high_ood.



# ── Regression metrics ───────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-10))


# ── Calibration metrics ──────────────────────────────────────────────────────

def ece(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for regression prediction intervals.

    Args:
        y_true: Ground truth values ``(N,)``.
        lower: Lower bound of prediction intervals ``(N,)``.
        upper: Upper bound of prediction intervals ``(N,)``.
        n_bins: Number of confidence bins.

    Returns:
        ECE scalar in [0, 1].
    """
    # We measure the actual coverage at each nominal confidence level
    # by varying the interval width proportionally.
    midpoint = (lower + upper) / 2.0
    half_widths = (upper - lower) / 2.0
    residuals = np.abs(y_true - midpoint)

    alphas = np.linspace(0.0, 1.0, n_bins + 1)[1:]
    ece_val = 0.0
    for alpha in alphas:
        covered = (residuals <= alpha * half_widths).mean()
        ece_val += abs(covered - alpha) / n_bins
    return float(ece_val)


def nll_gaussian(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variance: np.ndarray,
) -> float:
    """Negative log-likelihood under a Gaussian predictive distribution.

    Args:
        y_true: Ground truth ``(N,)``.
        y_pred: Predicted mean ``(N,)``.
        variance: Predicted variance ``(N,)`` — from MC-Dropout.

    Returns:
        Mean NLL scalar.
    """
    var = np.maximum(variance, 1e-6)
    nll = 0.5 * (np.log(2 * np.pi * var) + (y_true - y_pred) ** 2 / var)
    return float(nll.mean())


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 10,
) -> float:
    if np.allclose(y_true, y_pred):
        return 0.0
    from scipy.stats import norm
    alphas = np.linspace(0.0, 1.0, n_bins + 1)[1:]
    ece_val = 0.0
    for alpha in alphas:
        z = norm.ppf((1 + alpha) / 2)
        lower = y_pred - z * sigma
        upper = y_pred + z * sigma
        covered = ((y_true >= lower) & (y_true <= upper)).mean()
        ece_val += abs(covered - alpha) / n_bins
    return float(ece_val)

def compute_auprc(y_true_binary: np.ndarray, y_scores: np.ndarray) -> float:
    return float(average_precision_score(y_true_binary, y_scores))

# ── OOD detection metrics ─────────────────────────────────────────────────────

def auroc_ood(
    ood_scores: np.ndarray,
    is_ood_labels: np.ndarray,
) -> float:
    """AUROC for OOD detection.

    Args:
        ood_scores: Predicted OOD scores ``(N,)``.
        is_ood_labels: Binary labels ``(N,)`` — 1 = OOD, 0 = in-distribution.

    Returns:
        AUROC scalar.
    """
    if len(np.unique(is_ood_labels)) < 2:
        logger.warning("AUROC undefined (only one class present), returning 0.5")
        return 0.5
    return float(roc_auc_score(is_ood_labels, ood_scores))


def fpr_at_tpr95(
    ood_scores: np.ndarray,
    is_ood_labels: np.ndarray,
) -> float:
    """False Positive Rate at 95% True Positive Rate (FPR95).

    Standard OOD detection metric: at the threshold where 95% of OOD
    samples are correctly detected, what fraction of in-distribution
    samples are falsely flagged?

    Args:
        ood_scores: Predicted OOD scores ``(N,)``.
        is_ood_labels: Binary labels ``(N,)`` — 1 = OOD, 0 = in-distribution.

    Returns:
        FPR95 scalar in [0, 1].
    """
    ood_scores_pos = ood_scores[is_ood_labels == 1]
    ood_scores_neg = ood_scores[is_ood_labels == 0]

    if len(ood_scores_pos) == 0 or len(ood_scores_neg) == 0:
        return float("nan")

    threshold = np.percentile(ood_scores_pos, 5)  # 95% TPR threshold
    fpr = float((ood_scores_neg >= threshold).mean())
    return fpr


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def recall_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[set[str]],
    k: int,
) -> float:
    """Recall@K: fraction of queries for which ≥1 relevant item appears in top-K.

    Args:
        retrieved_ids: Nested list ``(N, k_max)`` of retrieved material IDs.
        relevant_ids: List of ``N`` sets of truly relevant material IDs.
        k: Cutoff.

    Returns:
        Recall@K scalar in [0, 1].
    """
    hits = 0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        top_k = set(retrieved[:k])
        if top_k & relevant:
            hits += 1
    return float(hits / max(len(retrieved_ids), 1))


def mrr(
    retrieved_ids: list[list[str]],
    relevant_ids: list[set[str]],
) -> float:
    """Mean Reciprocal Rank.

    Args:
        retrieved_ids: Nested list ``(N, k_max)`` of retrieved material IDs.
        relevant_ids: List of ``N`` sets of truly relevant material IDs.

    Returns:
        MRR scalar in [0, 1].
    """
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        for rank, rid in enumerate(retrieved, start=1):
            if rid in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


# ── Per-severity-bin evaluation ───────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ood_scores: np.ndarray,
    variance: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    is_ood_labels: np.ndarray | None = None,
    retrieved_ids: list[list[str]] | None = None,
    relevant_ids: list[set[str]] | None = None,
    physical_relevance: np.ndarray | None = None,
    ood_threshold: float = 1.0,
    gate_mask: np.ndarray | None = None,
    base_preds: np.ndarray | None = None,
    pure_retrieval_preds: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Compute all evaluation metrics across all severity bins."""
    results: dict[str, dict[str, float]] = {}

    bins = {
        "low_ood": (0.0, ood_threshold),
        "high_ood": (ood_threshold, float("inf")),
        "all": (0.0, float("inf")),
    }

    for bin_name, (lo, hi) in bins.items():
        mask = (ood_scores >= lo) & (ood_scores < hi) if hi != float("inf") else (ood_scores >= lo)
        if mask.sum() == 0:
            continue

        bin_metrics: dict[str, float] = {}
        yt = y_true[mask]
        yp = y_pred[mask]
        oo = ood_scores[mask]

        bin_metrics["mae"] = mae(yt, yp)
        bin_metrics["rmse"] = rmse(yt, yp)
        bin_metrics["r2"] = r2(yt, yp)
        bin_metrics["n_samples"] = float(mask.sum())

        if variance is not None:
            bin_metrics["nll"] = nll_gaussian(yt, yp, variance[mask])

        if lower is not None and upper is not None:
            bin_metrics["ece"] = ece(yt, lower[mask], upper[mask])
            bin_metrics["conformal_coverage"] = float(np.mean((yt >= lower[mask]) & (yt <= upper[mask])))

        if is_ood_labels is not None:
            lbl = is_ood_labels[mask]
            if len(np.unique(lbl)) == 2:
                bin_metrics["auroc"] = auroc_ood(oo, lbl)
                bin_metrics["auprc"] = float(average_precision_score(lbl, oo))
                bin_metrics["fpr95"] = fpr_at_tpr95(oo, lbl)
            elif len(np.unique(is_ood_labels)) == 2:
                # If only 1 class in the bin but 2 overall, FPR/AUROC are technically undefined
                # in this bin alone.
                pass

        if retrieved_ids is not None and relevant_ids is not None:
            # We must slice lists using the boolean mask
            if gate_mask is not None:
                combined_mask = mask & gate_mask
                bin_retrieved = [ret for m, ret in zip(combined_mask, retrieved_ids) if m]
                bin_relevant = [rel for m, rel in zip(combined_mask, relevant_ids) if m]
            else:
                bin_retrieved = [ret for m, ret in zip(mask, retrieved_ids) if m]
                bin_relevant = [rel for m, rel in zip(mask, relevant_ids) if m]
            
            if len(bin_retrieved) > 0:
                bin_metrics["recall_at_1"] = recall_at_k(bin_retrieved, bin_relevant, 1)
                bin_metrics["recall_at_5"] = recall_at_k(bin_retrieved, bin_relevant, 5)
                bin_metrics["recall_at_10"] = recall_at_k(bin_retrieved, bin_relevant, 10)
                bin_metrics["mrr"] = mrr(bin_retrieved, bin_relevant)
            else:
                bin_metrics["recall_at_1"] = -1.0
                bin_metrics["recall_at_5"] = -1.0
                bin_metrics["recall_at_10"] = -1.0
                bin_metrics["mrr"] = -1.0
                import logging
                logging.getLogger(__name__).info("RECALL_NOT_COMPUTED_GATING_PATH: %s has 0 retrieved samples", bin_name)
        if physical_relevance is not None:
            bin_metrics["physical_relevance"] = float(np.mean(physical_relevance[mask]))

        # Gating stats
        if gate_mask is not None and base_preds is not None and pure_retrieval_preds is not None:
            bin_gate = gate_mask[mask]
            bin_y_true = yt
            
            n_tot = len(bin_gate)
            n_retrieved = int(bin_gate.sum())
            n_fallback = int(n_tot - n_retrieved)
            
            bin_metrics["n_retrieved"] = float(n_retrieved)
            bin_metrics["n_fallback"] = float(n_fallback)
            bin_metrics["pct_retrieved"] = float(n_retrieved / max(n_tot, 1) * 100)
            bin_metrics["pct_fallback"] = float(n_fallback / max(n_tot, 1) * 100)
            
            if n_retrieved > 0:
                bin_metrics["mae_retrieved_samples"] = mae(bin_y_true[bin_gate], pure_retrieval_preds[mask][bin_gate])
            else:
                bin_metrics["mae_retrieved_samples"] = float('nan')
                
            if n_fallback > 0:
                bin_metrics["mae_fallback_samples"] = mae(bin_y_true[~bin_gate], base_preds[mask][~bin_gate])
            else:
                bin_metrics["mae_fallback_samples"] = float('nan')

        results[bin_name] = bin_metrics

    return results
