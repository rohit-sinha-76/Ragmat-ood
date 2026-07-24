"""Gap 1 - Adaptive Gating Analysis for RAGMat-OOD.

Answers the single most important open question:
  Does the Mahalanobis OOD detector separate element-out from IID,
  and does adaptive gating (CGCNN -> RF fallback) recover RF performance?

ALL data loaded from cached checkpoints + graph files. ZERO retraining.
Estimated runtime: 20-40 min on CPU (embedding extraction is bottleneck).

Usage:
    wsl bash -c "source ~/miniforge3/etc/profile.d/conda.sh && \
      conda activate ragmat && python run_gating_analysis.py"

Outputs written to results/:
    gating_analysis_TIMESTAMP.json
    gating_final_report.md
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.metrics import roc_auc_score

# Project setup
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ragmat.encoders.cgcnn import CGCNNEncoder
from ragmat.ood.mahalanobis import MahalanobisDetector

# Paths
_CHECKPOINTS_DIR = _PROJECT_ROOT / "final_result" / "checkpoints"
_GRAPHS_DIR      = _PROJECT_ROOT / "data" / "graphs"
_SPLITS_DIR      = _PROJECT_ROOT / "data" / "splits"
_RESULTS_DIR     = _PROJECT_ROOT / "final_result"
_LOGS_DIR        = _PROJECT_ROOT / "final_result" / "logs"

# Hardware optimisation: T1000 / i7-14700 (20 usable threads)
torch.set_num_threads(20)
BATCH_SIZE  = 512   # inference only
NUM_WORKERS = 0     # main process avoids pickling overhead

# CGCNN spec (must match training)
CGCNN_SPEC = dict(node_dim=92, edge_dim=40, hidden_dim=64,
                  n_conv_layers=3, dropout_rate=0.1)
DIM = 64

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PROPS  = ["formation_energy", "band_gap"]
SPLITS = ["iid", "family_out", "element_out"]

# Known Tier-0 RF baseline MAEs (from phase3_final_report)
RF_TEST_MAE = {
    "formation_energy": {"iid": 0.1063, "family_out": 0.2366, "element_out": 0.1805},
    "band_gap":         {"iid": 0.2261, "family_out": 0.2529, "element_out": 0.3203},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOGS_DIR / "gating_analysis.log"),
    ],
)
logger = logging.getLogger("gating")


# I/O helpers

def load_cached_graphs(prop: str, split: str, partition: str) -> list:
    p = _GRAPHS_DIR / f"graphs_{prop}_{split}_{partition}.pt"
    logger.info("Loading %s (%.1f GB) ...", p.name, p.stat().st_size / 1e9)
    t0 = time.time()
    data = torch.load(str(p), weights_only=False)
    logger.info("  Loaded %d graphs in %.0fs", len(data), time.time() - t0)
    return data


def load_encoder(prop: str, split: str, device: torch.device):
    """Returns (frozen_encoder, y_scaler, train_ids)."""
    opt_name = f"tier1_{prop}_{split}_optimized_best.pt"
    path = _CHECKPOINTS_DIR / opt_name
    if not path.exists():
        path = _PROJECT_ROOT / "old_files" / opt_name
    
    if not path.exists():
        name = f"tier1_{prop}_{split}_base_best.pt"
        path = _CHECKPOINTS_DIR / name
        if not path.exists():
            path = _PROJECT_ROOT / "old_files" / name
    
    name = path.name
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    encoder = CGCNNEncoder(
        node_dim=CGCNN_SPEC["node_dim"],
        edge_dim=CGCNN_SPEC["edge_dim"],
        hidden_dim=DIM,
        n_conv_layers=CGCNN_SPEC["n_conv_layers"],
        dropout_rate=CGCNN_SPEC["dropout_rate"],
    ).to(device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    for p_ in encoder.parameters():
        p_.requires_grad = False

    y_scaler  = pickle.loads(ckpt["y_scaler"])
    train_ids = ckpt["train_ids"]
    logger.info("Loaded %s  val_mae=%.4f", name, ckpt.get("val_mae", float("nan")))
    return encoder, y_scaler, train_ids


def extract_train_embeddings_only(
    encoder, graph_list: list, device: torch.device, tag: str = ""
) -> np.ndarray:
    """
    Extract ONLY embeddings from a graph list (no predictions).
    Memory-efficient: processes in batches, never holds full list in GPU memory.
    Returns ndarray (N, DIM).
    """
    loader = PyGLoader(graph_list, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS)
    emb_chunks = []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            emb = encoder.get_embedding(batch.to(device))
            emb_chunks.append(emb.cpu().numpy())
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                done = (i + 1) * BATCH_SIZE
                logger.info("  %s  batch %d  ~%d samples  %.0fs", tag, i+1, done, elapsed)

    result = np.concatenate(emb_chunks, axis=0)
    logger.info("  %s  done: %d embeddings in %.0fs", tag, len(result), time.time() - t0)
    return result


def extract_test_predictions(
    encoder, y_scaler, graph_list: list,
    device: torch.device, id_to_label: dict, tag: str = ""
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run frozen encoder on test graphs.
    Returns: (material_ids, y_true, y_pred_orig_scale, embeddings)
    """
    loader = PyGLoader(graph_list, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS)
    preds_chunks, embs_chunks, ids_out = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            pred_scaled, emb = encoder(batch)
            pred_orig = y_scaler.inverse_transform(pred_scaled.cpu().numpy())
            preds_chunks.append(pred_orig)
            embs_chunks.append(emb.cpu().numpy())
            ids_out.extend(batch.material_id)
            if (i + 1) % 20 == 0:
                logger.info("  %s  batch %d  %.0fs", tag, i+1, time.time()-t0)

    preds_np = np.concatenate(preds_chunks).ravel()
    embs_np  = np.concatenate(embs_chunks, axis=0)
    y_true   = np.array([id_to_label[mid] for mid in ids_out])
    logger.info("  %s  done: %d samples in %.0fs", tag, len(ids_out), time.time()-t0)
    return ids_out, y_true, preds_np, embs_np


def load_rf_csv(prop: str, split: str) -> tuple[dict[str, float], dict[str, float]]:
    """Load Tier-0 RF predictions. Returns (dict of y_true, dict of y_pred) mapped by material_id."""
    path = _RESULTS_DIR / f"predictions_tier0_{prop}_{split}_none.csv"
    if not path.exists():
        logger.warning("RF CSV not found: %s", path.name)
        return {}, {}
    y_true_dict, y_pred_dict = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mid = row["material_id"]
            y_true_dict[mid] = float(row["y_true"])
            y_pred_dict[mid] = float(row["y_pred"])
    logger.info("RF CSV loaded: %s  n=%d", path.name, len(y_true_dict))
    return y_true_dict, y_pred_dict


def bootstrap_mae_ci(
    y_true: np.ndarray, y_pred: np.ndarray,
    n_boot: int = 5000, alpha: float = 0.05
) -> dict:
    """Paired bootstrap 95% CI on MAE (same index resampled for both arrays)."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    boot_maes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_maes.append(float(np.abs(y_true[idx] - y_pred[idx]).mean()))
    mae_obs = float(np.abs(y_true - y_pred).mean())
    return {
        "mae":   mae_obs,
        "ci_lo": float(np.percentile(boot_maes, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot_maes, 100 * (1 - alpha / 2))),
        "n":     n,
    }


# Core analysis

def analyse_property(prop: str, device: torch.device) -> dict:
    logger.info("=" * 70)
    logger.info("PROPERTY: %s", prop.upper())
    logger.info("=" * 70)
    result = {}

    # Load JARVIS labels once
    from ragmat.data.loader import JARVISLoader
    logger.info("Loading JARVIS labels ...")
    raw_data = JARVISLoader().load(prop)
    id_to_label = {d[2]: d[1] for d in raw_data}
    del raw_data
    logger.info("Labels loaded: %d", len(id_to_label))

    # Step 1: Fit Mahalanobis detector on IID TRAIN embeddings
    logger.info("[1/5] Fitting Mahalanobis on IID train embeddings ...")
    iid_enc, iid_scaler, _ = load_encoder(prop, "iid", device)
    iid_train_graphs = load_cached_graphs(prop, "iid", "train")
    train_embs = extract_train_embeddings_only(
        iid_enc, iid_train_graphs, device, tag=f"{prop}/iid/train")
    del iid_train_graphs  # free ~17 GB RAM

    detector = MahalanobisDetector(threshold_percentile=95.0)
    detector.fit(train_embs)
    del train_embs
    logger.info("Detector fitted. norm_threshold=%.4f", detector.normalized_threshold)

    # Step 2: Score each split's test set using its own encoder
    logger.info("[2/5] Scoring all test splits ...")
    split_ood_scores    = {}
    split_cgcnn_preds   = {}
    split_ytrue         = {}
    split_ids           = {}

    for split in SPLITS:
        logger.info("  Processing %s/%s ...", prop, split)
        enc, scaler, _ = load_encoder(prop, split, device)
        test_graphs = load_cached_graphs(prop, split, "test")

        ids_arr, y_true_arr, pred_arr, embs_arr = extract_test_predictions(
            enc, scaler, test_graphs, device, id_to_label,
            tag=f"{prop}/{split}/test"
        )
        del test_graphs, enc

        ood_scores = detector.score(embs_arr)
        del embs_arr

        split_ood_scores[split]  = ood_scores
        split_cgcnn_preds[split] = pred_arr
        split_ytrue[split]       = y_true_arr
        split_ids[split]         = ids_arr

        # Save element-out predictions to CSV for Gap 4
        if split == "element_out":
            csv_path = _RESULTS_DIR / f"predictions_tier1_{prop}_{split}_base.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["material_id", "y_true", "y_pred", "ood_score"])
                for i in range(len(ids_arr)):
                    writer.writerow([ids_arr[i], y_true_arr[i], pred_arr[i], ood_scores[i]])
            logger.info("Saved CGCNN predictions: %s", csv_path.name)

        result[split] = {
            "n_test": int(len(y_true_arr)),
            "ood_scores": {
                "mean": float(ood_scores.mean()),
                "std":  float(ood_scores.std()),
                "p25":  float(np.percentile(ood_scores, 25)),
                "p50":  float(np.percentile(ood_scores, 50)),
                "p75":  float(np.percentile(ood_scores, 75)),
                "p90":  float(np.percentile(ood_scores, 90)),
                "p95":  float(np.percentile(ood_scores, 95)),
                "p99":  float(np.percentile(ood_scores, 99)),
            },
            "cgcnn_mae_ci": bootstrap_mae_ci(y_true_arr, pred_arr),
        }
        logger.info(
            "  %s/%s  ood_mean=%.4f ood_std=%.4f p95=%.4f  cgcnn_mae=%.4f",
            prop, split,
            ood_scores.mean(), ood_scores.std(), np.percentile(ood_scores, 95),
            result[split]["cgcnn_mae_ci"]["mae"],
        )

    # Step 3: AUROC
    logger.info("[3/5] AUROC computation ...")
    s_iid = split_ood_scores["iid"]
    s_fo  = split_ood_scores["family_out"]
    s_eo  = split_ood_scores["element_out"]

    auroc_eo = float(roc_auc_score(
        np.concatenate([np.zeros(len(s_iid)), np.ones(len(s_eo))]),
        np.concatenate([s_iid, s_eo])
    ))
    auroc_fo = float(roc_auc_score(
        np.concatenate([np.zeros(len(s_iid)), np.ones(len(s_fo))]),
        np.concatenate([s_iid, s_fo])
    ))
    result["auroc_element_out_vs_iid"] = auroc_eo
    result["auroc_family_out_vs_iid"]  = auroc_fo
    logger.info("AUROC element-out vs IID : %.4f", auroc_eo)
    logger.info("AUROC family-out  vs IID : %.4f", auroc_fo)

    # Step 4: Gating sweep on element-out
    logger.info("[4/5] Gating sweep on element-out ...")
    ids_eo    = split_ids["element_out"]
    y_eo      = split_ytrue["element_out"]
    p_eo_cgcnn = split_cgcnn_preds["element_out"]
    ood_eo    = split_ood_scores["element_out"]

    y_rf_eo_dict, p_rf_eo_dict = load_rf_csv(prop, "element_out")
    gate_sweep   = []
    optimal_gate = None

    if p_rf_eo_dict:
        # Match by material_id
        matched_idx = []
        p_rf_aligned = []
        for i, mid in enumerate(ids_eo):
            if mid in p_rf_eo_dict:
                matched_idx.append(i)
                p_rf_aligned.append(p_rf_eo_dict[mid])

        logger.info("Matched %d/%d element-out samples between CGCNN and RF", len(matched_idx), len(ids_eo))

        if matched_idx:
            y_eo_matched = y_eo[matched_idx]
            p_eo_cgcnn_matched = p_eo_cgcnn[matched_idx]
            ood_eo_matched = ood_eo[matched_idx]
            p_rf_aligned = np.array(p_rf_aligned)

            rf_base   = RF_TEST_MAE[prop]["element_out"]
            cgcnn_mae = float(np.abs(y_eo_matched - p_eo_cgcnn_matched).mean())

            for thresh in THRESHOLDS:
                use_rf = ood_eo_matched >= thresh
                gated_pred = np.where(use_rf, p_rf_aligned, p_eo_cgcnn_matched)
                ci = bootstrap_mae_ci(y_eo_matched, gated_pred)

                entry = {
                    "threshold":            thresh,
                    "gated_mae":            ci["mae"],
                    "ci_lo":                ci["ci_lo"],
                    "ci_hi":                ci["ci_hi"],
                    "n_to_rf":              int(use_rf.sum()),
                    "pct_routed_to_rf":     round(100.0 * use_rf.sum() / len(ood_eo_matched), 1),
                    "delta_vs_rf_baseline": round(ci["mae"] - rf_base, 4),
                    "improvement_vs_cgcnn": round(cgcnn_mae - ci["mae"], 4),
                }
                gate_sweep.append(entry)
                logger.info(
                    "  thresh=%.1f  gated_mae=%.4f [%.4f, %.4f]  "
                    "rf_pct=%.0f%%  delta_rf=%+.4f  gain_vs_cgcnn=+%.4f",
                    thresh, ci["mae"], ci["ci_lo"], ci["ci_hi"],
                    entry["pct_routed_to_rf"],
                    entry["delta_vs_rf_baseline"],
                    entry["improvement_vs_cgcnn"],
                )

            optimal_gate = min(gate_sweep, key=lambda x: x["gated_mae"])
            logger.info(
                "Optimal: thresh=%.1f  gated_mae=%.4f  (RF=%.4f CGCNN=%.4f)",
                optimal_gate["threshold"], optimal_gate["gated_mae"], rf_base, cgcnn_mae
            )
    else:
        logger.warning("RF CSV not loaded, skipping sweep")

    result["element_out"]["gate_sweep"]   = gate_sweep
    result["element_out"]["optimal_gate"] = optimal_gate

    # Step 5: RF bootstrap CIs for all splits
    logger.info("[5/5] RF bootstrap CIs for all splits ...")
    for split in SPLITS:
        y_rf_dict, p_rf_dict = load_rf_csv(prop, split)
        if y_rf_dict and p_rf_dict:
            y_rf_arr = np.array(list(y_rf_dict.values()))
            p_rf_arr = np.array(list(p_rf_dict.values()))
            result[split]["rf_mae_ci"] = bootstrap_mae_ci(y_rf_arr, p_rf_arr)

    del iid_enc  # release encoder
    return result


# Report generation

def generate_report(all_results: dict) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    L = [
        "# Gating Analysis Report — RAGMat-OOD (Gap 1)",
        f"**Generated**: {ts}",
        "",
        "> This report closes Gap 1 of the paper: adaptive gating using Mahalanobis OOD detection.",
        "",
        "---",
        "## 1. OOD Score Distributions",
        "Detector fitted on IID train embeddings. Scores normalised to [0,1]; higher = more OOD.",
        "",
        "| Property | Split | N | Mean | Std | p50 | p90 | p95 | p99 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for prop in PROPS:
        d = all_results.get(prop, {})
        if "error" in d:
            continue
        for split in SPLITS:
            if split not in d:
                continue
            s = d[split]["ood_scores"]
            n = d[split]["n_test"]
            L.append(
                f"| {prop} | {split} | {n} | {s['mean']:.4f} | {s['std']:.4f} | "
                f"{s['p50']:.4f} | {s['p90']:.4f} | {s['p95']:.4f} | {s['p99']:.4f} |"
            )

    L += [
        "", "---",
        "## 2. OOD Detector AUROC",
        "AUROC > 0.80 = gate usable | 0.65-0.80 = marginal | < 0.65 = blind.",
        "",
        "| Property | AUROC element-out vs IID | AUROC family-out vs IID | Verdict |",
        "|---|---|---|---|",
    ]
    for prop in PROPS:
        d = all_results.get(prop, {})
        if "error" in d:
            continue
        ae = d.get("auroc_element_out_vs_iid", float("nan"))
        af = d.get("auroc_family_out_vs_iid", float("nan"))
        v  = "✅ Gate works" if ae > 0.80 else ("⚠️ Marginal" if ae > 0.65 else "❌ Gate blind")
        L.append(f"| {prop} | {ae:.4f} | {af:.4f} | {v} |")

    L += [
        "", "---",
        "## 3. Element-Out Gating Sweep",
        "Strategy: use CGCNN if ood_score < threshold, fall back to RF if ood_score >= threshold.",
        "",
        "| Property | Threshold | Gated MAE | 95% CI | % to RF | Δ vs RF | Gain vs CGCNN |",
        "|---|---|---|---|---|---|---|",
    ]
    for prop in PROPS:
        d = all_results.get(prop, {})
        if "error" in d:
            continue
        sweep = d.get("element_out", {}).get("gate_sweep", [])
        opt   = d.get("element_out", {}).get("optimal_gate")
        for row in sweep:
            star = " ⭐" if opt and row["threshold"] == opt["threshold"] else ""
            sgn  = "+" if row["delta_vs_rf_baseline"] >= 0 else ""
            L.append(
                f"| {prop} | {row['threshold']:.1f}{star} | {row['gated_mae']:.4f} | "
                f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] | "
                f"{row['pct_routed_to_rf']:.0f}% | "
                f"{sgn}{row['delta_vs_rf_baseline']:.4f} | "
                f"+{row['improvement_vs_cgcnn']:.4f} |"
            )
        if not sweep:
            L.append(f"| {prop} | — | — | — | — | — | — |")

    L += [
        "", "---",
        "## 4. Full MAE Comparison — Bootstrap 95% CIs",
        "",
        "| Property | Split | Model | MAE | 95% CI | N |",
        "|---|---|---|---|---|---|",
    ]
    for prop in PROPS:
        d = all_results.get(prop, {})
        if "error" in d:
            continue
        for split in SPLITS:
            if split not in d:
                continue
            for key, label in [("cgcnn_mae_ci", "CGCNN Tier-1"), ("rf_mae_ci", "RF Tier-0")]:
                ci = d[split].get(key)
                if ci:
                    L.append(
                        f"| {prop} | {split} | {label} | {ci['mae']:.4f} | "
                        f"[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] | {ci['n']} |"
                    )

    L += ["", "---", "## 5. H5 Hypothesis Verdict", ""]
    for prop in PROPS:
        d = all_results.get(prop, {})
        if "error" in d:
            L += [f"### {prop}", f"ERROR: {d['error']}", ""]
            continue
        ae  = d.get("auroc_element_out_vs_iid", 0.0)
        opt = d.get("element_out", {}).get("optimal_gate")
        cgcnn_eo = d["element_out"]["cgcnn_mae_ci"]["mae"]
        rf_eo    = RF_TEST_MAE[prop]["element_out"]

        L.append(f"### {prop.replace('_',' ').title()}")
        if ae > 0.80:
            L.append(f"- ✅ **OOD Detector**: AUROC={ae:.3f} — element-out IS separable from IID")
        elif ae > 0.65:
            L.append(f"- ⚠️ **OOD Detector**: AUROC={ae:.3f} — marginal separability")
        else:
            L.append(f"- ❌ **OOD Detector**: AUROC={ae:.3f} — gate cannot distinguish splits")

        if opt:
            gap_total  = cgcnn_eo - rf_eo
            gap_closed = cgcnn_eo - opt["gated_mae"]
            rec_pct    = 100.0 * gap_closed / gap_total if gap_total > 0 else 0.0
            L.append(
                f"- Best gated MAE = **{opt['gated_mae']:.4f}** "
                f"@ threshold {opt['threshold']} "
                f"({opt['pct_routed_to_rf']:.0f}% routed to RF)"
            )
            L.append(f"  - CGCNN base MAE = {cgcnn_eo:.4f}")
            L.append(f"  - RF baseline    = {rf_eo:.4f}")
            L.append(f"  - Recovery       = **{rec_pct:.1f}%** of CGCNN→RF gap closed")
            if rec_pct >= 50:
                L.append("- ✅ **H5 PASS**: Gating recovers ≥50% of the element-out performance gap")
            else:
                L.append(f"- ❌ **H5 FAIL**: Only {rec_pct:.1f}% recovery (threshold: 50%)")
        else:
            L.append("- ⚠️ Gating sweep not available")
        L.append("")

    return "\n".join(L)


def main():
    t0     = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s  |  Threads: %d", device, torch.get_num_threads())

    all_results = {}
    for prop in PROPS:
        try:
            all_results[prop] = analyse_property(prop, device)
        except Exception as exc:
            logger.error("FAILED %s: %s", prop, exc, exc_info=True)
            all_results[prop] = {"error": str(exc)}

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = _RESULTS_DIR / f"gating_analysis_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("JSON: %s", json_path)

    report = generate_report(all_results)
    md_path = _RESULTS_DIR / "gating_final_report.md"
    with open(md_path, "w") as f:
        f.write(report)
    logger.info("Report: %s", md_path)

    # Stdout summary
    runtime = (time.time() - t0) / 60
    print("\n" + "=" * 72)
    print("GATING ANALYSIS COMPLETE")
    print("=" * 72)
    for prop in PROPS:
        d = all_results.get(prop, {})
        if "error" in d:
            print(f"  {prop}: ERROR — {d['error']}")
            continue
        ae  = d.get("auroc_element_out_vs_iid", float("nan"))
        opt = d.get("element_out", {}).get("optimal_gate")
        cm  = d.get("element_out", {}).get("cgcnn_mae_ci", {}).get("mae", float("nan"))
        rf  = RF_TEST_MAE[prop]["element_out"]
        print(f"\n  {prop.upper()}")
        print(f"    AUROC element-out vs IID : {ae:.4f}")
        print(f"    CGCNN element-out MAE    : {cm:.4f}")
        print(f"    RF baseline              : {rf:.4f}")
        if opt:
            gap_total  = cm - rf
            gap_closed = cm - opt["gated_mae"]
            rec_pct    = 100.0 * gap_closed / gap_total if gap_total > 0 else 0.0
            print(f"    Best gated MAE           : {opt['gated_mae']:.4f}  "
                  f"[thresh={opt['threshold']}, {opt['pct_routed_to_rf']:.0f}% to RF]")
            print(f"    Gap recovery             : {rec_pct:.1f}%")
    print(f"\n  Runtime: {runtime:.1f} min")
    print(f"  Report: {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
