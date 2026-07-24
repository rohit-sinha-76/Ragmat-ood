#!/usr/bin/env python
"""Unified Inference Pipeline for RAGMat-OOD.

Allows researchers to predict Formation Energy and Band Gap of arbitrary crystal structures (e.g. CIF files)
using the trained GNN model, classical Random Forest model, Mahalanobis OOD gating, and ZSNI.
"""
import os
import sys
import gc
import argparse
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGLoader

# Add project root and scripts directory to python path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from pymatgen.core import Structure, Element
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from ragmat.encoders.graph_builder import CrystalGraphBuilder, _ELEMENTS, _ELEMENT_INDEX
from ragmat.encoders.cgcnn import CGCNNEncoder
from ragmat.features.matminer_descriptors import MatminerFeaturizer
from ragmat.ood.mahalanobis import MahalanobisDetector
from scripts.run_gating_analysis import load_encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inference")

# Excluded elements in the Element-Out split
ELEMENT_OUT_MISSING = {
    'Y', 'Hf', 'Sc', 'At', 'In', 'Ga', 'Se', 'Fr', 'Ta', 'Ra', 'Nb', 'Rn', 'Po', 'Te', 'Bi'
}

def get_element_features():
    """Extract and standardize Row/Group features for all 92 elements."""
    features = []
    for el_str in _ELEMENTS:
        try:
            el = Element(el_str)
            row = el.row
            group = el.group
            features.append([row, group])
        except Exception:
            features.append([0.0, 0.0])
    feats = np.array(features, dtype=np.float32)
    means = feats.mean(axis=0)
    stds = feats.std(axis=0)
    stds[stds == 0] = 1.0
    return (feats - means) / stds

def apply_zsni_patch(encoder):
    """Zero-Shot Node Imputation: Patches uninitialized embedding rows with periodic table neighbors."""
    logger.info("Applying Zero-Shot Node Imputation (ZSNI) weight patching...")
    element_features = get_element_features()
    missing_indices = [_ELEMENT_INDEX[el] for el in ELEMENT_OUT_MISSING if el in _ELEMENT_INDEX]
    seen_indices = set(range(92)) - set(missing_indices)
    
    imputation_map = {}
    k = 2
    for m in missing_indices:
        m_feat = element_features[m]
        dists = []
        for s in seen_indices:
            s_feat = element_features[s]
            dist = np.linalg.norm(m_feat - s_feat)
            dists.append((dist, s))
        dists.sort()
        nearest = [s for d, s in dists[:k]]
        imputation_map[m] = nearest
        
    with torch.no_grad():
        weight = encoder.embedding.weight.data # (64, 92)
        for m, nearest in imputation_map.items():
            imputed = torch.stack([weight[:, s] for s in nearest]).mean(dim=0)
            weight[:, m] = imputed
            
    logger.info("Successfully patched embedding weight rows for %d unseen elements.", len(missing_indices))

def load_rf_model(prop, split_type):
    """Load pre-trained Tier 0 Random Forest baseline and scaler with clean memory management."""
    rf_ckpt_path = _PROJECT_ROOT / "checkpoints" / f"tier0_{prop}_{split_type}_none_model.pkl"
    if not rf_ckpt_path.exists():
        rf_ckpt_path = _PROJECT_ROOT / "old_files" / f"tier0_{prop}_{split_type}_none_model.pkl"
    if not rf_ckpt_path.exists():
        rf_ckpt_path = _PROJECT_ROOT / "checkpoints" / f"tier0_{prop}_{split_type}_model.pkl"
        if not rf_ckpt_path.exists():
            rf_ckpt_path = _PROJECT_ROOT / "old_files" / f"tier0_{prop}_{split_type}_model.pkl"
    if not rf_ckpt_path.exists():
        logger.warning("RF checkpoint tier0_%s_%s_model.pkl not found. Initializing fallback RF model.", prop, split_type)
        rf = RandomForestRegressor(n_estimators=1, random_state=42)
        scaler = StandardScaler()
        dummy_x = np.zeros((2, 145))
        dummy_y = np.zeros(2)
        scaler.fit(dummy_x)
        rf.fit(scaler.transform(dummy_x), dummy_y)
        return rf, scaler
        
    gc.collect()
    with open(rf_ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    gc.collect()
    return ckpt["model"], ckpt["scaler"]

def load_mahalanobis_detector(prop):
    """Load serialized Mahalanobis detector for gating."""
    detector_path = _PROJECT_ROOT / "checkpoints" / f"mahalanobis_detector_{prop}.pkl"
    if not detector_path.exists():
        detector_path = _PROJECT_ROOT / "old_files" / f"mahalanobis_detector_{prop}.pkl"
    if not detector_path.exists():
        logger.warning("Mahalanobis detector %s not found. Initializing fallback detector.", detector_path.name)
        detector = MahalanobisDetector()
        dummy_emb = np.random.randn(20, 64)
        detector.fit(dummy_emb)
        return detector
    gc.collect()
    with open(detector_path, "rb") as f:
        detector = pickle.load(f)
    gc.collect()
    return detector

def run_predictions(structures, ids, prop, split_type, enable_zsni, threshold, device):
    """Execute prediction pipeline on structures for a single target property."""
    logger.info("Initializing models for property: %s (split: %s)", prop, split_type)
    
    # 1. Load GNN encoder
    encoder, y_scaler, _ = load_encoder(prop, split_type, device)
    
    # 2. Apply ZSNI patch if requested
    if enable_zsni and split_type == "element_out":
        apply_zsni_patch(encoder)
        
    # 3. Load RF model and scaler
    rf_model, rf_scaler = load_rf_model(prop, split_type)
    
    # 4. Load Mahalanobis OOD detector
    detector = load_mahalanobis_detector(prop)
    
    # 5. Build PyG crystal graphs
    builder = CrystalGraphBuilder(cutoff_radius=8.0, n_gaussian_basis=40)
    graphs = [builder.structure_to_graph(s, y=0.0, material_id=jid) for s, jid in zip(structures, ids)]
    
    # 6. Extract GNN embeddings and raw predictions
    logger.info("Running GNN inference...")
    gnn_preds = []
    embeddings = []
    
    loader = PyGLoader(graphs, batch_size=16, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_scaled, emb = encoder(batch)
            pred_orig = y_scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
            gnn_preds.extend(pred_orig.tolist())
            embeddings.append(emb.cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    
    # 7. Score OOD distance
    logger.info("Evaluating OOD status with Mahalanobis detector...")
    ood_scores = detector.score(embeddings)
    
    # 8. Extract Matminer features & predict using RF
    logger.info("Extracting Matminer features for RF fallback...")
    featurizer = MatminerFeaturizer(n_jobs=1)
    X_raw, valid_ids = featurizer.featurize_dataset(structures, ids)
    X_scaled = featurizer.transform(X_raw, rf_scaler)
    rf_preds_raw = rf_model.predict(X_scaled).flatten()
    
    # Map valid RF predictions to original ID order (fill NaN for failed featurizations)
    rf_preds_map = {mid: val for mid, val in zip(valid_ids, rf_preds_raw)}
    rf_preds = [rf_preds_map.get(jid, float("nan")) for jid in ids]
    
    # 9. Gate predictions
    final_preds = []
    decisions = []
    
    for i, jid in enumerate(ids):
        score = ood_scores[i]
        is_ood = score >= threshold
        
        # Decide fallback
        if is_ood:
            decisions.append(f"RF Fallback (OOD score {score:.3f} >= {threshold})")
            final_preds.append(rf_preds[i])
        else:
            decisions.append(f"GNN Prediction (OOD score {score:.3f} < {threshold})")
            final_preds.append(gnn_preds[i])
            
    return {
        "gnn_pred": gnn_preds,
        "rf_pred": rf_preds,
        "final_pred": final_preds,
        "ood_score": ood_scores.tolist(),
        "decision": decisions
    }

def main():
    parser = argparse.ArgumentParser(description="RAGMat-OOD Unified Inference Pipeline CLI")
    parser.add_argument(
        "--cif", 
        required=True, 
        help="Path to a single CIF file, a directory of CIF files, or space-separated list of CIFs"
    )
    parser.add_argument(
        "--property", 
        default="both", 
        choices=["formation_energy", "band_gap", "both"],
        help="Target property to predict (default: both)"
    )
    parser.add_argument(
        "--split-type",
        default="element_out",
        choices=["iid", "family_out", "element_out"],
        help="Context split under which the GNN was trained (default: element_out)"
    )
    parser.add_argument(
        "--enable-zsni",
        action="store_true",
        help="Enable Zero-Shot Node Imputation (ZSNI) for unseen elements in GNN (only relevant under element_out)"
    )
    parser.add_argument(
        "--fe-threshold",
        type=float,
        default=0.3,
        help="OOD gating threshold for formation energy (default: 0.3)"
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=0.9,
        help="OOD gating threshold for band gap (default: 0.9)"
    )
    parser.add_argument(
        "--output",
        help="Path to save prediction results as a CSV file"
    )
    args = parser.parse_args()

    # Locate CIF files
    cif_paths = []
    cif_input = args.cif
    
    # Handle folder or pattern or space separated list
    if os.path.isdir(cif_input):
        cif_paths = list(Path(cif_input).glob("*.cif"))
    elif "*" in cif_input or "?" in cif_input:
        import glob
        cif_paths = [Path(p) for p in glob.glob(cif_input)]
    else:
        cif_paths = [Path(p.strip()) for p in cif_input.split(",") if p.strip()]
        if len(cif_paths) == 1 and not cif_paths[0].exists():
            # Try space-separated fallback
            cif_paths = [Path(p.strip()) for p in cif_input.split() if p.strip()]

    # Validate exists
    valid_paths = [p for p in cif_paths if p.exists()]
    if not valid_paths:
        logger.error("No valid CIF files found matching input: %s", args.cif)
        sys.exit(1)
        
    logger.info("Found %d CIF files for inference.", len(valid_paths))
    
    # Parse structures
    structures = []
    ids = []
    for p in valid_paths:
        try:
            struct = Structure.from_file(str(p))
            structures.append(struct)
            ids.append(p.stem)
        except Exception as exc:
            logger.error("Failed to parse CIF structure from %s: %s", p.name, exc)
            
    if not structures:
        logger.error("No structures successfully parsed. Exiting.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    props_to_run = ["formation_energy", "band_gap"] if args.property == "both" else [args.property]
    results = {}
    
    for prop in props_to_run:
        threshold = args.fe_threshold if prop == "formation_energy" else args.bg_threshold
        pred_dict = run_predictions(
            structures=structures,
            ids=ids,
            prop=prop,
            split_type=args.split_type,
            enable_zsni=args.enable_zsni,
            threshold=threshold,
            device=device
        )
        results[prop] = pred_dict

    # Print pretty report
    print("\n" + "="*90)
    print(" RAGMat-OOD Prediction Report")
    print("="*90)
    
    for i, jid in enumerate(ids):
        print(f"Material: {jid}")
        for prop in props_to_run:
            data = results[prop]
            print(f"  [{prop.replace('_', ' ').title()}]")
            print(f"    GNN Pred: {data['gnn_pred'][i]:.4f}")
            print(f"    RF Pred : {data['rf_pred'][i]:.4f}")
            print(f"    OOD Dist: {data['ood_score'][i]:.4f}")
            print(f"    Decision: {data['decision'][i]}")
            print(f"    ★ Final Pred: {data['final_pred'][i]:.4f}")
        print("-"*90)
        
    # Save CSV if requested
    if args.output:
        out_rows = []
        for i, jid in enumerate(ids):
            row = {"material_id": jid}
            for prop in props_to_run:
                row[f"{prop}_gnn_pred"] = results[prop]["gnn_pred"][i]
                row[f"{prop}_rf_pred"] = results[prop]["rf_pred"][i]
                row[f"{prop}_ood_score"] = results[prop]["ood_score"][i]
                row[f"{prop}_decision"] = results[prop]["decision"][i]
                row[f"{prop}_final_pred"] = results[prop]["final_pred"][i]
            out_rows.append(row)
            
        df = pd.DataFrame(out_rows)
        df.to_csv(args.output, index=False)
        logger.info("Saved prediction report to %s", args.output)

if __name__ == "__main__":
    main()
