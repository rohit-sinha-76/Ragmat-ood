#!/usr/bin/env python
"""Pre-compute and serialize the Mahalanobis OOD detectors.

Fits the Mahalanobis detectors on the IID train embeddings for both properties,
saving them to checkpoints/ so that they can be loaded instantly during inference.
"""
import sys
import pickle
import logging
from pathlib import Path
import numpy as np
import torch

# Add project root and scripts directory to python path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from ragmat.encoders.cgcnn import CGCNNEncoder
from ragmat.ood.mahalanobis import MahalanobisDetector
from run_gating_analysis import load_cached_graphs, extract_train_embeddings_only, load_encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("serialize_detectors")

PROPERTIES = ["formation_energy", "band_gap"]
_CHECKPOINTS_DIR = _PROJECT_ROOT / "checkpoints"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device for embedding extraction: %s", device)
    
    _CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for prop in PROPERTIES:
        logger.info("=" * 60)
        logger.info("PROCESSING PROPERTY: %s", prop.upper())
        logger.info("=" * 60)
        
        # 1. Load IID encoder
        logger.info("Loading IID GNN encoder...")
        encoder, _, _ = load_encoder(prop, "iid", device)
        
        # 2. Load IID Train graphs
        logger.info("Loading IID train graphs...")
        train_graphs = load_cached_graphs(prop, "iid", "train")
        
        # 3. Extract train embeddings
        logger.info("Extracting GNN embeddings for IID train split...")
        train_embs = extract_train_embeddings_only(
            encoder, train_graphs, device, tag=f"{prop}/iid/train"
        )
        del train_graphs # Free memory
        
        # 4. Fit Mahalanobis detector
        logger.info("Fitting Mahalanobis detector on IID train embeddings...")
        detector = MahalanobisDetector(threshold_percentile=95.0)
        detector.fit(train_embs)
        del train_embs # Free memory
        
        # 5. Save detector pickle
        save_path = _CHECKPOINTS_DIR / f"mahalanobis_detector_{prop}.pkl"
        logger.info("Saving fitted detector to %s ...", save_path)
        with open(save_path, "wb") as f:
            pickle.dump(detector, f)
            
        logger.info("Successfully serialized detector for %s. Normalized Threshold: %.4f", prop, detector.normalized_threshold)

if __name__ == "__main__":
    main()
