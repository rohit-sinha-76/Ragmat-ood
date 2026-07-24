"""Evaluation runner for RAGMat-OOD.

Loads a trained model checkpoint and config, evaluates it on the test partition,
computes all metrics per OOD severity bin, and saves results to JSON and CSV.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import logging
import os
import pickle
import sys
from pathlib import Path

# Ensure project root is in sys.path when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as PyGLoader

from eval.metrics import compute_all_metrics
from ragmat.config import ExperimentConfig
from ragmat.data.loader import JARVISLoader
from ragmat.data.splitter import DataSplitter
from ragmat.encoders.cgcnn import CGCNNEncoder
from ragmat.encoders.graph_builder import CrystalGraphBuilder
from ragmat.features.matminer_descriptors import MatminerFeaturizer
from ragmat.gating import AdaptiveGate
from ragmat.ood.mahalanobis import MahalanobisDetector
from ragmat.retrieval.faiss_index import FAISSIndex
from ragmat.retrieval.concat_features import (
    concat_retrieval_features,
    concat_random_retrieval_features,
)
from ragmat.uncertainty.conformal import ConformalPredictor, SklearnConformalPredictor
from ragmat.uncertainty.mc_dropout import MCDropoutUQ
from ragmat.logging_utils import (
    AnomalyLogger, check_ood_scores_valid, check_low_ood_not_all,
    check_mae_range, check_n_samples, check_result_file_collision,
)
from ragmat.explain import ExplainabilityModule

try:
    import wandb
except ImportError:
    wandb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_INDICES_DIR = _PROJECT_ROOT / "data" / "indices"


def _debug_report(
    hypothesis_id: str,
    location: str,
    msg: str,
    data: dict,
    run_id: str = "pre-fix",
) -> None:
    """Best-effort debug reporting for the retrieval silent failure session."""
    env_path = _PROJECT_ROOT / ".dbg" / "retrieval-silent-failure.env"
    url = "http://127.0.0.1:7777/event"
    session_id = "retrieval-silent-failure"
    try:
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line.startswith("DEBUG_SERVER_URL="):
                        url = line.split("=", 1)[1]
                    elif line.startswith("DEBUG_SESSION_ID="):
                        session_id = line.split("=", 1)[1]
        import urllib.request

        payload = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
        }
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
        ).read()
    except Exception:
        pass


def run_evaluation(checkpoint_path: str | Path, config_path: str | Path, output_dir: str | Path) -> None:
    """Run full evaluation."""
    ckpt_path = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig.from_yaml(config_path)
    # #region debug-point E:eval-config-resolution
    _debug_report(
        "E",
        "eval/run_eval.py:run_evaluation",
        "[DEBUG] resolved evaluation config",
        {
            "config_path": str(config_path),
            "checkpoint_path": str(ckpt_path),
            "output_dir": str(out_dir),
            "experiment_name": cfg.experiment_name,
            "tier": cfg.tier,
            "retrieval_mode": cfg.retrieval_mode,
            "fusion_method": cfg.fusion_method,
        },
    )
    # #endregion
    logger.info("Evaluating: %s", cfg.experiment_name)

    if wandb is not None:
        try:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.experiment_name + "_eval",
                config=cfg.to_dict(),
                reinit=True,
            )
        except Exception as exc:
            logger.warning("wandb init failed (proceeding without): %s", exc)

    # 1. Load data
    loader = JARVISLoader()
    data = loader.load(cfg.target_property)
    structures = [d[0] for d in data]
    labels = [d[1] for d in data]
    ids = [d[2] for d in data]

    splitter = DataSplitter(seed=cfg.seed)
    split_filename = f"split_{cfg.split_type}_{cfg.target_property}.json"
    split_path = splitter.splits_dir / split_filename
    if split_path.exists():
        logger.info(f"Loading cached split from {split_path}")
        with open(split_path, "r") as f:
            split_dict = json.load(f)
    else:
        split_dict = splitter.split(ids, labels, structures, cfg.split_type, cfg.target_property)
    train_ids = split_dict["train"]
    test_ids = split_dict["test"]

    id_to_struct = dict(zip(ids, structures))
    id_to_label = dict(zip(ids, labels))

    if cfg.tier == 0:
        _eval_tier0(cfg, ckpt_path, out_dir, train_ids, split_dict['val'], test_ids, id_to_struct, id_to_label)
    else:
        _eval_tier1(cfg, ckpt_path, out_dir, train_ids, test_ids, id_to_struct, id_to_label)


def _eval_tier0(cfg, ckpt_path, out_dir, train_ids, val_ids, test_ids, id_to_struct, id_to_label):
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    if cfg.fusion_method != "cross_attention":
        model = ckpt["model"]
    scaler = ckpt["scaler"]

    featurizer = MatminerFeaturizer(n_jobs=-1)
    test_structs = [id_to_struct[i] for i in test_ids]
    X_test_raw, test_valid_ids = featurizer.featurize_dataset(test_structs, test_ids)
    X_test = featurizer.transform(X_test_raw, scaler)
    y_test = np.array([id_to_label[i] for i in test_valid_ids])

    # Featurize validation set for Conformal Prediction
    val_structs = [id_to_struct[i] for i in val_ids]
    X_val_raw, val_valid_ids = featurizer.featurize_dataset(val_structs, val_ids)
    X_val = featurizer.transform(X_val_raw, scaler)
    y_val = np.array([id_to_label[i] for i in val_valid_ids])

    # Build id_to_label_full from the already-loaded id_to_label (same data, no reload needed)
    id_to_label_full = id_to_label
    train_structs = [id_to_struct[i] for i in train_ids]
    # Featurize training set for retrieval pool
    logger.info("Featurizing training set for retrieval pool (%d structures) ...", len(train_structs))
    X_train_raw, train_valid_ids = featurizer.featurize_dataset(train_structs, train_ids)
    X_train = featurizer.transform(X_train_raw, scaler)
    y_train_full = np.array([id_to_label[i] for i in train_valid_ids])
    
    # Fit OOD detector on base training features
    detector = MahalanobisDetector()
    detector.fit(X_train)
    ood_scores = detector.score(X_test)
    val_ood_scores = detector.score(X_val)

    # Load FAISS index
    if cfg.retrieval_mode in ["true_neighbor", "random_control"]:
        index_name = FAISSIndex.index_name(0, "matminer", cfg.target_property, cfg.split_type)
        index_path = _INDICES_DIR / index_name
        
        if not (Path(str(index_path) + ".index").exists()):
            logger.error("FAISS index not found at %s - cannot use retrieval mode", index_path)
            X_test_concat, X_val_concat = X_test, X_val
        else:
            index = FAISSIndex(dim=X_train.shape[1], property_name=cfg.target_property, split_name=cfg.split_type)
            index.load(str(index_path))
            
            if cfg.retrieval_mode == "true_neighbor":
                X_test_concat = concat_retrieval_features(X_test, index, X_train, train_valid_ids, cfg.top_k, "mean")
                X_val_concat = concat_retrieval_features(X_val, index, X_train, train_valid_ids, cfg.top_k, "mean")
            else:
                X_test_concat = concat_random_retrieval_features(X_test, X_train, cfg.top_k, "mean", cfg.seed + 999)
                X_val_concat = concat_random_retrieval_features(X_val, X_train, cfg.top_k, "mean", cfg.seed + 998)
            assert X_test_concat.shape[1] > X_test.shape[1], f"Concat failed: {X_test_concat.shape} vs {X_test.shape}"
    else:
        X_test_concat, X_val_concat = X_test, X_val
    if cfg.fusion_method == "cross_attention":
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from ragmat.fusion.cross_attention import CrossAttentionFusionHead
        from ragmat.fusion.random_control import RandomRetrievalFusionHead
        device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
        dim = X_train.shape[1]
        base_fusion = CrossAttentionFusionHead(embedding_dim=dim)
        if cfg.retrieval_mode == "random_control":
            fusion_model = RandomRetrievalFusionHead(base_fusion, X_train, top_k=cfg.top_k)
        else:
            fusion_model = base_fusion
        pt_ckpt = torch.load(ckpt_path.with_name(ckpt_path.name.replace("_model.pkl", "_best.pt")), weights_only=False)
        fusion_model.load_state_dict(pt_ckpt["model_state_dict"])
        fusion_model = fusion_model.to(device)
        fusion_model.eval()
        
        class PyTorchWrapper:
            def __init__(self, model, device, top_k, index, train_feats, train_ids, retrieval_mode):
                self.model = model
                self.device = device
                self.top_k = top_k
                self.index = index
                self.train_feats = train_feats
                self.id_to_idx = {mid: idx for idx, mid in enumerate(train_ids)}
                self.retrieval_mode = retrieval_mode
                self.dim = train_feats.shape[1]
            def predict(self, X_input):
                if self.retrieval_mode == "true_neighbor":
                    _, n_ids_nested = self.index.query(X_input, self.top_k)
                    all_n_embs = []
                    for n_ids in n_ids_nested:
                        feats = [self.train_feats[self.id_to_idx[nid]] if nid in self.id_to_idx else np.zeros(self.dim, dtype=np.float32) for nid in n_ids]
                        all_n_embs.append(np.stack(feats))
                    X_n = np.stack(all_n_embs)
                else:
                    X_n = np.zeros((len(X_input), self.top_k, self.dim), dtype=np.float32)
                self.model.eval()
                with torch.no_grad():
                    preds = []
                    ds = TensorDataset(torch.tensor(X_input, dtype=torch.float32), torch.tensor(X_n, dtype=torch.float32))
                    dl = DataLoader(ds, batch_size=128, shuffle=False)
                    for q, n in dl:
                        p = self.model(q.to(self.device), neighbor_embeddings=n.to(self.device))
                        preds.append(p.cpu().numpy())
                    return np.concatenate(preds).flatten()
        
        model = PyTorchWrapper(fusion_model, device, cfg.top_k, index if cfg.retrieval_mode == "true_neighbor" else None, X_train, train_valid_ids, cfg.retrieval_mode)

    if cfg.fusion_method == "cross_attention":
        preds = model.predict(X_test)
        val_preds = model.predict(X_val)
    else:
        preds = model.predict(X_test_concat)
        val_preds = model.predict(X_val_concat)
    
    # --- Adaptive Gating ---
    retrieved_ids = None
    relevant_ids = None
    physical_relevance = None
    gate_stats = {}

    if cfg.retrieval_mode != "none" and 'index' in locals():
        logger.info("Applying Adaptive Gating and Fallback Model...")
        fallback_ckpt = ckpt_path.parent / f"tier0_{cfg.target_property}_{cfg.split_type}_none_model.pkl"
        with open(fallback_ckpt, "rb") as fb:
            base_model = pickle.load(fb)["model"]
        
        base_preds = base_model.predict(X_test)
        base_val_preds = base_model.predict(X_val)

        _, test_n_ids = index.query(X_test, cfg.top_k)
        _, val_n_ids = index.query(X_val, cfg.top_k)
        retrieved_ids = test_n_ids

        id_to_idx = {mid: idx for idx, mid in enumerate(train_valid_ids)}
        
        def get_variance(n_ids):
            vars_out = []
            for ids in n_ids:
                vals = [y_train_full[id_to_idx[nid]] for nid in ids if nid in id_to_idx]
                vars_out.append(np.var(vals) if len(vals) > 1 else 0.0)
            return np.array(vars_out)
        
        gate_mask = None
        if getattr(cfg, 'gating', False):
            gate = AdaptiveGate()
            gate_mask = gate.batch_gate(ood_scores, get_variance(test_n_ids))
            val_gate_mask = gate.batch_gate(val_ood_scores, get_variance(val_n_ids))
            gate.log_stats(gate_mask)
            
            preds = np.where(gate_mask, preds, base_preds)
            val_preds = np.where(val_gate_mask, val_preds, base_val_preds)
        else:
            logger.info("Gating is OFF in config. Using pure retrieval predictions.")
        # Retrieval Metrics
        logger.info("Computing retrieval metrics...")
        relevant_ids = []
        for i, yt in enumerate(y_test):
            margin = max(0.1, 0.1 * abs(yt))
            matches = set()
            for rid in retrieved_ids[i]:
                if rid in id_to_label_full and abs(id_to_label_full[rid] - yt) < margin:
                    matches.add(rid)
            relevant_ids.append(matches)
        logger.info("Retrieval metrics computed.")
            
        logger.info("Initializing ExplainabilityModule...")
        explainer = ExplainabilityModule(top_k=cfg.top_k)
        physical_relevance = []
        for i in range(len(test_valid_ids)):
            n_ids_i = retrieved_ids[i][:cfg.top_k]
            n_feats = []
            n_labels = []
            for nid in n_ids_i:
                if nid in id_to_idx:
                    idx = id_to_idx[nid]
                    n_feats.append(X_train[idx])
                    n_labels.append(y_train_full[idx])
                else:
                    n_feats.append(np.zeros_like(X_test[i]))
                    n_labels.append(float("nan"))
            physical_relevance.append(explainer.explain(X_test[i], n_ids_i, np.array(n_feats), n_labels, test_valid_ids[i])["physical_relevance_score"])
        physical_relevance = np.array(physical_relevance)

    # --- Conformal Prediction ---
    conformal = SklearnConformalPredictor()
    conformal.calibrate(y_val, val_preds, coverage=cfg.uq.conformal_coverage)
    lower, upper = conformal.predict_interval(preds)

    # --- OOD Ground Truth ---
    is_ood_labels = np.ones_like(ood_scores) if cfg.split_type != "iid" else np.zeros_like(ood_scores)
    
    # --- Anomaly checks (spec-mandated) ---
    check_ood_scores_valid(ood_scores, cfg.experiment_name)
    check_low_ood_not_all(ood_scores, cfg.split_type, cfg.experiment_name, threshold=detector.normalized_threshold)
    _all_mae = float(abs(preds - y_test).mean())
    check_mae_range(_all_mae, cfg.split_type, cfg.target_property, cfg.experiment_name)
    
    results = compute_all_metrics(
        y_test, preds, ood_scores,
        lower=lower, upper=upper,
        is_ood_labels=is_ood_labels,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        physical_relevance=physical_relevance,
        ood_threshold=detector.normalized_threshold,
        gate_mask=gate_mask if 'gate_mask' in locals() else None,
        base_preds=base_preds if 'base_preds' in locals() else None,
        pure_retrieval_preds=model.predict(X_test_concat) if cfg.fusion_method != "cross_attention" and 'X_test_concat' in locals() else (model.predict(X_test) if cfg.fusion_method == "cross_attention" else None)
    )

    _ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    res_file = out_dir / f"results_{cfg.experiment_name}_{_ts}.json"
    check_result_file_collision(res_file, cfg.experiment_name)
    # #region debug-point E:tier0-result-path
    _debug_report(
        "E",
        "eval/run_eval.py:_eval_tier0",
        "[DEBUG] tier0 results will be written",
        {
            "retrieval_mode": cfg.retrieval_mode,
            "experiment_name": cfg.experiment_name,
            "result_file": str(res_file),
            "prediction_file": str(out_dir / f"predictions_{cfg.experiment_name}.csv"),
        },
    )
    # #endregion
    with open(res_file, "w") as f:
        json.dump({cfg.experiment_name: results}, f, indent=2)

    if wandb is not None and wandb.run is not None:
        flat_results = {}
        for bin_name, metrics in results.items():
            for k, v in metrics.items():
                flat_results[f"{bin_name}_{k}"] = v
        wandb.log(flat_results)

    df = pd.DataFrame({
        "material_id": test_valid_ids,
        "y_true": y_test,
        "y_pred": preds,
        "ood_score": ood_scores,
    })
    df.to_csv(out_dir / f"predictions_{cfg.experiment_name}.csv", index=False)
    logger.info("Tier 0 eval done.")


def _eval_tier1(cfg, ckpt_path, out_dir, train_ids, test_ids, id_to_struct, id_to_label):
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = CGCNNEncoder(
        hidden_dim=cfg.cgcnn.hidden_dim,
        n_conv_layers=cfg.cgcnn.n_conv_layers,
        dropout_rate=cfg.cgcnn.dropout_rate,
    ).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "encoder_state_dict" in ckpt:
        model.load_state_dict(ckpt["encoder_state_dict"])
    else:
        raise KeyError("Neither model_state_dict nor encoder_state_dict found in checkpoint.")
    model.eval()

    builder = CrystalGraphBuilder(cutoff_radius=cfg.cgcnn.cutoff_radius, n_gaussian_basis=cfg.cgcnn.n_gaussian_basis)
    test_data = builder.build_dataset([id_to_struct[i] for i in test_ids], [id_to_label[i] for i in test_ids], test_ids)
    test_loader = PyGLoader(test_data, batch_size=cfg.cgcnn.batch_size, shuffle=False)
    train_data = builder.build_dataset([id_to_struct[i] for i in train_ids], [id_to_label[i] for i in train_ids], train_ids)
    train_loader = PyGLoader(train_data, batch_size=cfg.cgcnn.batch_size, shuffle=False)

    # Need val_loader for conformal calibration
    # Use cached split to avoid re-running spglib (crash risk)
    loader_full = JARVISLoader()
    data_full = loader_full.load(cfg.target_property)
    ids_full = [d[2] for d in data_full]
    _splitter_t1 = DataSplitter(seed=cfg.seed)
    _split_path_t1 = _splitter_t1.splits_dir / f"split_{cfg.split_type}_{cfg.target_property}.json"
    if _split_path_t1.exists():
        logger.info("Tier 1: Loading cached split from %s", _split_path_t1)
        with open(_split_path_t1, "r") as f:
            split_dict = json.load(f)
    else:
        split_dict = _splitter_t1.split(
            ids_full, [d[1] for d in data_full], [d[0] for d in data_full],
            cfg.split_type, cfg.target_property
        )
    val_ids = split_dict["val"]
    id_to_struct_full = dict(zip(ids_full, [d[0] for d in data_full]))
    id_to_label_full = dict(zip(ids_full, [d[1] for d in data_full]))
    
    val_data = builder.build_dataset([id_to_struct_full[i] for i in val_ids], [id_to_label_full[i] for i in val_ids], val_ids)
    val_loader = PyGLoader(val_data, batch_size=cfg.cgcnn.batch_size, shuffle=False)

    logger.info("Extracting embeddings for Mahalanobis ...")
    train_embs = []
    with torch.no_grad():
        for b in train_loader:
            train_embs.append(model.get_embedding(b.to(device)).cpu().numpy())
    train_embs = np.concatenate(train_embs)
    detector = MahalanobisDetector()
    detector.fit(train_embs)

    # Instantiate Fusion Head if needed
    fusion_model = None
    if cfg.retrieval_mode != "none":
        from ragmat.fusion.concat import ConcatFusionHead
        from ragmat.fusion.cross_attention import CrossAttentionFusionHead
        if cfg.fusion_method == "concat":
            fusion_model = ConcatFusionHead(embedding_dim=cfg.cgcnn.hidden_dim).to(device)
        elif cfg.fusion_method == "cross_attention":
            fusion_model = CrossAttentionFusionHead(embedding_dim=cfg.cgcnn.hidden_dim).to(device)
        
        # Load best fusion head from checkpoint
        if "fusion_state_dict" in ckpt:
            fusion_model.load_state_dict(ckpt["fusion_state_dict"])
            logger.info("Loaded fusion head weights from checkpoint.")
        else:
            logger.warning("No fusion_state_dict found in checkpoint! Using base model only.")
            fusion_model = None
        
        if fusion_model:
            fusion_model.eval()

    # Load FAISS index if needed
    index = None
    if cfg.retrieval_mode != "none":
        index_name = FAISSIndex.index_name(1, "cgcnn", cfg.target_property, cfg.split_type)
        index_path = _INDICES_DIR / index_name
        if Path(str(index_path) + ".index").exists():
            index = FAISSIndex(dim=cfg.cgcnn.hidden_dim, property_name=cfg.target_property, split_name=cfg.split_type)
            index.load(str(index_path))

    def evaluate_loader(loader):
        preds_base, preds_fusion, embs, y_trues, ids_list = [], [], [], [], []
        id_to_idx = {mid: idx for idx, mid in enumerate(train_ids)}
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                p_base, e_base = model(b)
                preds_base.append(p_base.cpu().numpy())
                embs.append(e_base.cpu().numpy())
                y_trues.append(b.y.view(-1, 1).cpu().numpy())
                ids_list.extend(b.material_id)
                
                if index and fusion_model:
                    e_np = e_base.cpu().numpy()
                    _, n_ids_nested = index.query(e_np, cfg.top_k)
                    batch_n_embs = []
                    for n_ids in n_ids_nested:
                        feats = [train_embs[id_to_idx[nid]] if nid in id_to_idx else np.zeros(cfg.cgcnn.hidden_dim, dtype=np.float32) for nid in n_ids]
                        batch_n_embs.append(np.stack(feats))
                    n_embs_tensor = torch.tensor(np.stack(batch_n_embs), device=device, dtype=torch.float32)
                    p_fusion = fusion_model(e_base, neighbor_embeddings=n_embs_tensor)
                    preds_fusion.append(p_fusion.cpu().numpy())
                else:
                    preds_fusion.append(p_base.cpu().numpy())

        return (
            np.concatenate(preds_base).squeeze(),
            np.concatenate(preds_fusion).squeeze(),
            np.concatenate(embs),
            np.concatenate(y_trues).squeeze(),
            ids_list
        )

    logger.info("Evaluating test set...")
    test_preds_base, test_preds, test_embs, y_trues, ids_list = evaluate_loader(test_loader)
    logger.info("Evaluating val set...")
    val_preds_base, val_preds, val_embs, val_trues, _ = evaluate_loader(val_loader)
    
    ood_scores = detector.score(test_embs)
    val_ood_scores = detector.score(val_embs)
    
    # Adaptive Gating
    y_train_full = np.array([id_to_label_full[i] for i in train_ids])
    id_to_idx = {mid: idx for idx, mid in enumerate(train_ids)}
    
    if index and fusion_model:
        _, test_n_ids = index.query(test_embs, cfg.top_k)
        _, val_n_ids = index.query(val_embs, cfg.top_k)
        retrieved_ids = test_n_ids
        
        def get_variance(n_ids):
            vars_out = []
            for ids in n_ids:
                vals = [y_train_full[id_to_idx[nid]] for nid in ids if nid in id_to_idx]
                vars_out.append(np.var(vals) if len(vals) > 1 else 0.0)
            return np.array(vars_out)
            
        gate_mask = None
        if getattr(cfg, 'gating', False):
            gate = AdaptiveGate()
            gate_mask = gate.batch_gate(ood_scores, get_variance(test_n_ids))
            val_gate_mask = gate.batch_gate(val_ood_scores, get_variance(val_n_ids))
            gate.log_stats(gate_mask)
            
            preds_final = np.where(gate_mask, test_preds, test_preds_base)
            val_preds_final = np.where(val_gate_mask, val_preds, val_preds_base)
        else:
            logger.info("Gating is OFF in config.")
            preds_final = test_preds
            val_preds_final = val_preds
    else:
        preds_final = test_preds
        val_preds_final = val_preds
        retrieved_ids = None

    preds = preds_final

    # --- Uncertainty Quantification (UQ) ---
    if index and fusion_model:
        logger.info("Running MC-Dropout UQ on FusionHead...")
        batch_n_embs = []
        for n_ids in retrieved_ids:
            feats = [train_embs[id_to_idx[nid]] if nid in id_to_idx else np.zeros(cfg.cgcnn.hidden_dim, dtype=np.float32) for nid in n_ids]
            batch_n_embs.append(np.stack(feats))
        n_embs_tensor = torch.tensor(np.stack(batch_n_embs), device=device, dtype=torch.float32)
        full_e_base = torch.tensor(test_embs, device=device, dtype=torch.float32)
        
        def forward_fn():
            return fusion_model(full_e_base, neighbor_embeddings=n_embs_tensor)
            
        _, var_preds = MCDropoutUQ.predict_with_uncertainty(fusion_model, forward_fn)
        
        # We must also generate base model variances for the points that were gated out
        logger.info("Running MC-Dropout UQ on base model for gated points (batched)...")
        uq_loader = PyGLoader(test_data, batch_size=cfg.cgcnn.batch_size, shuffle=False)
        all_base_var_preds = []
        for batch in uq_loader:
            batch = batch.to(device)
            def base_forward_fn():
                return model(batch)
            _, base_var_batch = MCDropoutUQ.predict_with_uncertainty(model, base_forward_fn)
            all_base_var_preds.append(base_var_batch.cpu().numpy())
        base_var_preds = np.concatenate(all_base_var_preds, axis=0).squeeze()
        
        var_preds = var_preds.cpu().numpy().squeeze()
        var_preds = np.where(gate_mask, var_preds, base_var_preds)
    else:
        logger.info("Running MC-Dropout UQ on base model (batched)...")
        uq_loader = PyGLoader(test_data, batch_size=cfg.cgcnn.batch_size, shuffle=False)
        all_var_preds = []
        for batch in uq_loader:
            batch = batch.to(device)
            def forward_fn():
                return model(batch)
            _, var_batch = MCDropoutUQ.predict_with_uncertainty(model, forward_fn)
            all_var_preds.append(var_batch.cpu().numpy())
        var_preds = np.concatenate(all_var_preds, axis=0).squeeze()

    logger.info("Running SklearnConformalPredictor calibration...")
    conformal = SklearnConformalPredictor()
    conformal.calibrate(val_trues, val_preds_final, coverage=0.9)
    lower, upper = conformal.predict_interval(preds)

    # --- Retrieval Metrics & Explainability ---
    relevant_ids = None
    physical_relevance = None
    
    if index:
        # Vectorized O(N_test * N_train) → avoids Python-level O(N²) loop memory explosion
        relevant_ids = []
        y_train_arr = y_train_full  # already a numpy array
        train_ids_arr = np.array(train_ids)
        for yt in y_trues:
            margin = max(0.1, 0.1 * abs(float(yt)))
            matches = train_ids_arr[np.abs(y_train_arr - yt) < margin]
            relevant_ids.append(set(matches.tolist()))
            
        explainer = ExplainabilityModule(top_k=cfg.top_k)
        physical_relevance = []
        for i in range(len(ids_list)):
            n_ids = retrieved_ids[i][:cfg.top_k]
            n_feats = []
            n_labels = []
            for nid in n_ids:
                if nid in id_to_idx:
                    idx = id_to_idx[nid]
                    n_feats.append(train_embs[idx])
                    n_labels.append(y_train_full[idx])
                else:
                    n_feats.append(np.zeros_like(test_embs[i]))
                    n_labels.append(float("nan"))
            res = explainer.explain(test_embs[i], n_ids, np.array(n_feats), n_labels, ids_list[i])
            physical_relevance.append(res["physical_relevance_score"])
        physical_relevance = np.array(physical_relevance)
        
    is_ood_labels = np.ones_like(ood_scores) if cfg.split_type != "iid" else np.zeros_like(ood_scores)

    results = compute_all_metrics(
        y_trues, preds, ood_scores,
        variance=var_preds,
        lower=lower,
        upper=upper,
        is_ood_labels=is_ood_labels,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        physical_relevance=physical_relevance,
        ood_threshold=detector.normalized_threshold,
        gate_mask=gate_mask if 'gate_mask' in locals() else None,
        base_preds=test_preds_base if 'test_preds_base' in locals() else None,
        pure_retrieval_preds=test_preds if 'test_preds' in locals() else None
    )
    _ts1 = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    res_file = out_dir / f"results_{cfg.experiment_name}_{_ts1}.json"
    check_result_file_collision(res_file, cfg.experiment_name)
    with open(res_file, "w") as f:
        json.dump({cfg.experiment_name: results}, f, indent=2)

    if wandb is not None and wandb.run is not None:
        flat_results = {}
        for bin_name, metrics in results.items():
            for k, v in metrics.items():
                flat_results[f"{bin_name}_{k}"] = v
        wandb.log(flat_results)

    df = pd.DataFrame({
        "material_id": ids_list,
        "y_true": y_trues,
        "y_pred": preds,
        "ood_score": ood_scores,
    })
    df.to_csv(out_dir / f"predictions_{cfg.experiment_name}.csv", index=False)
    logger.info("Tier 1 eval done. Results saved to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.config, args.output_dir)
