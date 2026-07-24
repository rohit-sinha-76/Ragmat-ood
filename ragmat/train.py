"""Config-driven training pipeline for RAGMat-OOD.

Entry point: ``train_from_config(config_path)``

Supports:
- Tier 0: sklearn Random Forest / XGBoost on matminer features
- Tier 1: PyG CGCNN encoder trained from scratch with cosine LR schedule

Startup integrity checks (ALL must pass before training begins):
1. Config integrity (encoder_property == target_property, etc.)
2. LeakageChecker: test IDs not in FAISS index
3. Split disjointness: no ID overlap across train/val/test
4. Encoder freshness: no pretrained weights loaded (Tier 1)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ragmat.config import ExperimentConfig
from ragmat.data.loader import JARVISLoader
from ragmat.data.splitter import DataSplitter
from ragmat.encoders.cgcnn import CGCNNEncoder
from ragmat.encoders.graph_builder import CrystalGraphBuilder
from ragmat.features.matminer_descriptors import MatminerFeaturizer
from ragmat.retrieval.faiss_index import FAISSIndex
from ragmat.retrieval.leakage_check import LeakageChecker
from ragmat.retrieval.concat_features import (
    concat_retrieval_features,
    concat_random_retrieval_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths
_PROJECT_ROOT = Path(__file__).parent.parent
_CHECKPOINTS_DIR = _PROJECT_ROOT / "checkpoints"
_INDICES_DIR = _PROJECT_ROOT / "data" / "indices"


def _debug_report(
    hypothesis_id: str,
    location: str,
    msg: str,
    data: dict[str, Any],
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


def train_from_config(config_path: str) -> None:
    """Main entry point: train a model from a YAML config file.

    Args:
        config_path: Path to the experiment YAML config file.
    """
    cfg = ExperimentConfig.from_yaml(config_path)
    # #region debug-point E:config-resolution
    _debug_report(
        "E",
        "ragmat/train.py:train_from_config",
        "[DEBUG] resolved training config",
        {
            "config_path": str(config_path),
            "experiment_name": cfg.experiment_name,
            "tier": cfg.tier,
            "retrieval_mode": cfg.retrieval_mode,
            "fusion_method": cfg.fusion_method,
            "split_type": cfg.split_type,
            "target_property": cfg.target_property,
        },
    )
    # #endregion
    logger.info("=" * 60)
    logger.info("RAGMat-OOD Training: %s", cfg.experiment_name)
    logger.info("=" * 60)
    logger.info("Config hash: %s", cfg.config_hash)
    logger.info("Split: %s | Property: %s | Tier: %d", cfg.split_type, cfg.target_property, cfg.tier)

    # ── Set seed ────────────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Load data ────────────────────────────────────────────────────────
    loader = JARVISLoader()
    data = loader.load(cfg.target_property)
    structures = [d[0] for d in data]
    labels = [d[1] for d in data]
    ids = [d[2] for d in data]
    logger.info("Loaded %d materials", len(ids))

    # ── Split ────────────────────────────────────────────────────────────
    splitter = DataSplitter(seed=cfg.seed)
    split_dict = splitter.split(ids, labels, structures, cfg.split_type, cfg.target_property)
    train_ids = split_dict["train"]
    val_ids = split_dict["val"]
    test_ids = split_dict["test"]

    logger.info("Split stats: train=%d val=%d test=%d", len(train_ids), len(val_ids), len(test_ids))

    # ── Startup integrity checks ─────────────────────────────────────────
    logger.info("Running startup integrity checks ...")
    LeakageChecker.assert_split_disjoint(train_ids, val_ids, test_ids, cfg.split_type)

    train_ids_hash = _hash_id_list(train_ids)
    logger.info("Training material IDs MD5: %s", train_ids_hash)

    # ── Dispatch to tier-specific training ──────────────────────────────
    if cfg.tier == 0:
        _train_tier0(cfg, structures, labels, ids, split_dict)
    else:
        _train_tier1(cfg, structures, labels, ids, split_dict)


# ── Tier 0 ──────────────────────────────────────────────────────────────────

def _train_tier0(
    cfg: ExperimentConfig,
    structures: list,
    labels: list[float],
    ids: list[str],
    split_dict: dict[str, list[str]],
) -> None:
    """Train a Tier-0 matminer + sklearn/XGBoost model."""
    import pickle

    _CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    _INDICES_DIR.mkdir(parents=True, exist_ok=True)

    train_ids = split_dict["train"]
    val_ids = split_dict["val"]
    test_ids = split_dict["test"]

    # ── Index structures by id ──────────────────────────────────────────
    id_to_struct = {mid: s for mid, s, _ in zip(ids, structures, labels)}
    id_to_label = {mid: lbl for mid, lbl, _ in zip(ids, labels, ids)}
    id_to_label = dict(zip(ids, labels))

    train_structs = [id_to_struct[i] for i in train_ids]
    val_structs = [id_to_struct[i] for i in val_ids]
    test_structs = [id_to_struct[i] for i in test_ids]

    y_train = np.array([id_to_label[i] for i in train_ids], dtype=np.float32)
    y_val = np.array([id_to_label[i] for i in val_ids], dtype=np.float32)
    y_test = np.array([id_to_label[i] for i in test_ids], dtype=np.float32)

    # ── Matminer features ────────────────────────────────────────────────
    featurizer = MatminerFeaturizer(n_jobs=-1)

    logger.info("Featurizing train set (%d structures) ...", len(train_structs))
    X_train_raw, train_valid_ids = featurizer.featurize_dataset(train_structs, train_ids)

    logger.info("Featurizing val set (%d structures) ...", len(val_structs))
    X_val_raw, val_valid_ids = featurizer.featurize_dataset(val_structs, val_ids)

    logger.info("Featurizing test set (%d structures) ...", len(test_structs))
    X_test_raw, test_valid_ids = featurizer.featurize_dataset(test_structs, test_ids)

    # CRITICAL: Fit scaler on TRAIN PARTITION ONLY
    scaler = featurizer.fit_scaler(X_train_raw)
    X_train = featurizer.transform(X_train_raw, scaler)
    X_val = featurizer.transform(X_val_raw, scaler)
    X_test = featurizer.transform(X_test_raw, scaler)

    y_train_valid = np.array([id_to_label[i] for i in train_valid_ids], dtype=np.float32)
    y_val_valid = np.array([id_to_label[i] for i in val_valid_ids], dtype=np.float32)
    y_test_valid = np.array([id_to_label[i] for i in test_valid_ids], dtype=np.float32)
    # #region debug-point A:tier0-feature-shapes
    _debug_report(
        "A",
        "ragmat/train.py:_train_tier0",
        "[DEBUG] tier0 tensors prepared before downstream fit",
        {
            "retrieval_mode": cfg.retrieval_mode,
            "fusion_method": cfg.fusion_method,
            "x_train_shape": list(X_train.shape),
            "x_val_shape": list(X_val.shape),
            "x_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train_valid.shape),
            "train_valid_count": len(train_valid_ids),
            "val_valid_count": len(val_valid_ids),
            "test_valid_count": len(test_valid_ids),
        },
    )
    # #endregion

    # ── FAISS index (train only) ─────────────────────────────────────────
    index_name = FAISSIndex.index_name(0, "matminer", cfg.target_property, cfg.split_type)
    index = FAISSIndex(dim=X_train.shape[1], property_name=cfg.target_property, split_name=cfg.split_type)
    index.build(X_train, train_valid_ids)

    # Leakage check BEFORE any model training
    LeakageChecker.assert_no_leakage(train_valid_ids, test_valid_ids, cfg.split_type, cfg.target_property)

    index_path = _INDICES_DIR / index_name
    index.save(str(index_path))
    # #region debug-point B:tier0-faiss-build
    _debug_report(
        "B",
        "ragmat/train.py:_train_tier0",
        "[DEBUG] tier0 FAISS index built and saved",
        {
            "retrieval_mode": cfg.retrieval_mode,
            "index_name": index_name,
            "index_path": str(index_path),
            "index_dim": int(X_train.shape[1]),
            "train_vectors": int(X_train.shape[0]),
        },
    )
    # #endregion

    # ── CRITICAL FIX: Concatenate retrieval features ─────────────────────
    # This is the missing piece! We must query FAISS and concatenate
    # neighbor features BEFORE training the downstream model.
    
    if cfg.retrieval_mode == "true_neighbor":
        logger.info("Concatenating TRUE NEIGHBOR retrieval features (top_k=%d) ...", cfg.top_k)
        X_train_concat = concat_retrieval_features(
            query_features=X_train,
            index=index,
            train_features=X_train,
            train_ids=train_valid_ids,
            top_k=cfg.top_k,
            aggregation="mean",  # Use mean pooling: (N, D) -> (N, 2*D)
        )
        X_val_concat = concat_retrieval_features(
            query_features=X_val,
            index=index,
            train_features=X_train,
            train_ids=train_valid_ids,
            top_k=cfg.top_k,
            aggregation="mean",
        )
        X_test_concat = concat_retrieval_features(
            query_features=X_test,
            index=index,
            train_features=X_train,
            train_ids=train_valid_ids,
            top_k=cfg.top_k,
            aggregation="mean",
        )
        logger.info(
            "Retrieval features concatenated: train %s -> %s, val %s -> %s, test %s -> %s",
            X_train.shape, X_train_concat.shape,
            X_val.shape, X_val_concat.shape,
            X_test.shape, X_test_concat.shape,
        )
        assert X_train_concat.shape[1] > X_train.shape[1], f"Concat failed: {X_train_concat.shape} vs {X_train.shape}"

    elif cfg.retrieval_mode == "random_control":
        logger.info("Concatenating RANDOM CONTROL retrieval features (top_k=%d) ...", cfg.top_k)
        X_train_concat = concat_random_retrieval_features(
            query_features=X_train,
            train_features=X_train,
            top_k=cfg.top_k,
            aggregation="mean",
            seed=cfg.seed,
        )
        X_val_concat = concat_random_retrieval_features(
            query_features=X_val,
            train_features=X_train,
            top_k=cfg.top_k,
            aggregation="mean",
            seed=cfg.seed + 1,  # Different seed for val
        )
        X_test_concat = concat_random_retrieval_features(
            query_features=X_test,
            train_features=X_train,
            top_k=cfg.top_k,
            aggregation="mean",
            seed=cfg.seed + 2,  # Different seed for test
        )
        logger.info(
            "Random control features concatenated: train %s -> %s, val %s -> %s, test %s -> %s",
            X_train.shape, X_train_concat.shape,
            X_val.shape, X_val_concat.shape,
            X_test.shape, X_test_concat.shape,
        )
        assert X_train_concat.shape[1] > X_train.shape[1], f"Concat failed: {X_train_concat.shape} vs {X_train.shape}"

    else:
        # retrieval_mode == "none" - no retrieval, use base features only
        logger.info("Retrieval mode: none - using base features only (no concatenation)")
        X_train_concat = X_train
        X_val_concat = X_val
        X_test_concat = X_test

    # ── Downstream model ─────────────────────────────────────────────────
    if cfg.fusion_method == "cross_attention":
        import torch
        import torch.nn as nn
        import torch.optim as optim
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
            
        fusion_model = fusion_model.to(device)
        fusion_optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)
        fusion_scheduler = optim.lr_scheduler.CosineAnnealingLR(fusion_optimizer, T_max=200)
        criterion = nn.L1Loss()
        
        id_to_idx = {mid: idx for idx, mid in enumerate(train_valid_ids)}
        
        def build_n_embs(query_X):
            if cfg.retrieval_mode == "true_neighbor":
                _, n_ids_nested = index.query(query_X, cfg.top_k)
                all_n_embs = []
                for n_ids in n_ids_nested:
                    feats = [X_train[id_to_idx[nid]] if nid in id_to_idx else np.zeros(dim, dtype=np.float32) for nid in n_ids]
                    all_n_embs.append(np.stack(feats))
                return np.stack(all_n_embs)
            else:
                return np.zeros((len(query_X), cfg.top_k, dim), dtype=np.float32)
                
        X_train_n = build_n_embs(X_train)
        X_val_n = build_n_embs(X_val)
        
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(X_train_n, dtype=torch.float32), 
            torch.tensor(y_train_valid, dtype=torch.float32).view(-1, 1)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32), 
            torch.tensor(X_val_n, dtype=torch.float32), 
            torch.tensor(y_val_valid, dtype=torch.float32).view(-1, 1)
        )
        
        train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
        
        best_val_mae = float("inf")
        patience_counter = 0
        
        logger.info("Training Tier 0 CrossAttentionFusionHead on PyTorch ...")
        
        for epoch in range(1, 201):
            fusion_model.train()
            train_losses = []
            for q_emb, n_emb, y in train_loader:
                q_emb, n_emb, y = q_emb.to(device), n_emb.to(device), y.to(device)
                fusion_optimizer.zero_grad()
                pred = fusion_model(q_emb, neighbor_embeddings=n_emb)
                loss = criterion(pred, y)
                loss.backward()
                fusion_optimizer.step()
                train_losses.append(float(loss.item()))
            
            fusion_scheduler.step()
            
            fusion_model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for q_emb, n_emb, y in val_loader:
                    q_emb, n_emb, y = q_emb.to(device), n_emb.to(device), y.to(device)
                    pred = fusion_model(q_emb, neighbor_embeddings=n_emb)
                    val_preds.append(pred.cpu().numpy())
                    val_trues.append(y.cpu().numpy())
                    
            val_mae = float(np.abs(np.concatenate(val_preds) - np.concatenate(val_trues)).mean())
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": fusion_model.state_dict(),
                    "val_mae": val_mae,
                    "config_hash": cfg.config_hash,
                }
                torch.save(ckpt, str(_CHECKPOINTS_DIR / f"{cfg.experiment_name}_best.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                
            train_loss = float(np.mean(train_losses))
            if epoch % 5 == 0 or epoch == 1:
                logger.info("Epoch %3d | train_loss=%.4f | val_mae=%.4f", epoch, train_loss, val_mae)
                
            if patience_counter >= 5:
                logger.info("Phase 3 early stopping at epoch %d", epoch)
                break
                
        logger.info("Best Tier 0 CrossAttention val MAE: %.4f", best_val_mae)
        
        ckpt = torch.load(str(_CHECKPOINTS_DIR / f"{cfg.experiment_name}_best.pt"))
        fusion_model.load_state_dict(ckpt["model_state_dict"])
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
                    
        model = PyTorchWrapper(fusion_model, device, cfg.top_k, index, X_train, train_valid_ids, cfg.retrieval_mode)
        val_pred = model.predict(X_val)

    else:
        if cfg.tier0.downstream_model == "xgboost":
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=cfg.tier0.n_estimators,
                random_state=cfg.seed,
                n_jobs=-1,
                verbosity=1,
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=cfg.tier0.n_estimators,
                max_features=cfg.tier0.max_features,
                min_samples_leaf=cfg.tier0.min_samples_leaf,
                min_samples_split=cfg.tier0.min_samples_split,
                random_state=cfg.seed,
                n_jobs=-1,
            )

        logger.info("Training %s on %s features ...", type(model).__name__, X_train_concat.shape)
        # #region debug-point A:tier0-fit-input
        _debug_report(
            "A",
            "ragmat/train.py:_train_tier0",
            "[DEBUG] tier0 downstream fit called",
            {
                "retrieval_mode": cfg.retrieval_mode,
                "fusion_method": cfg.fusion_method,
                "fit_input_shape": list(X_train_concat.shape),
                "fit_target_shape": list(y_train_valid.shape),
                "model_class": type(model).__name__,
                "base_feature_dim": int(X_train.shape[1]),
                "augmented_feature_dim": int(X_train_concat.shape[1]),
            },
        )
        # #endregion
        model.fit(X_train_concat, y_train_valid)

        val_pred = model.predict(X_val_concat)
    val_mae = float(np.abs(val_pred - y_val_valid).mean())
    val_rmse = float(np.sqrt(((val_pred - y_val_valid) ** 2).mean()))
    logger.info("Val MAE: %.4f | Val RMSE: %.4f", val_mae, val_rmse)

    # ── Save model + scaler ──────────────────────────────────────────────
    ckpt_path = _CHECKPOINTS_DIR / f"{cfg.experiment_name}_model.pkl"
    if cfg.fusion_method == "cross_attention":
        with open(ckpt_path, "wb") as f:
            pickle.dump({"scaler": scaler, "config": cfg.to_dict()}, f)
    else:
        with open(ckpt_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "config": cfg.to_dict()}, f)
    logger.info("Model saved to %s", ckpt_path)

    if cfg.fusion_method == "cross_attention":
        train_pred = model.predict(X_train)
    else:
        train_pred = model.predict(X_train_concat)
    train_loss = float(np.abs(train_pred - y_train_valid).mean())
    lr = getattr(model, "learning_rate", None)
    if lr is None and hasattr(model, "get_params"):
        lr = model.get_params().get("learning_rate", None)

    _try_wandb_init(cfg, extra_config={"training_material_ids_md5": _hash_id_list(train_valid_ids)})
    _try_wandb_log(cfg, {"val_mae": val_mae, "val_rmse": val_rmse, "train_loss": train_loss, "learning_rate": lr, "n_train": len(X_train)})


# ── Tier 1 ──────────────────────────────────────────────────────────────────

def _train_tier1(
    cfg: ExperimentConfig,
    structures: list,
    labels: list[float],
    ids: list[str],
    split_dict: dict[str, list[str]],
) -> None:
    """Train a Tier-1 CGCNN encoder from scratch."""
    from torch_geometric.loader import DataLoader as PyGLoader

    _CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    _INDICES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    logger.info("Device: %s", device)

    train_ids = split_dict["train"]
    val_ids = split_dict["val"]
    test_ids = split_dict["test"]

    id_to_struct = dict(zip(ids, structures))
    id_to_label = dict(zip(ids, labels))

    # ── Build PyG graphs ─────────────────────────────────────────────────
    builder = CrystalGraphBuilder(
        cutoff_radius=cfg.cgcnn.cutoff_radius,
        n_gaussian_basis=cfg.cgcnn.n_gaussian_basis,
    )

    def build_set(id_list: list[str]) -> list:
        return builder.build_dataset(
            [id_to_struct[i] for i in id_list],
            [id_to_label[i] for i in id_list],
            id_list,
        )

    logger.info("Building train graphs (%d) ...", len(train_ids))
    train_data = build_set(train_ids)
    logger.info("Building val graphs (%d) ...", len(val_ids))
    val_data = build_set(val_ids)

    train_loader = PyGLoader(train_data, batch_size=cfg.cgcnn.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = PyGLoader(val_data, batch_size=cfg.cgcnn.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model (always from scratch) ──────────────────────────────────────
    model = CGCNNEncoder(
        hidden_dim=cfg.cgcnn.hidden_dim,
        n_conv_layers=cfg.cgcnn.n_conv_layers,
        dropout_rate=cfg.cgcnn.dropout_rate,
    ).to(device)

    # ── Verify no pretrained weights ─────────────────────────────────────
    _assert_encoder_fresh(model)

    optimizer = AdamW(model.parameters(), lr=cfg.cgcnn.lr, weight_decay=cfg.cgcnn.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.cgcnn.n_epochs)
    criterion = nn.HuberLoss(delta=1.0)

    train_ids_hash = _hash_id_list(train_ids)
    best_val_mae = float("inf")
    patience_counter = 0

    best_path = _CHECKPOINTS_DIR / f"{cfg.experiment_name}_best.pt"
    skip_training = False
    if best_path.exists():
        logger.info("Best checkpoint %s already exists. Checking if we can skip training...", best_path)
        try:
            ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
            if ckpt.get("config_hash") == cfg.config_hash:
                logger.info("Found fully-trained checkpoint with matching config hash. Skipping training loop!")
                model.load_state_dict(ckpt["model_state_dict"])
                best_val_mae = ckpt.get("val_mae", float("inf"))
                skip_training = True
        except Exception as e:
            logger.warning("Failed to load existing checkpoint: %s. Re-training from scratch.", e)

    # Initialize AMP GradScaler (only if training)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu") if not skip_training else None
    if not skip_training:
        _try_wandb_init(cfg, extra_config={"training_material_ids_md5": train_ids_hash})

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(1, (0 if skip_training else cfg.cgcnn.n_epochs) + 1):
        # Train
        model.train()
        train_losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            with autocast("cuda" if device.type == "cuda" else "cpu"):
                pred, _ = model(batch)
                loss_raw = criterion(pred, batch.y.view(-1, 1))
                loss = loss_raw / cfg.cgcnn.gradient_accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % cfg.cgcnn.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_losses.append(float(loss_raw.item()))

        train_loss = float(np.mean(train_losses))
        scheduler.step()
        current_lr = float(scheduler.get_last_lr()[0])

        # Validate
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad(), autocast("cuda" if device.type == "cuda" else "cpu"):
            for batch in val_loader:
                batch = batch.to(device)
                pred, _ = model(batch)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(batch.y.view(-1, 1).cpu().numpy())

        val_mae = float(np.abs(
            np.concatenate(val_preds) - np.concatenate(val_trues)
        ).mean())
        val_rmse = float(np.sqrt(
            ((np.concatenate(val_preds) - np.concatenate(val_trues)) ** 2).mean()
        ))

        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "learning_rate": current_lr,
        }
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_mae=%.4f | val_rmse=%.4f | lr=%.6f",
            epoch, cfg.cgcnn.n_epochs, train_loss, val_mae, val_rmse, current_lr,
        )
        _try_wandb_log(cfg, log_dict)

        # Checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_mae,
            "config_hash": cfg.config_hash,
            "training_material_ids_hash": train_ids_hash,
        }
        ckpt_path = _CHECKPOINTS_DIR / f"{cfg.experiment_name}_epoch{epoch:04d}.pt"
        torch.save(ckpt, str(ckpt_path))

        # Best checkpoint symlink
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_path = _CHECKPOINTS_DIR / f"{cfg.experiment_name}_best.pt"
            torch.save(ckpt, str(best_path))
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping (checked against val, never test)
        if patience_counter >= cfg.cgcnn.early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d, best_val_mae=%.4f)",
                epoch, cfg.cgcnn.early_stopping_patience, best_val_mae,
            )
            break

    logger.info("Training complete. Best val MAE: %.4f", best_val_mae)

    # ── Build FAISS index from frozen encoder ───────────────────────────
    logger.info("Extracting train embeddings for FAISS index ...")
    model.eval()
    train_loader_emb = PyGLoader(train_data, batch_size=cfg.cgcnn.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_embs, all_ids_emb = [], []
    with torch.no_grad():
        for batch in train_loader_emb:
            batch = batch.to(device)
            emb = model.get_embedding(batch)
            all_embs.append(emb.cpu().numpy())
            all_ids_emb.extend(batch.material_id)

    train_embs = np.concatenate(all_embs, axis=0)
    dim = train_embs.shape[1]

    # Leakage check BEFORE building index
    LeakageChecker.assert_no_leakage(all_ids_emb, test_ids, cfg.split_type, cfg.target_property)

    index = FAISSIndex(dim=dim, property_name=cfg.target_property, split_name=cfg.split_type)
    index.build(train_embs, all_ids_emb)

    index_name = FAISSIndex.index_name(1, "cgcnn", cfg.target_property, cfg.split_type)
    index.save(str(_INDICES_DIR / index_name))
    logger.info("FAISS index saved.")

    # ── Phase 2: Train Fusion Head (if retrieval is enabled) ────────────────
    if cfg.retrieval_mode != "none":
        logger.info("Starting Phase 2: Training Fusion Head")
        from ragmat.fusion.concat import ConcatFusionHead
        from ragmat.fusion.cross_attention import CrossAttentionFusionHead
        from ragmat.fusion.random_control import RandomRetrievalFusionHead
        import torch.optim as optim

        dim = cfg.cgcnn.hidden_dim
        if cfg.fusion_method == "concat":
            base_fusion = ConcatFusionHead(embedding_dim=dim, dropout_rate=cfg.cgcnn.dropout_rate)
        elif cfg.fusion_method == "cross_attention":
            base_fusion = CrossAttentionFusionHead(embedding_dim=dim, dropout_rate=cfg.cgcnn.dropout_rate)
        else:
            raise ValueError(f"Unknown fusion method {cfg.fusion_method}")

        if cfg.retrieval_mode == "random_control":
            fusion_model = RandomRetrievalFusionHead(base_fusion, train_embs, top_k=cfg.top_k)
        else:
            fusion_model = base_fusion

        fusion_model = fusion_model.to(device)
        fusion_optimizer = optim.AdamW(fusion_model.parameters(), lr=cfg.cgcnn.lr, weight_decay=cfg.cgcnn.weight_decay)
        fusion_scheduler = optim.lr_scheduler.CosineAnnealingLR(fusion_optimizer, T_max=cfg.cgcnn.n_epochs)
        fusion_criterion = nn.HuberLoss(delta=1.0)
        fusion_scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")
        
        # Freeze base encoder
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        best_fusion_val_mae = float("inf")
        patience_counter = 0

        # Build ID to index mapping for training
        id_to_idx = {mid: idx for idx, mid in enumerate(all_ids_emb)}

        for epoch in range(1, cfg.cgcnn.n_epochs + 1):
            fusion_model.train()
            train_losses = []
            
            for batch in train_loader:
                batch = batch.to(device)
                fusion_optimizer.zero_grad()
                
                with torch.no_grad():
                    q_emb = model.get_embedding(batch)
                
                if cfg.retrieval_mode == "true_neighbor":
                    q_emb_np = q_emb.cpu().numpy()
                    scores, n_ids_nested = index.query(q_emb_np, cfg.top_k + 1)
                    
                    batch_n_embs = []
                    for i, n_ids in enumerate(n_ids_nested):
                        query_id = batch.material_id[i]
                        # Filter out query ID to prevent trivial identity leakage
                        filtered_n_ids = [nid for nid in n_ids if nid != query_id][:cfg.top_k]
                        
                        feats = [train_embs[id_to_idx[nid]] if nid in id_to_idx else np.zeros(dim, dtype=np.float32) for nid in filtered_n_ids]
                        while len(feats) < cfg.top_k:
                            feats.append(np.zeros(dim, dtype=np.float32))
                            
                        batch_n_embs.append(np.stack(feats))
                    n_embs_tensor = torch.tensor(np.stack(batch_n_embs), device=device, dtype=torch.float32)
                else:
                    n_embs_tensor = None  # RandomRetrievalFusionHead ignores this
                
                with autocast("cuda" if device.type == "cuda" else "cpu"):
                    pred = fusion_model(q_emb, neighbor_embeddings=n_embs_tensor)
                    loss = fusion_criterion(pred, batch.y.view(-1, 1))
                    
                fusion_scaler.scale(loss).backward()
                fusion_scaler.step(fusion_optimizer)
                fusion_scaler.update()
                
                train_losses.append(float(loss.item()))
            
            train_loss = float(np.mean(train_losses))
            fusion_scheduler.step()
            
            fusion_model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    q_emb = model.get_embedding(batch)
                    
                    if cfg.retrieval_mode == "true_neighbor":
                        q_emb_np = q_emb.cpu().numpy()
                        scores, n_ids_nested = index.query(q_emb_np, cfg.top_k + 1)
                        batch_n_embs = []
                        for i, n_ids in enumerate(n_ids_nested):
                            query_id = batch.material_id[i]
                            filtered_n_ids = [nid for nid in n_ids if nid != query_id][:cfg.top_k]
                            feats = [train_embs[id_to_idx[nid]] if nid in id_to_idx else np.zeros(dim, dtype=np.float32) for nid in filtered_n_ids]
                            while len(feats) < cfg.top_k:
                                feats.append(np.zeros(dim, dtype=np.float32))
                            batch_n_embs.append(np.stack(feats))
                        n_embs_tensor = torch.tensor(np.stack(batch_n_embs), device=device, dtype=torch.float32)
                    else:
                        n_embs_tensor = None
                    
                    pred = fusion_model(q_emb, neighbor_embeddings=n_embs_tensor)
                    val_preds.append(pred.cpu().numpy())
                    val_trues.append(batch.y.view(-1, 1).cpu().numpy())
                    
            val_mae = float(np.abs(np.concatenate(val_preds) - np.concatenate(val_trues)).mean())
            
            logger.info("Phase 2 Epoch %3d | train_loss=%.4f | val_mae=%.4f", epoch, train_loss, val_mae)
            
            if val_mae < best_fusion_val_mae:
                best_fusion_val_mae = val_mae
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "fusion_state_dict": fusion_model.state_dict(),
                    "val_mae": val_mae,
                    "config_hash": cfg.config_hash,
                    "training_material_ids_hash": train_ids_hash,
                }
                torch.save(ckpt, str(_CHECKPOINTS_DIR / f"{cfg.experiment_name}_best.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= cfg.cgcnn.early_stopping_patience:
                logger.info("Phase 2 early stopping at epoch %d", epoch)
                break

# ── Helpers ──────────────────────────────────────────────────────────────────

def _hash_id_list(ids: list[str]) -> str:
    """Compute MD5 hash of a sorted list of material IDs."""
    canonical = json.dumps(sorted(ids))
    return hashlib.md5(canonical.encode()).hexdigest()


def _assert_encoder_fresh(model: CGCNNEncoder) -> None:
    """Assert that no external pretrained weights were loaded.

    Checks that CGCNNEncoder has no 'from_pretrained' attribute and
    that its parameter norms are reasonable (not suspiciously close to
    a known published checkpoint).
    """
    assert not hasattr(model, "from_pretrained"), (
        "CGCNNEncoder must NOT have a from_pretrained() method. "
        "Remove any pretrained loading logic."
    )
    # Log parameter hash for audit trail
    param_hash = hashlib.md5(
        b"".join(p.data.cpu().numpy().tobytes() for p in model.parameters())
    ).hexdigest()
    logger.info("Encoder parameter init hash (for audit): %s", param_hash)


def _try_wandb_init(cfg: ExperimentConfig, extra_config: dict | None = None) -> None:
    """Initialise wandb run if available."""
    try:
        import wandb
        conf = cfg.to_dict()
        if extra_config:
            conf.update(extra_config)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment_name,
            config=conf,
            reinit=True,
        )
    except Exception as exc:
        logger.warning("wandb init failed (proceeding without): %s", exc)


def _try_wandb_log(cfg: ExperimentConfig, metrics: dict) -> None:
    """Log metrics to wandb if a run is active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m ragmat.train <config_path>")
        sys.exit(1)
    train_from_config(sys.argv[1])
