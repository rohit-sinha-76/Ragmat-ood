# Experiments Documentation

This document describes the dataset versioning, partitioning, experiment execution matrix, and hardware/runtime statistics for the RAGMat-OOD project.

---

## 1. Dataset Reference

- **Dataset**: JARVIS-DFT (`dft_3d` dataset)
- **Version**: `2026.6.12` (released 2026-06-12)
- **Download Command**:
 ```python
 from jarvis.db.figshare import data as jarvis_data
 raw = jarvis_data("dft_3d")
 ```
- **Filter Applied**: Skips any entries where the target property field value is missing, equal to `"na"`, or is `NaN`, or where structural data (`"atoms"`) is missing or fails conversion to a `pymatgen.Structure` object.
- **Dataset Size**: `93,902` total crystal structures.

---

## 2. Split Definitions

The exact split partition sizes are loaded from `data/splits/` JSON files:

| Split Filename | Target Property | Split Type | Total N | Train N | Val N | Test N |
|---|---|---|---|---|---|---|
| `split_iid_formation_energy.json` | formation_energy | IID | 93,902 | 65,732 | 9,390 | 18,780 |
| `split_iid_band_gap.json` | band_gap | IID | 93,902 | 65,733 | 9,389 | 18,780 |
| `split_family_out_formation_energy.json` | formation_energy | family_out | 93,902 | 64,089 | 7,120 | 22,693 |
| `split_family_out_band_gap.json` | band_gap | family_out | 93,902 | 64,089 | 7,120 | 22,693 |
| `split_element_out_formation_energy.json` | formation_energy | element_out | 93,902 | 59,392 | 6,599 | 27,911 |
| `split_element_out_band_gap.json` | band_gap | element_out | 93,902 | 59,392 | 6,599 | 27,911 |

- **IID Split**: Random 70% train, 10% validation, 20% test partition.
- **Family-Out Split**: Splitting based on crystal prototype families. Unseen families in training are held out.
- **Element-Out Split**: Excludes 15 elements from the training partition entirely, creating extreme out-of-distribution test queries.

---

## 3. Experiment Matrix

The following combinations were executed in the project:

### Tier 0 (Classical Baselines)
- **Model**: Random Forest Regressor / XGBoost Regressor (Magpie composition features + CrystalNN structural descriptors)
- **Target Properties**: `formation_energy`, `band_gap`
- **Splits**: `iid`, `family_out`, `element_out`
- **Configurations**:
 - `none`: No retrieval
 - `true_neighbor` (with mean concatenation): Retrieval-augmented with FAISS cosine neighbors
 - `random_control` (with mean concatenation): Baseline retrieval with random neighbors
 - `true_neighbor` (with cross-attention): Retrieval-augmented with cross-attention fusion head
 - `random_control` (with cross-attention): Baseline retrieval with random cross-attention

### Tier 1 (Graph Neural Networks)
- **Model**: Crystal Graph Neural Network (CGCNN)
- **Target Properties**: `formation_energy`, `band_gap`
- **Splits**: `iid`, `family_out`, `element_out`
- **Configurations**:
 - `base` (No retrieval): GNN trained from scratch
 - `true_neighbor`: GNN + retrieval-augmented head (`concat` for `formation_energy`, `cross_attention` for `band_gap`)
 - `random_control`: GNN + random neighbor control head

### Gating and Uncertainty Quantification
- **Mahalanobis Gate**: Fits OOD detector on IID train embeddings to gate test query predictions between Tier 1 GNN and Tier 0 RF fallback models.
- **Conformal Calibration**: Computes coverage intervals for all properties and splits under Conformal and Ensemble Variance models.
- **ZSNI (Zero-Shot Neighbor Imputation)**: Imputes missing GNN elemental embeddings by averaging the features of top-k nearest seen elements on physical characteristics or Pettifor coordinates.

---

## 4. Hyperparameters

The hyperparameters are extracted directly from YAML configuration files:

### Tier 0 (configs/base.yaml)
- **Model Class**: `RandomForestRegressor`
- **Estimators (`n_estimators`)**: `200`
- **Feature Extractors**: `ElementProperty_magpie` + `CrystalNNFingerprint_SiteStatsFingerprint`
- **Nearest Neighbors (`top_k`)**: `10`
- **Bootstrap resampling runs**: `10,000`

### Tier 1 (configs/base.yaml)
- **Node Dimension (`node_dim`)**: `92` (representing 92 chemical elements)
- **Edge Dimension (`edge_dim`)**: `40`
- **Convolution Layers (`n_conv_layers`)**: `3`
- **Hidden Layer Dimension (`hidden_dim`)**: `64`
- **Dropout Rate (`dropout_rate`)**: `0.1` (base), `0.2` (optimized)
- **Learning Rate (`lr`)**: `0.001` with cosine annealing scheduler
- **Optimizer**: `AdamW` with `weight_decay = 1e-5`
- **GNN training epochs**: `200` (base) / `400` (optimized)
- **Early stopping patience**: `30` (base) / `60` (optimized)
- **Distance cutoff radius**: `8.0` Angstroms

---

## 5. Hardware Used

Not recorded in repository

---

## 6. Runtime Estimates

The runtime estimates are derived from execution shell scripts and Phase 6 logs:
- **First GNN run (compiling graph cache)**: `~2-4 hours`
- **Subsequent GNN runs (graphs cached)**: `~40 min CPU` / `~10 min GPU`
- **Tier 1 GNN Training Time (per configuration)**:
 - `band_gap` on `element_out`: `157.9` min
 - `band_gap` on `family_out`: `175.4` min
 - `band_gap` on `iid`: `161.4` min
 - `formation_energy` on `element_out`: `104.9` min
 - `formation_energy` on `family_out`: `116.5` min
- **Bootstrap CI Calculation**: `~1 min`
- **Physical Interpretability Script**: `<1 min`
- **Gating Analysis Sweep**: `20-40 min CPU` / `5-10 min GPU`
