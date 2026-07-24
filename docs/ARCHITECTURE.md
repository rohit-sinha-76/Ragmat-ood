# Repository Architecture Reference

This document provides a technical description of the data flow, package structure, configurations, and hyperparameters for the RAGMat-OOD project.

---

## 1. Pipeline Overview

The computational pipeline executes in the following sequence:
1. **Data Loading (`JARVISLoader`)**: Downloads the raw JARVIS-DFT dataset, parses atomic configurations into `pymatgen.Structure` objects, maps them to targets, and caches raw entries to JSON.
2. **Data Partitioning (`DataSplitter`)**: Splits data into IID, crystal-family-out, or element-out configurations. Checksums of splits are stored in `data/checksums.txt`.
3. **Graph Construction (`CrystalGraphBuilder`)**: Converts pymatgen structures to PyTorch Geometric `Data` objects containing node features (atomic numbers mapped to 92-dimensional one-hot arrays) and edge features (distance smearing over 40 Gaussian basis functions).
4. **Encoding (`CGCNNEncoder`)**: Extracts GNN embeddings from materials structures using message-passing convolution layers.
5. **Retrieval indexing (`FAISSIndex`)**: Collects training-set embeddings, applies L2 normalization, builds a FAISS IndexFlatIP (Inner Product) index for cosine similarity queries, and persists them to disk.
6. **Feature Fusion (`ConcatFusionHead` / `CrossAttentionFusionHead`)**: Queries the retrieval index for the `top_k` neighbors of an input material and merges neighbor features with query GNN representations.
7. **Adaptive Gating (`AdaptiveGate` / `MahalanobisDetector`)**: Inspects test-set embeddings using a Mahalanobis distance detector. If the query OOD score exceeds the threshold, the prediction falls back to a classical Random Forest model trained on Magpie composition and CrystalNN structural descriptors (`MatminerFeaturizer`).
8. **Uncertainty Calibration (`ConformalPredictor` / `MCDropoutUQ`)**: Calibrates prediction confidence intervals on validation datasets.

---

## 2. Data Flow Diagram

```
[Raw JARVIS-DFT Data]
 |
 v
 JARVISLoader.load() ---> (Structure, label, material_id)
 |
 v
 DataSplitter.split() ---> train_ids, val_ids, test_ids
 |
 +----------------------------+-----------------------------+
 | | |
 (Train Set) (Val Set) (Test Set)
 | | |
 v v v
CrystalGraphBuilder CrystalGraphBuilder CrystalGraphBuilder
 | | |
 [PyG Train Graphs] [PyG Val Graphs] [PyG Test Graphs]
 | | |
 v v v
CGCNNEncoder (Forward) CGCNNEncoder (Forward) CGCNNEncoder (Forward)
 | | |
 [Train Embeddings] [Val Embeddings] [Test Embeddings]
 | | |
 v | |
FAISSIndex.build() | |
 | | |
 [FAISS Index] <--------------------+-----------------------------+
 | (cosine similarity queries)
 v
FAISSIndex.query() ---> scores, retrieved_ids
 |
 v
FusionHead.forward() ---> Predicts property (Concatenation / Cross-Attention)
 |
 v
MahalanobisDetector.score() ---> OOD Scores
 |
 v
AdaptiveGate.batch_gate() ----> Gated routing (GNN vs. Tier-0 RF fallback)
 |
 v
[Final Evaluated Predictions & Metrics]
```

---

## 3. Module Reference

### `ragmat/config.py`
- `class ConfigIntegrityError(Exception)`: Raised when configuration parameters violate consistency rules.
- `class CGCNNConfig`: Dataclass containing parameters for the CGCNN encoder.
- `class Tier0Config`: Dataclass containing parameters for the Random Forest baseline.
- `class UQConfig`: Dataclass containing parameters for Uncertainty Quantification.
- `class WandbConfig`: Dataclass containing Weights & Biases credentials and project names.
- `class ExperimentConfig`: Handles experiment validation and hashing.
 - `from_yaml(cls, config_path: str | Path) -> ExperimentConfig`: Load config from YAML.
 - `_validate(self) -> None`: Validates property matching and model constraints.
 - `to_dict(self) -> dict`: Serialize config to dictionary.
 - `config_hash(self) -> str`: Unique MD5 hash of configurations.

### `ragmat/data/loader.py`
- `class JARVISLoader`: Downloads and caches the JARVIS dataset.
 - `__init__(self, raw_dir: str | Path, dataset_name: str = "dft_3d") -> None`
 - `load(self, target_property: str, max_samples: Optional[int] = None) -> list[tuple]`
 - `_load_or_download(self, cache_path: Path) -> list[dict]`
 - `_atoms_to_structure(atoms: dict) -> Structure`

### `ragmat/data/splitter.py`
- `class DataSplitter`: Handles deterministic dataset partitioning.
 - `__init__(self, splits_dir: str | Path, checksums_file: str | Path, seed: int = 42, val_fraction: float = 0.1, test_fraction: float = 0.2) -> None`
 - `split(self, material_ids: list[str], labels: list[float], structures, split_type: Literal['iid', 'family_out', 'element_out'], target_property: str) -> dict[str, list[str]]`
 - `_iid_split(self, ids: np.ndarray, labels: np.ndarray) -> dict[str, list[str]]`
 - `_family_out_split(self, ids: np.ndarray, labels: np.ndarray, structures) -> dict[str, list[str]]`
 - `_element_out_split(self, ids: np.ndarray, labels: np.ndarray, structures) -> dict[str, list[str]]`
 - `_assert_disjoint(split_dict: dict[str, list[str]], split_type: str) -> None`
 - `load_split(split_type: str, target_property: str, splits_dir: str | Path) -> dict[str, list[str]]`

### `ragmat/encoders/cgcnn.py`
- `class CGCNNLayer(MessagePassing)`: CGCNN graph convolution layer.
 - `__init__(self, hidden_dim: int, edge_dim: int) -> None`
 - `forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor`
 - `message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor`
- `class CGCNNEncoder(nn.Module)`: Deep CGCNN structural encoder.
 - `__init__(self, node_dim: int = 92, edge_dim: int = 40, hidden_dim: int = 64, n_conv_layers: int = 3, dropout_rate: float = 0.1) -> None`
 - `forward(self, data: Batch) -> tuple[Tensor, Tensor]`
 - `get_embedding(self, data: Batch) -> Tensor`

### `ragmat/encoders/graph_builder.py`
- `class CrystalGraphBuilder`: Translates crystal structures to PyG Graphs.
 - `__init__(self, cutoff_radius: float = 8.0, n_gaussian_basis: int = 40, gaussian_min: float = 0.0, gaussian_max: float = 8.0) -> None`
 - `structure_to_graph(self, structure: Structure, y: float, material_id: str) -> Data`
 - `build_dataset(self, structures: list[Structure], targets: list[float], ids: list[str]) -> list[Data]`
 - `_gaussian_smear(self, distances: torch.Tensor) -> torch.Tensor`

### `ragmat/explain.py`
- `class ExplainabilityModule`: Computes explainability scores.
 - `__init__(self, top_k: int) -> None`
 - `explain(self, query_features: np.ndarray, neighbor_ids: list[str], neighbor_features: np.ndarray, neighbor_labels: np.ndarray, query_id: str) -> dict`

### `ragmat/features/matminer_descriptors.py`
- `class MatminerFeaturizer`: Featurizes pymatgen structures with Magpie and CrystalNN descriptors.
 - `__init__(self, n_jobs: int = -1) -> None`
 - `featurize_dataset(self, structures: list[Structure], ids: list[str]) -> tuple[np.ndarray, list[str]]`
 - `fit_scaler(self, X_train: np.ndarray) -> StandardScaler`
 - `transform(X: np.ndarray, scaler: StandardScaler) -> np.ndarray`

### `ragmat/fusion/concat.py`
- `class ConcatFusionHead(nn.Module)`: Simple concatenation fusion layer.
 - `__init__(self, embedding_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.1) -> None`
 - `forward(self, query_embedding: Tensor, neighbor_embeddings: Tensor, neighbor_mask: Tensor | None = None) -> Tensor`

### `ragmat/fusion/cross_attention.py`
- `class CrossAttentionFusionHead(nn.Module)`: Multi-head cross-attention layer.
 - `__init__(self, embedding_dim: int, n_heads: int = 4, hidden_dim: int = 128, dropout_rate: float = 0.1) -> None`
 - `forward(self, query_embedding: Tensor, neighbor_embeddings: Tensor) -> Tensor`

### `ragmat/fusion/random_control.py`
- `class RandomRetrievalFusionHead(nn.Module)`: Baseline random retriever wrapper.
 - `__init__(self, base_fusion_head: nn.Module, train_embeddings_pool: np.ndarray, top_k: int) -> None`
 - `forward(self, query_embedding: Tensor, neighbor_embeddings: Tensor | None = None) -> Tensor`

### `ragmat/gating.py`
- `class AdaptiveGate`: Controls evaluation fallback.
 - `__init__(self, ood_threshold: float, coherence_threshold: float = 100.0) -> None`
 - `should_retrieve(self, ood_score: float, neighbor_property_variance: float) -> bool`
 - `batch_gate(self, ood_scores: np.ndarray, neighbor_variances: np.ndarray) -> np.ndarray`
 - `log_stats(self, gate_decisions: np.ndarray) -> None`

### `ragmat/ood/mahalanobis.py`
- `class MahalanobisDetector`: OOD detection based on Mahalanobis distance.
 - `__init__(self, threshold_percentile: float = 95.0) -> None`
 - `fit(self, train_embeddings: np.ndarray) -> None`
 - `normalized_threshold(self) -> float`
 - `score(self, embeddings: np.ndarray) -> np.ndarray`
 - `_compute_distances(self, embeddings: np.ndarray) -> np.ndarray`

### `ragmat/retrieval/concat_features.py`
- `def concat_retrieval_features(query_features: np.ndarray, index: FAISSIndex, train_features: np.ndarray, train_ids: list[str], top_k: int, aggregation: Literal['mean', 'concat_all']) -> np.ndarray`
- `def concat_random_retrieval_features(query_features: np.ndarray, train_features: np.ndarray, top_k: int, aggregation: Literal['mean', 'concat_all'], seed: int) -> np.ndarray`

### `ragmat/retrieval/faiss_index.py`
- `class FAISSIndex`: Exact cosine similarity retrieval using IndexFlatIP.
 - `__init__(self, dim: int, property_name: str, split_name: str) -> None`
 - `build(self, embeddings: np.ndarray, material_ids: list[str]) -> None`
 - `query(self, query_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, list[list[str]]]`
 - `save(self, path: str) -> None`
 - `load(self, path: str) -> None`
 - `index_name(tier: int, representation: str, property_name: str, split_name: str) -> str`

### `ragmat/retrieval/leakage_check.py`
- `class LeakageChecker`: Integrity assertion utility.
 - `assert_no_leakage(index_material_ids: Sequence[str], test_material_ids: Sequence[str], split_name: str, property_name: str) -> None`
 - `assert_split_disjoint(train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str], split_name: str) -> None`

### `ragmat/uncertainty/conformal.py`
- `class ConformalPredictor`: Calibrates uncertainty intervals for GNNs.
 - `__init__(self) -> None`
 - `calibrate(self, model: nn.Module, val_loader: DataLoader, coverage: float) -> None`
 - `predict_interval(self, predictions: Tensor) -> tuple[Tensor, Tensor]`
 - `half_width(self) -> float | None`
- `class SklearnConformalPredictor`: Conformal calibration for scikit-learn models.
 - `__init__(self) -> None`
 - `calibrate(self, y_true: np.ndarray, y_pred: np.ndarray, coverage: float) -> None`
 - `predict_interval(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
 - `half_width(self) -> float | None`

### `ragmat/uncertainty/mc_dropout.py`
- `class MCDropoutUQ`: Performs MC dropout.
 - `predict_with_uncertainty(model: nn.Module, forward_fn: callable, n_passes: int) -> tuple[Tensor, Tensor]`

---

## 4. Configuration Reference

The system configuration uses parameters defined in standard YAML config files:

| Config Key | Description | Found Types | Default / Examples |
|---|---|---|---|
| `experiment_name` | Unique string identifying the run. | `str` | `P4v2_BG_1`, `tier1_formation_energy_iid_optimized` |
| `tier` | System tier selection (0 = RF, 1 = CGCNN). | `int` | `0`, `1` |
| `target_property` | Material property to predict. | `str` | `formation_energy`, `band_gap` |
| `split_type` | Partition strategy. | `str` | `iid`, `family_out`, `element_out` |
| `representation` | Features mode. | `str` | `matminer`, `cgcnn` |
| `retrieval_mode` | Neighbor query mode. | `str` | `none`, `true_neighbor`, `random_control` |
| `fusion_method` | Head type for merging embeddings. | `str` | `concat`, `cross_attention` |
| `encoder_property` | Checkpoint target property consistency label. | `str` | `formation_energy`, `band_gap` |
| `retrieval_index_property` | Retrieval property consistency label. | `str` | `formation_energy`, `band_gap` |
| `top_k` | Number of nearest neighbors to retrieve. | `int` | `10` |
| `gating` | Enables OOD gating. | `bool` | `true`, `false` |
| `seed` | Random seed for data and weight initialization. | `int` | `42` |

### Tier 1 GNN Config Settings (`cgcnn`)
- `batch_size`: Batch size for graph loader (`int`, default: `32` / `128`)
- `n_epochs`: Total number of GNN epochs (`int`, default: `200` / `400`)
- `early_stopping_patience`: Epoch patience before termination (`int`, default: `30` / `60`)
- `hidden_dim`: Hidden layer embedding width (`int`, default: `64`)
- `cutoff_radius`: Crystal neighbor cutoff distance (`float`, default: `8.0`)
- `n_conv_layers`: Convolution layer count (`int`, default: `3`)
- `dropout_rate`: Dropout probability (`float`, default: `0.10` / `0.20`)
- `weight_decay`: Optimizer weight decay (`float`, default: `1e-5`)
- `lr`: Base learning rate (`float`, default: `0.001`)
- `lr_scheduler`: Learning rate schedule type (`str`, default: `cosine`)

---

## 5. Hyperparameters

Below are the exact hyperparameters defined in `configs/base.yaml`:
- **Cutoff radius (`cutoff_radius`)**: `8.0` Angstroms
- **Gaussian smeared basis (`n_gaussian_basis`)**: `40`
- **CGCNN Message-passing layers (`n_conv_layers`)**: `3`
- **Embedding size (`hidden_dim`)**: `64`
- **Dropout Rate (`dropout_rate`)**: `0.1` (base), `0.2` (optimized)
- **Base Learning Rate (`lr`)**: `0.001`
- **Weight Decay (`weight_decay`)**: `1e-5`
- **Conformal Coverage Target**: `0.9`
- **MC-Dropout forward passes**: `30`
- **Random Forest estimators**: `200`

---

## 6. Inference Pipeline

The system includes a dedicated inference module to query property predictions on custom crystal structures (e.g. CIF files) without reloading massive training reference files:

### Serialized Mahalanobis Detectors
Instead of reconstructing the 17.8 GB graph dataset at inference time, pre-fitted Mahalanobis OOD detectors are serialized:
* **Checkpoints:** `checkpoints/mahalanobis_detector_formation_energy.pkl` and `checkpoints/mahalanobis_detector_band_gap.pkl`
* **Performance:** Allows OOD scoring, Gating decisions, and property predictions in under **100 milliseconds** per crystal structure.

### Inference execution flow:
```
[CIF Crystal Input File]
 
 
[Pymatgen Structure Parser]
 
 [CrystalGraphBuilder] [GNN Input Graphs]
 
 
 [GNN Forward] [GNN Predict]
 
 [GNN Embeddings]
 
 
 [ZSNI Rescue] [OOD Mahalanobis]
 
 [MatminerFeaturizer] 
 
 [OOD Gating Decision]
 [RF Scaler] 
 
 Gated Predict
 [Random Forest Downstream] [RF Predict] (Fallback if OOD)
```
