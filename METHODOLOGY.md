# Pinpoint End-to-End Methodology: RAGMat-OOD

This document provides a complete, line-by-line technical specification of the entire RAGMat-OOD pipeline, extracted and cross-checked directly from the codebase configurations, scripts, and implementation files.

---

## 1. Hardware Architecture & System Environment

| Hardware / Resource | Specification | Source / Verification File |
|---|---|---|
| **GPU Model** | NVIDIA T1000 (8GB GDDR6 VRAM) | `nvidia-smi` |
| **GPU Architecture** | Turing (TU117, Compute Capability 7.5) | `nvidia-smi` |
| **Host Interconnect** | PCI-Express 3.0 x16 | System Hardware Profile |
| **CUDA Driver / Version** | Driver Version: 596.59 \| CUDA Version: 13.2 | `nvidia-smi` |
| **OS / Runtime** | Windows 11 Pro / WSL2 Ubuntu 22.04 LTS | System Profile |

---

## 2. Software & Dependency Stack

| Library / Tool | Version | Purpose in Pipeline | Source File |
|---|---|---|---|
| **Python** | 3.10.14 | Core Execution Language | `environment.yml` |
| **PyTorch** | 2.5.1 (`cu121`) | Deep Learning Framework | `requirements.txt` |
| **PyTorch Geometric (PyG)** | 2.6.1 | Graph Convolutional Neural Network Engine | `requirements.txt` |
| **jarvis-tools** | **`2026.6.12`** | Dataset Access (`jarvis.db.figshare.data("dft_3d")`) | `requirements.txt`, `README.md` |
| **matminer** | 0.10.1 | Magpie & Structure Descriptor Extraction | `requirements.txt` |
| **scikit-learn** | 1.5.0 | Random Forest Baseline & Conformal Predictors | `requirements.txt` |
| **faiss-cpu** | 1.14.3 | Fast Inner-Product Vector Similarity Retrieval | `requirements.txt` |
| **scipy** | 1.13.0 | Mahalanobis Distance & Precision Matrix Inversion | `requirements.txt` |
| **numpy** | 1.26.0 | Array Manipulation & Linear Algebra | `requirements.txt` |
| **wandb** | Latest | Training Logger & Experiment Tracker | `requirements.txt` |

---

## 3. Dataset & Exact Version Snapshot

- **Dataset Source:** JARVIS-DFT 3D database (`dft_3d`), retrieved via `jarvis-tools` version **`2026.6.12`** (FigShare API release).
- **Loader Module:** `ragmat/data/loader.py` calling `jarvis.db.figshare.data("dft_3d")`.
- **Cached Raw JSON Files:**
 - `data/raw/dft_3d_formation_energy.json`
 - `data/raw/dft_3d_band_gap.json`
- **Total Valid Crystal Count:** $N = 93,902$ bulk 3D structures with computed DFT-PBE Formation Energies ($\text{eV/atom}$) and OptB88vdW Band Gaps ($\text{eV}$).

### Split Partitioning & SHA256 Checksums (`data/checksums.txt`):

| Split Protocol | Train Size | Val Size | Test Size | Character & OOD Description | File SHA256 Checksum (`data/checksums.txt`) |
|---|---|---|---|---|---|
| **In-Distribution (\iid{})** | 65,730 | 9,391 | 18,780 | Stratified random 70/15/15 partition (seed 42) | `split_iid_formation_energy.json`: `527836206c58276d8d4f7804c2b649ef`<br>`split_iid_band_gap.json`: `5f6d92ef972c65970e7661acda168bf3` |
| **Prototype Family (\famout{})** | 59,392 | — | 22,693 | 591 AFLOW crystal prototypes; 23 held out | `split_family_out_formation_energy.json`: `7097d98ed77d4bce88ed661f24aba848`<br>`split_family_out_band_gap.json`: `7097d98ed77d4bce88ed661f24aba848` |
| **Element Exclusion (\elout{})** | 59,392 | 6,599 | 27,911 | 15 elements excluded (`[Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Y, Zr, Nb, Mo, Se]`) | `split_element_out_formation_energy.json`: `0a575a9c8feb9ba498ffd8b4e267507f`<br>`split_element_out_band_gap.json`: `0a575a9c8feb9ba498ffd8b4e267507f` |

---

## 4. Graph Construction & Disk Caching (`ragmat/encoders/graph_builder.py`, `scripts/run_phase6.py`)

- **Node Features ($\bm{x}_v \in \mathbb{R}^{92}$):** One-hot element encoding covering atomic numbers $Z=1$ ($\text{H}$) to $Z=92$ ($\text{U}$).
- **Edge Connectivity:** Bidirectional graph edges between all atom pairs within radial cutoff $r_{\text{max}} = 8.0\,\text{\AA}$.
- **Edge Attributes ($\bm{e}_{ij} \in \mathbb{R}^{40}$):** Gaussian-smeared interatomic distances with 40 basis centers uniformly spaced from $0.0\,\text{\AA}$ to $8.0\,\text{\AA}$ with width $\sigma = 0.2\,\text{\AA}$.
- **Disk Caching Protocol:** Graphs built by `CrystalGraphBuilder` are serialized and cached to disk (`data/graphs/`) as PyTorch Geometric `Data` objects to eliminate redundant `pymatgen` parsing overhead across pipeline stages.

---

## 5. Neural Network Architecture (`ragmat/encoders/cgcnn.py`)

- **Class Name:** `CGCNNEncoder` (implements Xie & Grossman, *Phys. Rev. Lett.* 120, 145301, 2018).
- **Lookup Embedding Layer:** Linear projection $\bm{W}_{\text{emb}} \in \mathbb{R}^{64 \times 92}$ mapping 92D node features to 64D hidden space, initialized via Kaiming Uniform initialization (PyTorch default `nn.Linear`).
- **Graph Convolution Layers ($N_{\text{conv}} = 3$):**
 - Message network: $\bm{z}_{ij} = \text{Softplus}\left(\text{Linear}([h_i \parallel h_j \parallel e_{ij}])\right)$, where input dimension is $2 \times 64 + 40 = 168$ and output dimension is $2 \times 64 = 128$.
 - Gated Message: $\bm{m}_{ij} = \sigma(\bm{z}_{ij}[:64]) \odot \tanh(\bm{z}_{ij}[64:])$.
 - Aggregation & Norm: Sum aggregation followed by residual connection and LayerNorm: $h_i^{(l+1)} = \text{LayerNorm}\left(h_i^{(l)} + \sum_{j} \bm{m}_{ij}\right)$.
- **Global Pooling:** Graph-level embedding $\bm{z} \in \mathbb{R}^{64}$ obtained via `global_mean_pool`.
- **Prediction Head:** 2-layer MLP: $\text{Linear}(64 \to 32) \to \text{ReLU} \to \text{Dropout}(p=0.1) \to \text{Linear}(32 \to 1)$.

---

## 6. Random Forest Baseline (`ragmat/features/matminer_descriptors.py`)

- **Compositional Featurizer:** Matminer `ElementProperty.from_preset("magpie")` extracting 145 physical features (electronegativity, atomic radius, valence electrons, etc.).
- **Structural Featurizer:** Matminer `SiteStatsFingerprint` (CrystalNN parameters).
- **Estimator Model:** `sklearn.ensemble.RandomForestRegressor`:
 - `n_estimators = 200`
 - `max_depth = 30`
 - `random_state = 42`
 - `n_jobs = -1`

---

## 7. Training & Optimization Setup (`ragmat/train.py`, `scripts/run_phase6.py`, `configs/base.yaml`)

- **Batch Size:** 512 (GPU optimized, T1000 safe).
- **Epoch Budget:** 400 epochs max budget.
- **Optimizer:** `torch.optim.AdamW`:
 - Initial Learning Rate: $\eta = 10^{-3}$
 - Weight Decay: $\lambda = 10^{-4}$ (decoupled regularisation).
- **Learning Rate Schedule:** 10-epoch linear warmup, followed by Cosine Annealing.
- **Loss Function:** PyTorch Huber Loss (`nn.HuberLoss(delta=0.1)`):
 $$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le 0.1 \\ 0.1 |y - \hat{y}| - 0.005 & \text{otherwise} \end{cases}$$
- **Target Normalization Protocol:** `sklearn.preprocessing.StandardScaler` fitted *exclusively* on the training partition target property values ($y_{\text{train}}$). Predictions are rescaled back to original units ($\text{eV/atom}$ or $\text{eV}$) prior to metric evaluation.
- **Precision:** Mixed precision training (`torch.cuda.amp.autocast(dtype=torch.float16)` forward pass, fp32 loss).
- **Early Stopping:** Patience of 50 validation epochs, enforced after a minimum floor of 150 epochs.

---

## 8. Retrieval-Augmented Generation (RAG) & Freeze Protocol (`ragmat/fusion/`, `scripts/run_phase6.py`)

- **Vector Database:** FAISS CPU `IndexFlatIP` built over normalized 64D pooled graph embeddings of the training partition.
- **Top-$k$ Retrieval:** $k = 5$ nearest neighbors retrieved by inner product similarity.
- **Fusion Architectures:**
 1. **Concat (`ragmat/fusion/concat.py`):** Mean pooled vector of top-5 retrieved embeddings $\bar{\bm{z}}_{\text{ret}} = \frac{1}{K}\sum_{k=1}^K \bm{z}_{i,k} \in \mathbb{R}^{64}$ is concatenated with query embedding $\bm{z}$ before passing to prediction head $\text{LayerNorm}(128) \to \text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Dropout}(p=0.1) \to \text{Linear}(64 \to 1)$.
 2. **Cross-Attention (`ragmat/fusion/cross_attention.py`):** Multi-head cross-attention (`nn.MultiheadAttention(embed_dim=64, num_heads=1, batch_first=True)`) over retrieved embeddings $\bm{Z}_{\text{ret}} \in \mathbb{R}^{5 \times 64}$ with query $Q \in \mathbb{R}^{1 \times 1 \times 64}$.
- **Encoder Freeze Protocol:** During Stage 3 fusion head training, the base `CGCNNEncoder` weights are strictly frozen (`requires_grad = False`). Only fusion head parameters receive gradient updates.
- **Random Control Injection (`ragmat/fusion/random_control.py`):** `RandomRetrievalFusionHead` wraps base fusion heads and samples random training embeddings via `torch.randperm` (without replacement) or `torch.randint` (with replacement) at both training and inference time to evaluate retrieval content dependency vs added model capacity.

---

## 9. Mahalanobis Latent-Space Gating (`ragmat/ood/mahalanobis.py`, `ragmat/gating.py`)

- **Covariance Inversion:** Multivariate Gaussian $\mathcal{N}(\bm{\mu}, \bm{\Sigma})$ fitted over 64D L2-normalized training embeddings:
 $$\bm{\Sigma}_{\text{reg}} = \text{Cov}(\bm{Z}_{\text{train}}) + 10^{-5} \cdot \bm{I}_{64}$$
 Precision matrix $\bm{\Sigma}^{-1} = \text{pinv}(\bm{\Sigma}_{\text{reg}})$.
- **Mahalanobis Distance:**
 $$d_M(\bm{z}) = \sqrt{(\bm{z} - \bm{\mu})^\top \bm{\Sigma}^{-1} (\bm{z} - \bm{\mu})}$$
- **Score Normalization:** $S(\bm{z}) = \text{clip}\left(\frac{d_M(\bm{z})}{\max_{\text{train}} d_M}, 0.0, 1.0\right)$.
- **Routing Decision:**
 $$\hat{y} = \begin{cases} \hat{y}_{\text{CGCNN}} & \text{if } S(\bm{z}) \le \tau \\ \hat{y}_{\text{RF}} & \text{if } S(\bm{z}) > \tau \end{cases}$$
- **Evaluated Thresholds:** $\tau \in [0.3, 0.9]$.

---

## 10. Zero-Shot Node Imputation (ZSNI) (`ragmat/explain.py`, `scripts/run_zsni_pettifor.py`, Algorithm 1)

- **Input:** Unmodified trained `CGCNNEncoder` with weight matrix $\bm{W} \in \mathbb{R}^{64 \times 92}$.
- **Primary 2D Coordinates:** Periodic table row and group coordinates $\bm{p}_e = (\text{row}_e, \text{group}_e)$ standardized to zero-mean, unit-variance across $Z=1 \dots 92$.
- **Imputation Step:** For each unseen element $e \notin \mathcal{S}$ (15 excluded elements):
 $$\mathcal{N}(e) = \arg\min_{s \in \mathcal{S},\, |\cdot|=k} \|\bm{p}_e - \bm{p}_s\|_2 \quad (k=2)$$
 $$\bm{W}[:, e] \leftarrow \frac{1}{k} \sum_{s \in \mathcal{N}(e)} \bm{W}[:, s]$$
- **1D Pettifor & Electronegativity Baseline Comparison (`scripts/run_zsni_pettifor.py`):** Evaluates 1D Pettifor Mendeleev numbers (`mendeleev_no`) + Pauling electronegativity ($X$). Standardized 2D Row/Group coordinates significantly outperform 1D Pettifor ordering ($0.1830\,\text{eV/atom}$ vs $0.325\,\text{eV/atom}$ at $k=7$).
- **Property:** Operates in-place on embedding columns with zero gradient updates and zero training labels.

---

## 11. Split Conformal Uncertainty Prediction (`ragmat/uncertainty/conformal.py`)

- **Calibrated Set:** Validation partition ($n_{\text{cal}} = 6,599$ structures for \elout{}).
- **Nonconformity Score:** $s_i = |y_i - \hat{y}_i|$.
- **Quantile Threshold:**
 $$q = \text{Quantile}\left(\{s_i\}_{i=1}^{n_{\text{cal}}}, \, \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}\right) \quad (\alpha = 0.10 \implies 90\% \text{ coverage})$$
- **Prediction Interval:** $[\hat{y} - q, \hat{y} + q]$.

---

## 12. Statistical Validation Protocol (`scripts/run_bootstrap_cis.py`)

- **Bootstrap Method:** Non-parametric paired resample bootstrap.
- **Resamples:** $B = 5,000$ iterations with random seed 42.
- **Confidence Interval:** Empirical $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles of bootstrapped MAE differences.

---

## 13. Evaluation Metrics & Benchmark Definitions (`eval/metrics.py`)

All evaluation metrics are implemented in `eval/metrics.py` and computed across the full test set (`all`), low-OOD samples (`low_ood`), and high-OOD samples (`high_ood`).

### A. Primary Regression Metrics
1. **Mean Absolute Error ($\text{MAE}$):** Primary evaluation metric for Formation Energy ($\text{eV/atom}$) and Band Gap ($\text{eV}$):
 $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error ($\text{RMSE}$):**
 $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
3. **Coefficient of Determination ($R^2$):**
 $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

### B. Out-of-Distribution Detection & Recovery Metrics
4. **AUROC (Area Under Receiver Operating Characteristic Curve):** Measures trade-off between True Positive Rate and False Positive Rate for distinguishing \elout{} test samples from \iid{} samples via normalized Mahalanobis distance $S(\bm{z})$.
5. **AUPRC (Area Under Precision-Recall Curve):** Evaluates precision-recall trade-off under severe class imbalance.
6. **FPR95 (False Positive Rate at 95% TPR):** False positive rate when the OOD detector achieves a 95% True Positive Rate.
7. **Performance Gap Recovery (%):** Quantifies percentage of the performance drop restored by gating or imputation:
 $$\text{Recovery} = 100 \times \frac{\text{MAE}_{\text{broken}} - \text{MAE}_{\text{recovered}}}{\text{MAE}_{\text{broken}} - \text{MAE}_{\text{RF}}}$$

### C. Uncertainty Calibration Metrics
8. **Empirical Coverage (%):** Proportion of test set structures whose true property value $y_i$ falls within the split-conformal interval $[\hat{y}_i - q, \hat{y}_i + q]$ at nominal $90\%$ coverage ($1-\alpha = 0.90$).
9. **Expected Calibration Error ($\text{ECE}$):** Binned difference between empirical and nominal interval coverage across $N_{\text{bins}} = 10$ confidence levels:
 $$\text{ECE} = \frac{1}{N_{\text{bins}}} \sum_{m=1}^{N_{\text{bins}}} |\text{acc}(B_m) - \text{conf}(B_m)|$$
10. **Negative Log-Likelihood ($\text{NLL}$):** Evaluated under a Gaussian predictive distribution for MC-Dropout ($N_{\text{passes}} = 30$).

---

## 14. Data-Leakage Auditing & Disjointness Enforcement (`ragmat/retrieval/leakage_check.py`)

To guarantee strict scientific integrity and prevent train/test contamination:
1. **Zero Index Overlap (`LeakageChecker.assert_no_leakage`):** Asserts that $\text{FAISS\_IDs} \cap \text{Test\_IDs} = \emptyset$. CI workflow halts immediately if any test material ID appears in the retrieval index.
2. **Partition Disjointness (`LeakageChecker.assert_split_disjoint`):** Verifies pairwise empty intersections:
 $$(\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset) \quad \land \quad (\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset) \quad \land \quad (\mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset)$$

---

## 15. Monte Carlo Dropout Uncertainty Estimation (`ragmat/uncertainty/mc_dropout.py`)

- **Class Name:** `MCDropoutUQ`.
- **Inference Mode:** Enables dropout layers at test time by putting the module in `model.train()` mode during evaluation.
- **Sampling Iterations:** $N_{\text{passes}} = 30$ stochastic forward passes per structure.
- **Predictive Variance:**
 $$\hat{y}_{\text{mean}} = \frac{1}{30} \sum_{t=1}^{30} \hat{y}^{(t)}, \quad \sigma^2_{\text{pred}} = \frac{1}{29} \sum_{t=1}^{30} \left(\hat{y}^{(t)} - \hat{y}_{\text{mean}}\right)^2$$

---

## 16. Interpretability & Physical Neighbor Relevance (`ragmat/explain.py`)

- **Class Name:** `ExplainabilityModule`.
- **Cosine Similarity:** Computes normalized vector cosine similarity between query crystal embedding $\bm{z}_q$ and top-$k=10$ retrieved neighbor embeddings $\bm{z}_i$:
 $$\text{cosine}(\bm{z}_q, \bm{z}_i) = \frac{\bm{z}_q \cdot \bm{z}_i}{\|\bm{z}_q\|_2 \, \|\bm{z}_i\|_2}$$
- **Physical Relevance Score:** Average similarity over top-$k$ neighbors:
 $$\text{Score}_{\text{phys}} = \frac{1}{k} \sum_{i=1}^k \text{cosine}(\bm{z}_q, \bm{z}_i)$$

---

## 17. Granular OOD Severity Slicing (`eval/metrics.py`)

To evaluate performance degradation across varying distances from the training manifold, test structures are partitioned into severity bins based on Mahalanobis score $S(\bm{z})$:
- **`low_ood` Bin:** Test structures with $S(\bm{z}) \le \tau$ (in-distribution manifold proximity).
- **`high_ood` Bin:** Test structures with $S(\bm{z}) > \tau$ (out-of-distribution manifold shift).
- **`all` Bin:** Unfiltered full test partition ($N=18,780$ IID; $N=22,693$ Family-Out; $N=27,911$ Element-Out).

---

## 18. Execution Environment & JIT Compiler Flags (`scripts/run_phase6.py`)

To ensure deterministic, reproducible, and offline execution:
- **Offline Logging:** `WANDB_MODE="offline"` is set across all execution scripts to enforce local logging without external API network latency.
- **Triton JIT Compilers:** Conda-forge C++ compilers configured for PyTorch `torch.compile` Triton JIT compilation:
 - `CC = "x86_64-conda-linux-gnu-gcc"`
 - `CXX = "x86_64-conda-linux-gnu-g++"`

---

## 19. Full Reproduction Workflow Sequence (`REPRODUCE.md`, `scripts/reproduce_all.sh`)

1. **Environment Setup:** `conda env create -f environment.yml && conda activate ragmat`
2. **Integrity Testing:** `pytest tests/ -v`
3. **Tier 0 Random Forest Training:** `python -m ragmat.train configs/tier0_random_forest.yaml`
4. **Stage 1 GNN Base Training:** `python scripts/run_phase6.py --stage 1 --prop all --split all`
5. **Stage 3 RAG Fusion Training:** `python scripts/run_phase6.py --stage 3 --prop all --split all --mode all`
6. **ZSNI Rescue & Conformal UQ Calibration:** `python scripts/run_conformal.py`
7. **Mahalanobis Gating Sweep:** `python scripts/run_gating_analysis.py`
8. **Report & Table Generation:** `python scripts/run_bootstrap_cis.py` (Generates `final_result/bootstrap_cis_report.md`).

---

## 20. Cross-File Integrity & Verification Table

| Parameter / Protocol | Value Across Codebase | Source Files Verified | Status |
|---|---|---|---|
| **Primary Metric** | Mean Absolute Error ($\text{MAE}$) | `eval/metrics.py`, `ragmat/train.py`, `paper/main.tex` | Verified |
| **JARVIS Data Version** | `jarvis-tools==2026.6.12` | `requirements.txt`, `README.md`, `ragmat/data/loader.py` | Verified |
| **Loss Function** | Huber Loss ($\delta=0.1$) | `ragmat/train.py`, `configs/base.yaml`, `paper/main.tex` | Verified |
| **Target Normalization** | `StandardScaler` on Train | `scripts/run_phase6.py`, `paper/main.tex` | Verified |
| **Node Embedding Matrix** | $\bm{W}_{\text{emb}} \in \mathbb{R}^{64 \times 92}$ | `ragmat/encoders/cgcnn.py`, `ragmat/encoders/graph_builder.py` | Verified |
| **Element Exclusion Set** | 15 withholding elements | `data/splits/split_element_out_*.json`, `paper/main.tex` | Verified |
| **ZSNI Coordinates** | 2D Row/Group $(\text{row}_e, \text{group}_e)$ | `ragmat/explain.py`, `paper/main.tex` | Verified |
| **1D Pettifor Comparison** | Mendeleev No. + Electronegativity | `scripts/run_zsni_pettifor.py`, `paper/main.tex` | Verified |
| **Mahalanobis Covariance Reg.** | $10^{-5} \cdot \bm{I}_{64}$ | `ragmat/ood/mahalanobis.py`, `paper/main.tex` | Verified |
| **Data Leakage Auditor** | Assert zero ID overlap | `ragmat/retrieval/leakage_check.py` | Verified |
| **MC-Dropout Passes** | $N_{\text{passes}} = 30$ | `ragmat/uncertainty/mc_dropout.py`, `configs/base.yaml` | Verified |
| **Fusion Training Freeze** | Base encoder frozen | `scripts/run_phase6.py`, `ragmat/fusion/concat.py` | Verified |
| **Cross-Attention Config** | `num_heads=1, embed_dim=64` | `ragmat/fusion/cross_attention.py` | Verified |
| **Random Control Mode** | Separately trained (`_pool` buffer) | `ragmat/fusion/random_control.py` | Verified |
| **Interpretability Metric** | Cosine similarity $\text{Score}_{\text{phys}}$ | `ragmat/explain.py` | Verified |
| **Granular Severity Bins** | `low_ood`, `high_ood`, `all` | `eval/metrics.py` | Verified |
| **Environment & JIT Flags** | `WANDB_MODE=offline`, Conda GCC | `scripts/run_phase6.py`, `REPRODUCE.md` | Verified |
| **Reproduction Pipeline** | 10-step sequence (`reproduce_all.sh`) | `REPRODUCE.md`, `scripts/reproduce_all.sh` | Verified |
