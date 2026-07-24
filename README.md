# Compositional Distribution Shift in Crystal Graph Convolutional Neural Networks
> **Mechanistic Diagnosis, RAG Audit, Mahalanobis Gating, and Zero-Shot Imputation** 
> *Official repository for the paper: "Compositional Distribution Shift in Crystal Graph Convolutional Neural Networks: Mechanistic Diagnosis, RAG Audit, Mahalanobis Gating, and Zero-Shot Imputation"*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4.0-3776ab.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

### Overview

When machine learning models are deployed for **accelerated materials discovery**, they frequently encounter chemical compositions containing chemical elements that were never seen during model training. While standard benchmarks evaluate models under in-distribution (IID) splits, real-world deployment requires predicting properties for novel chemical spaces under **out-of-distribution (OOD) compositional shift**.

In this study, we investigate the root failure mechanism of **Crystal Graph Neural Networks (CGCNN)** under element exclusion across 93,902 JARVIS-DFT crystal structures. We discover that:

1. **The Weight-Level Failure Mode:** CGCNN's formation energy error increases **8.4-fold** under element exclusion because unvisited element columns in the first linear embedding layer receive zero gradients during training, retaining random initial values that inject noise into every downstream message-passing step.
2. **The RAG Audit:** Post-pooling retrieval augmentation (RAG) performs statistically indistinguishably from a capacity-matched random control vector, proving that late-stage fusion after global mean-pooling cannot recover structural detail already lost.
3. **Inference-Time Recovery:** We introduce two complementary recovery mechanisms operating **without retraining**:
 * **Mahalanobis Latent Gating:** Detects OOD failure states with $\text{AUROC} > 0.999$ and routes them to a Random Forest fallback.
 * **Zero-Shot Node Imputation (ZSNI):** Patches uninitialized embedding weight columns using periodic-table physical proximity (using tabulated 2D row/group coordinates), reducing Formation Energy error by **67.1%** and recovering split-conformal coverage from **18.5% to 58.6%**.

---

### Mechanistic Failure & Zero-Shot Recovery

#### 1. Why Element-Exclusion Collapse Occurs
Crystal GNNs like CGCNN convert atomic species into node features through a discrete linear lookup layer:

$$\mathbf{h}_i^{(0)} = \mathbf{W}_{\text{emb}} \, \text{one\_hot}(Z_i) + \mathbf{b}$$

When a specific element $Z_{\text{excluded}}$ (e.g., Selenium $\text{Se}$) is withheld during training:
* Column $\mathbf{W}_{\text{emb}}[:, Z_{\text{excluded}}]$ receives **zero gradient updates** ($\nabla_{\mathbf{W}_{\text{excluded}}} \mathcal{L} = \mathbf{0}$).
* The column retains its random initialization weights ($\mathcal{U}[-a, a]$).
* During test-time inference on a material containing $\text{Se}$, the random weights act as **pure noise vectors**, which propagate through all graph convolution layers and corrupt the final crystal embedding.

```text
Training Phase (14 Transition Metals + Se Withheld):
Element Index (Z): [ H, Li, Be, ... As, Se(Out), Br, ... ]
Gradient Update: [ dL/dW, dL/dW, ... 0.0000, dL/dW, ... ]
Embedding Matrix: [ Learned Weights ... | RANDOM NOISE | ... Learned Weights ]
 |
 v
 Corrupts Message Passing!
```

#### 2. Zero-Shot Node Imputation (ZSNI)
Instead of retraining the GNN model, **ZSNI** uses chemical periodic-table proximity (2D row and group coordinates) to reconstruct the uninitialized weight column prior to inference:

$$\hat{\mathbf{w}}_{Z_{\text{excluded}}} = \frac{1}{k_{\text{imp}}} \sum_{j \in \mathcal{N}_{k}(Z_{\text{excluded}})} \mathbf{w}_j$$

Where $\mathcal{N}_{k}$ represents the $k$-nearest seen chemical neighbors in the periodic table (e.g., averaging Arsenic $\text{As}$ and Bromine $\text{Br}$ embeddings to reconstruct Selenium $\text{Se}$).

---

### Dataset & Evaluation Splits

We benchmark all models across **93,902 crystal structures** from the **JARVIS-DFT** database. To systematically evaluate model behavior under increasing compositional and structural distribution shifts, we construct three distinct evaluation splits:

| Split Protocol | Split Logic & Description | Train Size | Val Size | Test Size | Key Evaluation Purpose |
|---|---|---|---|---|---|
| **IID Split** | Standard 80/10/10 random stratified split across the entire database. | 65,731 | 14,085 | 14,086 | Baseline predictive performance under in-distribution conditions. |
| **Family-Out Split** | Prototype family exclusion (withholding complete crystallographic space groups / structure prototypes). | 65,731 | 14,085 | 14,086 | Generalization across novel structural lattice prototypes. |
| **Element-Out Split** | **Strict Element Exclusion:** 15 withheld elements ($\text{Sc}, \text{Ti}, \text{V}, \text{Cr}, \text{Mn}, \text{Fe}, \text{Co}, \text{Ni}, \text{Cu}, \text{Zn}, \text{Y}, \text{Zr}, \text{Nb}, \text{Mo}$, and $\text{Se}$) assigned exclusively to the test set. | 59,391 | 6,600 | **27,911** | Hard compositional shift testing where test materials contain unseen elements. |

#### Pre-computed Split Indices
All split index JSON files are provided in `data/splits/`:
* `data/splits/split_iid_formation_energy.json`
* `data/splits/split_family_out_formation_energy.json`
* `data/splits/split_element_out_formation_energy.json`

---

### Installation & Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/rohit-sinha-76/Ragmat-ood.git
cd Ragmat-ood
```

#### 2. Option A: Conda Environment (Recommended)
```bash
conda env create -f environment.yml
conda activate ragmat
```

#### Option B: Docker Container
For 100% isolated containerized reproduction with GPU acceleration:
```bash
# Build and run using Docker Compose
docker-compose up --build

# Or directly with Docker
docker build -t ragmat-ood .
docker run --gpus all -it ragmat-ood bash
```

#### Option C: Standard Pip
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Verify System Integrity
Run the built-in PyTest test suite to verify environment integrity, graph builders, and model specs:
```bash
pytest tests/ -v
```

---

### Running Experiments & Reproduction Workflows

All training and evaluation pipelines are driven by master scripts in `scripts/` and configuration files in `configs/`:

* **`configs/base.yaml`**: Master configuration defining global defaults (hidden dimension = 64, cutoff radius = 8.0 , batch size = 512, weight decay = $1\times 10^{-4}$, Cosine Annealing LR schedule).
* **`configs/tier0_random_forest.yaml`**: Reference config for Random Forest baselines.
* **`configs/tier1_cgcnn.yaml`**: Reference config for Tier 1 CGCNN GNN encoders.

#### 1. Baseline Random Forest Training (Tier 0)
Train the Magpie descriptor Random Forest baseline:
```bash
python -m ragmat.train configs/tier0_random_forest.yaml
```

#### 2. CGCNN Base Encoder Training (Tier 1)
Train base CGCNN GNN models from scratch across all properties and evaluation splits:
```bash
# Train using clean Tier 1 config
python -m ragmat.train configs/tier1_cgcnn.yaml

# Or train all 6 base models via master script (2 properties 3 splits)
python scripts/run_phase6.py --stage 1 --prop all --split all
```

#### 3. RAG Retrieval Fusion & Random Controls
Train post-pooling fusion heads (`concat` and `cross_attention`) for both true-neighbor retrieval and matched random controls:
```bash
python scripts/run_phase6.py --stage 3 --prop all --split all --mode all
```

#### 4. Recovery Mechanisms & Uncertainty Quantification
Evaluate the two inference-time recovery strategies without retraining:

* **Mahalanobis OOD Gating & Fallback Routing:**
 ```bash
 python scripts/run_gating_analysis.py
 ```

* **Zero-Shot Node Imputation (ZSNI) & Split-Conformal UQ:**
 ```bash
 python scripts/run_conformal.py
 ```

#### 5. Generate Statistical Reports & Tables
Generate the exact Markdown summary reports corresponding to Tables 1–5 in the manuscript:
```bash
python scripts/run_bootstrap_cis.py # Generates final_result/bootstrap_cis_report.md
```

---

### Main Benchmark Results

Summary of model performance (Mean Absolute Error, MAE) across properties and evaluation splits (with 95% bootstrap confidence intervals):

| Target Property | Evaluation Split | Baseline Random Forest | Base CGCNN Encoder | RAG True-NN (Concat) | RAG Random Control | Recovery Mechanism |
|---|---|---|---|---|---|---|
| **Formation Energy** *(eV/atom)* | **IID** | 0.106 | **0.066** | 0.060 (0.059, 0.062) | 0.062 (0.060, 0.064) | — |
| | **Family-Out** | 0.237 | **0.133** | 0.140 (0.136, 0.144) | 0.142 (0.138, 0.146) | — |
| | **Element-Out** | 0.181 | **0.557** *(8.4 Error)* | 0.566 (0.556, 0.576) | 0.556 (0.546, 0.566) | **0.181** (Gated)<br> **0.183** (ZSNI, $k=2$) |
| **Band Gap** *(eV)* | **IID** | 0.226 | **0.177** | 0.173 (0.166, 0.180) | 0.172 (0.165, 0.179) | — |
| | **Element-Out** | 0.320 | **0.411** *(2.3× Error)* | 0.415 (0.405, 0.425) | 0.410 (0.400, 0.420) | **0.320** (Gated)<br> **0.322** (ZSNI, $k=2$) |

*All values reported in eV/atom (Formation Energy) or eV (Band Gap).*

#### Key Takeaways:
1. **Element Exclusion Collapse:** Under element exclusion, CGCNN formation energy MAE degrades from 0.066 to 0.557 eV/atom (8.4-fold error multiplier).
2. **RAG Equivalence:** RAG True-NN (0.566) and Random Control (0.556) show overlapping 95% bootstrap CIs under element exclusion, proving that late-stage fusion post-pooling provides no structural benefit over random capacity expansion.
3. **Recovery Success:** Mahalanobis Gating (0.181) and ZSNI (0.183) completely recover performance to the Random Forest baseline level without model retraining.

---

### Repository Structure

```text
ragmat-ood/
 configs/ # Master & reference configuration files
 base.yaml # Global hyperparameter defaults
 tier0_random_forest.yaml
 tier1_cgcnn.yaml
 data/ # Dataset split protocols & indices
 splits/ # Pre-computed split JSONs (IID, Family-Out, Element-Out)
 eval/ # Evaluation metrics & visualization utilities
 paper/ # Manuscript source (LaTeX, figures, bibliography)
 main.tex # Main paper source
 supplementary.tex # Supplementary information
 figures/ # Vector PDF and PNG figures
 ragmat/ # Core Python package
 data/ # JARVIS loader & dataset splitters
 detection/ # Mahalanobis OOD detector
 encoders/ # CGCNN GNN architecture & graph builder
 fusion/ # RAG retrieval fusion heads & random controls
 imputation/ # Zero-Shot Node Imputation (ZSNI)
 retrieval/ # FAISS indexer & leakage audit
 uncertainty/ # Split-conformal prediction & coverage
 final_result/ # Final publication Markdown reports & JSON metrics
 scripts/ # Production pipeline & reproduction scripts
 run_phase6.py # Master GNN training pipeline
 run_gating_analysis.py# Mahalanobis gating & fallback evaluation
 run_conformal.py # ZSNI recovery & conformal UQ evaluation
 run_bootstrap_cis.py # Statistical CI generator
 tests/ # PyTest unit testing suite
 CITATION.cff # Citation metadata
 Dockerfile # Docker reproduction setup
 LICENSE # MIT License
 METHODOLOGY.md # Comprehensive mathematical & benchmark formulation
 README.md # Project landing page
 REPRODUCE.md # Reproduction instructions
```

---

### Citation

If you use this codebase or research in your work, please cite the manuscript:

```bibtex
@unpublished{sinha2026compositional,
  title={Compositional Distribution Shift in Crystal Graph Convolutional Neural Networks: Mechanistic Diagnosis, RAG Audit, Mahalanobis Gating, and Zero-Shot Imputation},
  author={Sinha, Rohit},
  note={Preprint / Manuscript under review},
  year={2026},
  url={https://github.com/rohit-sinha-76/Ragmat-ood}
}
```

---

### License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

