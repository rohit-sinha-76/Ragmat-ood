# Compositional Distribution Shift in Crystal Graph Convolutional Neural Networks
> **Mechanistic Diagnosis, RAG Audit, Mahalanobis Gating, and Zero-Shot Imputation**  
> *Official repository for the research manuscript under peer review.*

[![CI](https://github.com/rohit-sinha-76/Ragmat-ood/actions/workflows/ci.yml/badge.svg)](https://github.com/rohit-sinha-76/Ragmat-ood/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG 2.6](https://img.shields.io/badge/PyG-2.6.1-3776AB.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Preprint](https://img.shields.io/badge/Status-Preprint%20Under%20Review-blue.svg)](#citation)

---

## Executive Summary

When machine learning models are deployed for **accelerated materials discovery**, they frequently encounter chemical compositions containing chemical elements that were never seen during model training. While standard benchmarks evaluate models under in-distribution (IID) splits, real-world deployment requires predicting properties for novel chemical spaces under **out-of-distribution (OOD) compositional shift**.

In this study, we investigate the root failure mechanism of **Crystal Graph Neural Networks (CGCNN)** under element exclusion across **93,902 JARVIS-DFT crystal structures**.

```
                           ┌────────────────────────────────────────────────────────┐
                           │               Compositional Shift Pipeline             │
                           └──────────────────────────┬─────────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         ┌──────────────────────────┐                                  ┌──────────────────────────┐
         │ Weight-Level Failure     │                                  │  Inference-Time Recovery │
         ├──────────────────────────┤                                  ├──────────────────────────┤
         │ • Zero Gradient Updates  │                                  │ • Mahalanobis Gating     │
         │ • Random Weight Columns  │                                  │   (AUROC > 0.999)        │
         │ • 8.4x Error Degredation │                                  │ • Zero-Shot Imputation   │
         └──────────────────────────┘                                  │   (67.1% Error Reduction)│
                                                                       └──────────────────────────┘
```

### Key Scientific Discoveries:
1. **The Weight-Level Failure Mode:** CGCNN's formation energy error increases **8.4-fold** under element exclusion (MAE degrades from 0.066 to 0.557 eV/atom). Unvisited element columns in the first linear embedding matrix **W**<sub>emb</sub> ∈ ℝ<sup>64×92</sup> receive zero gradients during training (∇<sub>**W**<sub>excluded</sub></sub> L = **0**), retaining random initial values that inject isotropic noise into every downstream message-passing layer.
2. **The RAG Audit:** Post-pooling retrieval augmentation (RAG) performs statistically indistinguishably from a capacity-matched random control vector (*p* = 0.568), proving that late-stage fusion after global mean-pooling cannot recover structural details already destroyed by uninitialized node features.
3. **Inference-Time Recovery (Without Retraining):**
   - **Mahalanobis Latent Space Gating:** Detects OOD failure states with AUROC > 0.999 and routes them to a Random Forest fallback, capping error at 0.181 eV/atom.
   - **Zero-Shot Node Imputation (ZSNI):** Reconstructs uninitialized embedding weight columns using periodic-table 2D coordinates, reducing Formation Energy error by **67.1%** (0.183 eV/atom) and recovering split-conformal coverage from **18.5% to 58.6%**.

---

## Mathematical Formulation

### 1. The Weight-Level Failure Mechanism
Crystal GNNs convert atomic species *Z*<sub>*i*</sub> into initial node representations **h**<sub>*i*</sub><sup>(0)</sup> via a linear lookup layer:

```text
h_i^{(0)} = W_emb · one_hot(Z_i) + b
```

When an element *Z*<sub>excluded</sub> (e.g., Selenium, Se) is withheld during training, the partial derivative is identically zero:

```text
∂L / ∂W_emb[:, Z_excluded] = 0
```

Consequently, **W**<sub>emb</sub>[:, *Z*<sub>excluded</sub>] retains its random initialization weights (~U[-a, a]), acting as an uncalibrated noise vector that corrupts all subsequent graph message-passing convolutions.

### 2. Zero-Shot Node Imputation (ZSNI)
Prior to inference, **ZSNI** imputes the uninitialized column by distance-weighted averaging of the *k*-nearest seen elements in 2D periodic table space (row<sub>j</sub>, group<sub>j</sub>):

```text
ŵ_{Z_excluded} = (1 / k_imp) * ∑_{j ∈ N_k(Z_excluded)} w_j
```

Where *k*<sub>imp</sub> = 2 optimal chemical neighbors (e.g., averaging Arsenic As and Bromine Br embeddings to reconstruct Selenium Se).


---

## Benchmark Results

Performance across 93,902 JARVIS-DFT crystals (with 95% bootstrap confidence intervals):

| Target Property | Evaluation Split | Baseline Random Forest | Base CGCNN Encoder | RAG True-NN (Concat) | RAG Random Control | Recovery Mechanism |
|---|---|---|---|---|---|---|
| **Formation Energy** *(eV/atom)* | **IID Split** | 0.106 | **0.066** | 0.060 (0.059, 0.062) | 0.062 (0.060, 0.064) | — |
| | **Family-Out** | 0.237 | **0.133** | 0.140 (0.136, 0.144) | 0.142 (0.138, 0.146) | — |
| | **Element-Out** | 0.181 | **0.557** *(8.4× Error)* | 0.566 (0.556, 0.576) | 0.556 (0.546, 0.566) | **0.181** (Gated)<br> **0.183** (ZSNI, *k*=2) |
| **Band Gap** *(eV)* | **IID Split** | 0.226 | **0.177** | 0.173 (0.166, 0.180) | 0.172 (0.165, 0.179) | — |
| | **Family-Out** | 0.253 | **0.174** | 0.170 (0.163, 0.177) | 0.171 (0.164, 0.178) | — |
| | **Element-Out** | 0.320 | **0.411** *(2.3× Error)* | 0.415 (0.405, 0.425) | 0.410 (0.400, 0.420) | **0.320** (Gated)<br> **0.322** (ZSNI, *k*=2) |

---

## Quick Start & Reproduction

### 1. Clone & Environment Setup
```bash
# Clone repository
git clone https://github.com/rohit-sinha-76/Ragmat-ood.git
cd Ragmat-ood

# Create Conda environment
conda env create -f environment.yml
conda activate ragmat
```

### 2. Fast Verification (35 Unit Tests)
```bash
pytest tests/ -v
```

### 3. Production Reproduction Pipeline
```bash
# 1. Run Random Forest Baselines
python -m ragmat.train configs/tier0_random_forest.yaml

# 2. Train Base CGCNN Encoders
python -m ragmat.train configs/tier1_cgcnn.yaml

# 3. Evaluate Mahalanobis OOD Gating
python scripts/run_gating_analysis.py

# 4. Evaluate Zero-Shot Node Imputation & Conformal UQ
python scripts/run_conformal.py

#### 5. Generate Statistical Reports & Tables
Generate the exact Markdown summary reports corresponding to Tables 1–5 in the manuscript:
```bash
python scripts/run_bootstrap_cis.py # Generates final_result/bootstrap_cis_report.md
```

---

### Paper-to-Code Mapping Matrix

For complete transparency, this matrix maps every table and figure in the manuscript to its generating script and metric dump:

| Manuscript Item | Description | Generating Script / Tool | Saved Output Location |
|---|---|---|---|
| **Table 1** | Primary MAE Benchmark across IID, Family-Out, & Element-Out | `python scripts/run_phase6.py` | `final_result/*.json` |
| **Table 2 & Fig 2** | Mahalanobis OOD Gating AUROC & Error Routing | `python scripts/run_gating_analysis.py` | `final_result/gating_report.md` |
| **Table 3 & Fig 3** | Zero-Shot Node Imputation (ZSNI) *k*-ablation | `python scripts/run_conformal.py` | `final_result/conformal_report.md` |
| **Table 4** | Split-Conformal Prediction Coverage & Interval Widths | `python scripts/run_conformal.py` | `final_result/conformal_report.md` |
| **Figure 1** | Main MAE Comparison Bar Charts | `python scripts/generate_figures.py` | `paper/figures/fig1_mae_comparison.pdf` |
| **Figure 4** | Periodic Table Chemical Proximity & Imputation Map | `python scripts/generate_figures.py` | `paper/figures/fig3_zsni_ablation.pdf` |
| **Entire Suite** | 1-Click Master Reproduction Workflow | `bash scripts/reproduce_all.sh` | `final_result/` |

---

### 4. Docker Reproduction
```bash
docker-compose up --build
```

---

## Repository Architecture

```text
Ragmat-ood/
├── configs/                          # Master & reference configuration files
│   ├── configs/base.yaml             # Global hyperparameter defaults
│   ├── configs/tier0_random_forest.yaml
│   └── configs/tier1_cgcnn.yaml
├── data/
│   └── splits/                       # Pre-computed split JSONs (IID, Family-Out, Element-Out)
├── docs/                             # Extended documentation & architectural design docs
├── final_result/                     # Metric JSON dumps & Markdown evaluation reports
├── paper/                            # Camera-ready manuscript source (main.tex, supplementary.tex, figures)
│   ├── paper/main.tex                # Main paper source
│   ├── paper/supplementary.tex       # Supplementary information
│   └── paper/figures/                # Vector PDF and PNG figures
├── ragmat/                           # Core Python research package
│   ├── ragmat/data/                  # JARVIS dataset loader & graph builders
│   ├── ragmat/detection/             # Mahalanobis OOD detector
│   ├── ragmat/encoders/              # CGCNN GNN message-passing architecture
│   ├── ragmat/fusion/                # RAG retrieval fusion heads & matched random controls
│   ├── ragmat/imputation/            # Zero-Shot Node Imputation (ZSNI) algorithm
│   ├── ragmat/retrieval/             # FAISS indexing & data leakage audit
│   └── ragmat/uncertainty/           # Split-conformal prediction calibration
├── scripts/                          # Production reproduction & inference CLI tools
│   ├── scripts/run_phase6.py         # Master GNN training pipeline
│   ├── scripts/run_gating_analysis.py  # Mahalanobis gating & fallback evaluation
│   ├── scripts/run_conformal.py      # ZSNI recovery & conformal UQ evaluation
│   └── scripts/run_bootstrap_cis.py  # Statistical CI generator
├── tests/                            # 35 unit and integration tests
├── .github/workflows/ci.yml          # Automated GitHub Actions CI workflow
├── CITATION.cff                      # Citation metadata
├── Dockerfile                        # Container reproduction environment
├── docker-compose.yml                # Multi-container reproduction setup
├── LICENSE                           # MIT License
├── METHODOLOGY.md                    # Comprehensive mathematical & benchmark formulation
└── REPRODUCE.md                      # Step-by-step reproduction instructions
```

---

## Citation

If you find this codebase or manuscript useful in your research, please cite:

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

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
