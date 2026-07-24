# Reproduction Instructions

Follow these step-by-step instructions to reproduce all key results, baseline models, retrieval fusion evaluations, and recovery metrics reported in the paper.

---

### Step 1: Environment Setup
Create the conda environment using the definition in `environment.yml` and activate it:
```bash
conda env create -f environment.yml
conda activate ragmat
```

Alternatively, using Docker for containerized reproduction:
```bash
docker-compose up --build
```

---

### Step 2: Data Ingestion
JARVIS-DFT crystal structures are fetched and parsed automatically on the first execution of any pipeline script using `jarvis-tools`. Pre-computed split index files are provided in `data/splits/`.

---

### Step 3: Run System Integrity Tests
Execute the pytest suite to verify environment integrity, PyTorch Geometric graph builders, and model specs:
```bash
pytest tests/ -v
```

---

### Step 4: Tier 0 Experiments (Random Forest Baselines)
Train the Magpie descriptor Random Forest baseline using the reference configuration:
```bash
python -m ragmat.train configs/tier0_random_forest.yaml
```

---

### Step 5: Tier 1 CGCNN Training
Train base CGCNN encoders from scratch across all properties and evaluation splits:
```bash
# Train using clean Tier 1 reference config
python -m ragmat.train configs/tier1_cgcnn.yaml

# Or train all 6 base models (2 properties x 3 splits) via the master script
python scripts/run_phase6.py --stage 1 --prop all --split all
```

---

### Step 6: RAG Retrieval Fusion Evaluation
Train post-pooling fusion heads (`concat` and `cross_attention`) for both true-neighbor retrieval and matched random controls:
```bash
python scripts/run_phase6.py --stage 3 --prop all --split all --mode all
```

---

### Step 7: Zero-Shot Node Imputation (ZSNI) & Conformal UQ
Run the Zero-Shot Node Imputation (ZSNI) weight column patching and split-conformal coverage evaluation:
```bash
python scripts/run_conformal.py
```

---

### Step 8: Mahalanobis Latent Gating Evaluation
Fit the 64-dimensional embedding OOD detector and evaluate adaptive fallback routing to Random Forest:
```bash
python scripts/run_gating_analysis.py
```

---

### Step 9: Generate Final Statistical Reports & Tables
Generate the statistical summary reports corresponding to Tables 1-5 in the manuscript:
```bash
python scripts/run_bootstrap_cis.py # Generates final_result/bootstrap_cis_report.md
python scripts/run_gating_analysis.py # Generates final_result/gating_final_report.md
python scripts/run_interpretability.py # Generates final_result/interpretability_report.md
```
