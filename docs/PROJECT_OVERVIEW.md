# RAGMat-OOD Project Overview

**For**: Junior developers joining the project 
**Last Updated**: 2024

---

## What is RAGMat-OOD?

RAGMat-OOD is a **machine learning system** that predicts material properties (formation energy, band gap) using **retrieval-augmented prediction** to handle **out-of-distribution (OOD) data**.

### The Problem

Traditional ML models perform poorly when test data differs from training data. In materials science, this happens when:
- Testing on materials from different crystal families (family-out)
- Testing on materials with rare elements (element-out)

### Our Solution

**Retrieval-Augmented Generation (RAG)** for materials:
1. When making a prediction, **retrieve similar training examples**
2. **Concatenate** retrieved features with query features
3. **Predict** using augmented features

**Analogy**: Like asking "what did similar materials do?" before predicting.

---

## Key Concepts

### 1. Two-Tier Architecture

**Tier 0**: Traditional features (matminer) + sklearn (Random Forest/XGBoost)
**Tier 1**: Graph Neural Networks (CGCNN) with learned embeddings

Both support retrieval-augmented prediction.

### 2. Three Retrieval Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `none` | No retrieval (baseline) | Performance comparison |
| `true_neighbor` | Retrieve semantically similar neighbors | Main approach |
| `random_control` | Retrieve random neighbors | Control experiment |

### 3. Three OOD Splits

| Split | Test Set | Difficulty |
|-------|----------|-----------|
| IID | Random 20% stratified split | In-distribution baseline |
| family_out | 23 AFLOW crystallographic prototype families held out | Medium OOD (Structural) |
| element_out | 15 withheld elements (14 transition metals + Se probe) | Hard OOD (Compositional) |

---

## Project Goals

1. **Diagnose OOD failure mechanisms** in lookup-layer GNNs under element exclusion
2. **Audit Retrieval-Augmented Generation (RAG)** against matched random controls
3. **Develop zero-shot inference-time recovery** (ZSNI & Mahalanobis gating)
4. **Quantify uncertainty & conformal coverage** under distribution shift

---

## Tech Stack

- **Python 3.10+**: Language
- **PyTorch 2.5**: Deep learning
- **PyG 2.6**: Graph neural networks
- **FAISS 1.14**: Fast similarity search
- **matminer 0.10**: Materials featurization
- **scikit-learn 1.5**: Traditional ML
- **pytest 8.0**: Testing

---

## Repository Structure

```
ragmat-ood/
 configs/ # Master & reference YAML experiment configs
 data/ # JARVIS-DFT data split indices
 ragmat/ # Core package (encoders, retrieval, ZSNI, gating)
 final_result/ # Final publication evaluation outputs & reports
 paper/ # LaTeX manuscript source & vector figures
 scripts/ # Production execution & reproduction scripts
 tests/ # Test suite
```

---

## Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate ragmat
```

### 2. Train a Model

```bash
python -m ragmat.train configs/tier0_random_forest.yaml
```

### 3. Evaluate

```bash
python -m eval.run_eval \
 --checkpoint checkpoints/tier0_formation_energy_iid_model.pkl \
 --config configs/tier0_random_forest.yaml \
 --output-dir final_result/
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Key Files for Junior Developers

**Start Here**:
1. `README.md` - Setup and basic usage
2. `docs/ARCHITECTURE.md` - System design
3. `docs/PIPELINE_FLOW.md` - Execution flows
4. `docs/TEST_GUIDE.md` - Testing guide

**Core Modules**:
- `ragmat/config.py` - Configuration system
- `ragmat/train.py` - Training pipeline
- `ragmat/retrieval/concat_features.py` - **CRITICAL** retrieval integration
- `eval/run_eval.py` - Evaluation pipeline

---

## Critical Integrity Rules

**NEVER**:
1. Build FAISS index from all data (only train!)
2. Fit scaler on all data (only train!)
3. Share FAISS index across properties
4. Load pretrained encoders
5. Forget to concatenate retrieval features

**ALWAYS**:
1. Validate zero overlap between splits
2. Check for data leakage
3. Run tests before committing
4. Use config files (no hardcoded hyperparameters)

---

## Recent Critical Bug Fix

**Bug**: Retrieval features were NOT being concatenated (silent failure)
**Impact**: Retrieval model = baseline model
**Fix**: Added `concat_retrieval_features()` in training and evaluation
**Status**: Fixed, all 15 tests pass

See `docs/FIX_SUMMARY.md` for details.

---

## Getting Help

- **Architecture**: Read `docs/ARCHITECTURE.md`
- **Pipeline**: Read `docs/PIPELINE_FLOW.md`
- **Testing**: Read `docs/TEST_GUIDE.md`
- **Code**: Read docstrings (all modules documented)

---

## Development Workflow

1. **Read** relevant docs
2. **Understand** architecture and integrity rules
3. **Write** code following patterns
4. **Test** with pytest
5. **Document** changes
6. **Commit** with clear messages

---

## Project Status

- **Codebase**: 87% complete
- **Tests**: 15/15 passing 
- **Critical Bug**: Fixed 
- **Next**: Full evaluation on all 12 configurations
