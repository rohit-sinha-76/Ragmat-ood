# RAGMat-OOD Pipeline Flow Documentation

**Version**: 1.0 
**Last Updated**: 2024 
**Target Audience**: Junior developers implementing features or debugging

---

## Table of Contents

1. [Overview](#overview)
2. [Training Pipeline (Tier 0)](#training-pipeline-tier-0)
3. [Training Pipeline (Tier 1)](#training-pipeline-tier-1)
4. [Evaluation Pipeline](#evaluation-pipeline)
5. [Retrieval Feature Concatenation](#retrieval-feature-concatenation)
6. [Configuration Loading](#configuration-loading)
7. [Data Loading](#data-loading)
8. [Splitting Strategy](#splitting-strategy)
9. [FAISS Index Building](#faiss-index-building)
10. [Common Workflows](#common-workflows)

---

## Overview

This document provides **step-by-step execution flows** for all major pipelines in RAGMat-OOD.

### Quick Start Commands

```bash
# Train a Tier 0 model
python -m ragmat.train configs/tier0_random_forest.yaml

# Evaluate a trained model
python -m eval.run_eval \
 --checkpoint checkpoints/tier0_formation_energy_iid_model.pkl \
 --config configs/tier0_random_forest.yaml \
 --output-dir final_result/

# Run tests
pytest tests/ -v
```

---

## Training Pipeline (Tier 0)

### High-Level Flow


```
Command: python -m ragmat.train configs/tier0_random_forest.yaml

1. Load & Validate Config (config.py)
2. Load JARVIS Data (data/loader.py)
3. Create Train/Val/Test Split (data/splitter.py)
4. Featurize Structures (features/matminer_descriptors.py)
5. Fit Scaler (train only) (features/matminer_descriptors.py)
6. Build FAISS Index (train) (retrieval/faiss_index.py)
7. Check Leakage (retrieval/leakage_check.py)
8. Concatenate Retrieval (retrieval/concat_features.py) CRITICAL
9. Train sklearn Model (train.py)
10. Save Checkpoint (checkpoints/)
```

### Detailed Step-by-Step

#### Step 1: Load & Validate Config

**File**: `ragmat/config.py`

```python
# Entry point
cfg = ExperimentConfig.from_yaml(config_path)

# What happens:
# 1. Load base.yaml (defaults)
# 2. Load experiment config (overrides)
# 3. Deep merge configs
# 4. Validate integrity rules:
# - encoder_property == target_property
# - retrieval_index_property == target_property
# - tier/representation consistency
# 5. Compute config hash (MD5)
```

