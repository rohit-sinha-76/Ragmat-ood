# RAGMat-OOD Testing Guide

**For**: Developers adding features or fixing bugs 
**Test Status**: 15/15 passing 

---

## Test Suite Overview

### Test Categories

| Category | Count | Purpose |
|----------|-------|---------|
| **Unit Tests** | 7 | Test individual components |
| **Integration Tests** | 8 | Test end-to-end flows |
| **Critical Tests** | 4 | CI blockers (must pass) |

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_retrieval_integration.py -v

# Run with coverage
pytest tests/ --cov=ragmat --cov-report=html
```

---

## Test List

### 1. test_cgcnn_forward.py
**Type**: Unit test 
**Tests**: CGCNN forward pass

```python
def test_cgcnn_forward():
 # Creates 10 dummy crystal graphs
 # Passes through CGCNN
 # Checks output shape is (10, 1)
 # Checks no NaN/Inf in output
```

**When to run**: After modifying `ragmat/encoders/cgcnn.py`

---

### 2. test_encoder_not_pretrained.py
**Type**: Critical integrity test (CI blocker) 
**Tests**: No pretrained weights loaded

```python
def test_no_from_pretrained_method():
 # Asserts CGCNNEncoder has no from_pretrained()
 # Prevents loading external checkpoints
```

**Why critical**: Fairness - all knowledge from training data only

---

### 3. test_faiss_index.py
**Type**: Unit test 
**Tests**: FAISS index build & query

```python
def test_faiss_index():
 # Build index from 100 vectors
 # Query with 10 vectors, top_k=5
 # Checks scores shape (10, 5)
 # Checks IDs shape (10, 5)
 # Checks scores are valid (not NaN)
```

**When to run**: After modifying `ragmat/retrieval/faiss_index.py`

---

### 4. test_matminer_featurizer.py
**Type**: Unit test 
**Tests**: Matminer featurization

```python
def test_matminer_featurizer():
 # Creates 10 simple structures (NaCl, Fe)
 # Featurizes them
 # Checks output shape (10, ~282)
 # Checks no NaN in features
```

**When to run**: After modifying `ragmat/features/matminer_descriptors.py`

---

### 5. test_no_retrieval_leakage.py (2 tests)
**Type**: Critical integrity test (CI blocker) 
**Tests**: No overlap between FAISS index and test set

```python
def test_no_leakage():
 # Train IDs: [a, b, c]
 # Test IDs: [x, y, z]
 # Asserts zero intersection
```

```python
def test_leakage_detected():
 # Train IDs: [a, b, c]
 # Test IDs: [b, x, y] # b overlaps!
 # Asserts AssertionError raised
```

**Why critical**: Prevents data leakage (test info in training)

---

### 6. test_retrieval_index_property_match.py (2 tests)
**Type**: Critical integrity test (CI blocker) 
**Tests**: Config property consistency

```python
def test_retrieval_index_property_match():
 # Config with matching properties
 # Should load successfully
```

```python
def test_retrieval_index_property_mismatch():
 # Config with mismatched properties
 # Should raise ConfigIntegrityError
```

**Why critical**: Prevents mixing formation_energy and band_gap indices

---

### 7. test_retrieval_integration.py (5 tests)
**Type**: Integration tests 
**Tests**: End-to-end retrieval feature concatenation

```python
def test_concat_retrieval_features_shape():
 # Query: (20, 50), train: (100, 50), top_k=10
 # Mean pooling: output (20, 100)
 # Concat_all: output (20, 550)
```

```python
def test_random_control_shape():
 # Random control: same shape as true retrieval
 # Output: (20, 100) with mean pooling
```

```python
def test_retrieval_vs_random_different():
 # True retrieval random control
 # Neighbor features should differ
```

```python
def test_no_nan_in_output():
 # Concatenated features have no NaN/Inf
```

```python
def test_retrieval_reproducibility():
 # Same inputs same outputs
 # Deterministic behavior
```

**Why critical**: Validates the critical bug fix

---

### 8. test_split_disjointness.py (2 tests)
**Type**: Critical integrity test (CI blocker) 
**Tests**: No overlap between train/val/test

```python
def test_split_disjointness_pass():
 # IDs: train [a,b], val [c,d], test [e,f]
 # Should pass (zero overlap)
```

```python
def test_split_disjointness_fail():
 # IDs: train [a,b], val [b,c], test [d,e]
 # Should fail (b overlaps)
```

**Why critical**: Guarantees fair evaluation

---

## Critical Tests (CI Blockers)

These tests MUST pass before merging code:

1. **test_no_retrieval_leakage.py** - Prevents data leakage
2. **test_split_disjointness.py** - Ensures split isolation
3. **test_encoder_not_pretrained.py** - No external knowledge
4. **test_retrieval_index_property_match.py** - Property consistency

**CI Pipeline**: GitHub Actions runs `pytest tests/` on every push

---

## Writing New Tests

### Template for Unit Test

```python
import numpy as np
import pytest

def test_my_feature():
 """Test description in one sentence."""
 # 1. Setup: Create test data
 input_data = np.random.randn(10, 5)
 
 # 2. Execute: Call the function
 result = my_function(input_data)
 
 # 3. Assert: Check expectations
 assert result.shape == (10, 5), f"Expected (10,5), got {result.shape}"
 assert not np.isnan(result).any(), "Result contains NaN"
 assert not np.isinf(result).any(), "Result contains Inf"
```

### Template for Integration Test

```python
def test_end_to_end_flow():
 """Test complete pipeline from input to output."""
 # 1. Setup entire pipeline
 loader = JARVISLoader()
 data = loader.load("formation_energy", max_samples=10)
 
 # 2. Execute pipeline
 splitter = DataSplitter()
 splits = splitter.split(...)
 
 # 3. Assert expected behavior
 assert len(splits["train"]) > 0
 assert len(set(splits["train"]) & set(splits["test"])) == 0
```

---

## Debugging Failed Tests

### Common Failures

#### 1. Shape Mismatch

```
AssertionError: Expected (20, 564), got (20, 282)
```

**Cause**: Retrieval features not concatenated 
**Fix**: Check `concat_retrieval_features()` is called

#### 2. Leakage Detected

```
AssertionError: Found 5 overlapping IDs between index and test
```

**Cause**: FAISS index built from wrong partition 
**Fix**: Use `train_ids` only in `index.build()`

#### 3. Config Validation Error

```
ConfigIntegrityError: encoder_property 'band_gap' must equal target_property 'formation_energy'
```

**Cause**: Mismatched config properties 
**Fix**: Set `encoder_property == target_property`

#### 4. NaN in Output

```
AssertionError: Result contains NaN
```

**Cause**: Invalid features or missing data handling 
**Fix**: Check for NaN handling in featurization

---

## Test Data

### Where Test Data Comes From

Most tests use **synthetic data** (no real JARVIS download needed):

```python
# Synthetic structures
import numpy as np
from pymatgen.core import Structure, Lattice

lattice = Lattice.cubic(5.0)
structure = Structure(lattice, ["Fe"], [[0, 0, 0]])

# Random features for testing
features = np.random.randn(100, 282).astype(np.float32)
```

### Tests That Use Real Data

Only these tests download JARVIS (skipped in CI with `max_samples=10`):
- `test_matminer_featurizer.py` (10 samples only)

---

## Code Coverage

**Target**: >90% coverage on core modules

```bash
# Generate coverage report
pytest tests/ --cov=ragmat --cov-report=html

# Open report
open htmlcov/index.html
```

**Current Coverage**:
- `ragmat/config.py`: 100%
- `ragmat/retrieval/concat_features.py`: 100%
- `ragmat/retrieval/faiss_index.py`: 95%
- `ragmat/train.py`: 85%

---

## Adding Tests for New Features

### Checklist

- [ ] Write unit test for new function
- [ ] Write integration test if affects pipeline
- [ ] Add docstring to test function
- [ ] Test edge cases (empty input, NaN, zero length)
- [ ] Run `pytest tests/ -v` locally
- [ ] Ensure all 15 existing tests still pass

### Example: Adding OOD Detection Test

```python
def test_mahalanobis_detector():
 """Test Mahalanobis OOD detection."""
 from ragmat.ood.mahalanobis import MahalanobisDetector
 
 # Train embeddings (in-distribution)
 train_embs = np.random.randn(100, 64).astype(np.float32)
 
 # Test embeddings (some OOD)
 test_embs = np.random.randn(20, 64).astype(np.float32)
 test_embs[10:] += 10.0 # Make second half OOD
 
 # Fit and score
 detector = MahalanobisDetector()
 detector.fit(train_embs)
 scores = detector.score(test_embs)
 
 # Assert
 assert scores.shape == (20,)
 assert scores[10:].mean() > scores[:10].mean() # OOD scores higher
```

---

## Continuous Integration

### GitHub Actions Workflow

**File**: `.github/workflows/ci.yml`

```yaml
name: Tests
on: [push, pull_request]
jobs:
 test:
 runs-on: ubuntu-latest
 steps:
 - uses: actions/checkout@v3
 - uses: actions/setup-python@v4
 with:
 python-version: '3.11'
 - run: pip install -r requirements.txt
 - run: pytest tests/ -v
```

**Trigger**: Every push to `main` and all pull requests 
**Fail Condition**: Any test failure blocks merge

---

## Best Practices

1. **Test one thing per test** - Easier to debug
2. **Use descriptive names** - `test_retrieval_features_shape` not `test1`
3. **Add docstrings** - Explain what is being tested
4. **Test edge cases** - Empty input, NaN, inf, zero length
5. **Keep tests fast** - Use small synthetic data
6. **Make tests deterministic** - Set random seeds
7. **Assert meaningful messages** - Include expected vs actual

---

## Troubleshooting

### Tests Pass Locally but Fail in CI

**Cause**: Environment differences (CUDA, file paths) 
**Fix**: Use `device="cpu"` in tests, use relative paths

### Tests Take Too Long

**Cause**: Using full JARVIS dataset 
**Fix**: Use `max_samples=10` or synthetic data

### Flaky Tests

**Cause**: Non-deterministic behavior 
**Fix**: Set random seeds (`np.random.seed(42)`)

---

## Summary

- **15 tests total**, all must pass
- **4 critical tests** (CI blockers)
- **Run before committing**: `pytest tests/ -v`
- **Add tests for new features**
- **Keep tests fast and deterministic**
