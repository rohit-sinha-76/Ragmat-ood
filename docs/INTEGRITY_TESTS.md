# Integrity Tests Reference

This document catalogs all unit and integration tests under the `tests/` directory, detailing their assertions and purposes.

---

## 1. Test Suite Catalog

### `tests/test_cgcnn_forward.py`
- `test_cgcnn_forward`
 - **Asserts**: Valid GNN forward predictions have shape `(2, 1)`, GNN embeddings have shape `(2, 64)`, and neither GNN predictions nor embeddings contain any `NaN` values.

### `tests/test_encoder_not_pretrained.py`
- `test_no_from_pretrained_method`
 - **Asserts**: The `CGCNNEncoder` class does not have weight-loading methods (`from_pretrained`, `load_pretrained`, `load_from_checkpoint`, `load_weights`) to guarantee training from scratch.
- `test_no_torch_load_in_class_source`
 - **Asserts**: The `CGCNNEncoder` class source code does not call `torch.load()` internally.
- `test_parameters_are_random_not_all_zero`
 - **Asserts**: The GNN parameters are non-zero after initialization, ensuring real randomized start values.
- `test_two_fresh_instances_differ`
 - **Asserts**: Two newly instantiated encoder instances do not share identical parameters, proving they are randomly initialized.

### `tests/test_explain.py`
- `test_explainability_cosine`
 - **Asserts**: The explainability physical relevance score matches the exact average of cosine similarities between a query and its retrieved neighbors.

### `tests/test_faiss_index.py`
- `test_faiss_index`
 - **Asserts**: The FAISS index builds successfully and returns query distances and neighbor IDs of correct shapes matching the query size and retrieval depth `top_k`.
- `test_faiss_computes_cosine_similarity`
 - **Asserts**: The index correctly L2-normalizes vectors prior to search, yielding a score of exactly `1.0` for identical vectors.

### `tests/test_logging.py`
- `test_critical_failure_raises`
 - **Asserts**: Critical failures create `logs/critical_failures.log` and correctly write structured JSON lines detailing the failure message, data payload, and type.

### `tests/test_matminer_featurizer.py`
- `test_matminer_featurizer`
 - **Asserts**: Structure featurization filters invalid inputs, generates descriptor arrays with dimensions greater than 100 columns, and has no `NaN` values.

### `tests/test_metrics_severity.py`
- `test_severity_bins`
 - **Asserts**: Splitting metrics into OOD severity bins correctly counts `low_ood` and `high_ood` samples when inputting one low and one high score.

### `tests/test_no_retrieval_leakage.py`
- `test_no_leakage`
 - **Asserts**: Disjoint training indices and test sets do not trigger leakage errors.
- `test_leakage_detected`
 - **Asserts**: Overlapping IDs between training indices and test sets raise an `AssertionError` matching the phrase `"DATA LEAKAGE DETECTED"`.

### `tests/test_ood_detector.py`
- `test_default_threshold_is_95th_pct`
 - **Asserts**: The default OOD gating threshold is set to the 95th percentile.
- `test_scores_nonzero_after_fit`
 - **Asserts**: Scores computed after fitting on training embeddings are not identical (variance > 0.0).
- `test_score_before_fit_raises`
 - **Asserts**: Requesting OOD scores before fitting on training data raises a `RuntimeError`.
- `test_indistrib_scores_below_threshold`
 - **Asserts**: More than 80% of in-distribution test samples receive OOD scores below the threshold.
- `test_ood_scores_above_threshold`
 - **Asserts**: More than 90% of clearly OOD test samples receive OOD scores above the threshold.
- `test_fit_uses_only_train_data`
 - **Asserts**: Training the detector does not access or modify test embeddings.

### `tests/test_retrieval_index_property_match.py`
- `test_valid_config_loads`
 - **Asserts**: Config files with matching target, retrieval, and encoder properties load successfully.
- `test_retrieval_index_mismatch_raises`
 - **Asserts**: Configuration files with mismatched target and retrieval properties raise validation errors.
- `test_encoder_property_mismatch_raises`
 - **Asserts**: Configuration files with mismatched target and GNN encoder properties raise validation errors.
- `test_tier0_with_cgcnn_raises`
 - **Asserts**: Setting Tier 0 (Random Forest) to load CGCNN representations raises a configuration error.

### `tests/test_retrieval_integration.py`
- `test_concat_retrieval_features_shape`
 - **Asserts**: Mean aggregation and full concatenation produce neighbor feature matrices of shape `(N, 2*D)` and `(N, (k+1)*D)` respectively.
- `test_random_control_shape`
 - **Asserts**: Random neighbor control produces features of shape `(N, 2*D)`.
- `test_retrieval_vs_random_different`
 - **Asserts**: True retrieval features and random control features differ (difference > 0.01) despite sharing identical shapes.
- `test_no_nan_in_output`
 - **Asserts**: Output feature matrices do not contain any `NaN` or infinite values.
- `test_retrieval_reproducibility`
 - **Asserts**: Retrieval features are fully deterministic given identical inputs.

### `tests/test_split_disjointness.py`
- `test_split_disjointness_pass`
 - **Asserts**: No errors are raised when splits are completely disjoint.
- `test_split_disjointness_fail`
 - **Asserts**: Mismatched partitions with intersecting IDs raise an `AssertionError` matching `"traintest"`.

### `tests/test_uq.py`
- `test_conformal_coverage`
 - **Asserts**: Conformal calibration intervals cover true labels at a rate between 85% and 95% (calibrated at 90% target coverage).
- `test_mc_dropout_variance`
 - **Asserts**: Active GNN dropout during inference yields non-zero prediction variance.

---

## 2. Test Execution

All tests are executed using:
```bash
pytest tests/ -v --cov=ragmat --cov-report=xml
```

### Expected Output
```
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- /home/hp-sam/miniforge3/envs/ragmat/bin/python3.11
cachedir: .pytest_cache
rootdir: /mnt/c/Users/HP/Desktop/research/ragmat-ood
configfile: pyproject.toml
plugins: cov-7.1.0
collecting ... collected 32 items

tests/test_cgcnn_forward.py::test_cgcnn_forward PASSED [ 3%]
tests/test_encoder_not_pretrained.py::test_no_from_pretrained_method PASSED [ 6%]
tests/test_encoder_not_pretrained.py::test_no_torch_load_in_class_source PASSED [ 9%]
tests/test_encoder_not_pretrained.py::test_parameters_are_random_not_all_zero PASSED [ 12%]
tests/test_encoder_not_pretrained.py::test_two_fresh_instances_differ PASSED [ 15%]
tests/test_explain.py::test_explainability_cosine PASSED [ 18%]
tests/test_faiss_index.py::test_faiss_index PASSED [ 21%]
tests/test_faiss_index.py::test_faiss_computes_cosine_similarity PASSED [ 25%]
tests/test_logging.py::test_critical_failure_raises PASSED [ 28%]
tests/test_matminer_featurizer.py::test_matminer_featurizer PASSED [ 31%]
tests/test_metrics_severity.py::test_severity_bins PASSED [ 34%]
tests/test_no_retrieval_leakage.py::test_no_leakage PASSED [ 37%]
tests/test_no_retrieval_leakage.py::test_leakage_detected PASSED [ 40%]
tests/test_ood_detector.py::test_default_threshold_is_95th_pct PASSED [ 43%]
tests/test_ood_detector.py::test_scores_nonzero_after_fit PASSED [ 46%]
tests/test_ood_detector.py::test_score_before_fit_raises PASSED [ 50%]
tests/test_ood_detector.py::test_indistrib_scores_below_threshold PASSED [ 53%]
tests/test_ood_detector.py::test_ood_scores_above_threshold PASSED [ 56%]
tests/test_ood_detector.py::test_fit_uses_only_train_data PASSED [ 59%]
tests/test_retrieval_index_property_match.py::test_valid_config_loads PASSED [ 62%]
tests/test_retrieval_index_property_match.py::test_retrieval_index_mismatch_raises PASSED [ 65%]
tests/test_retrieval_index_property_match.py::test_encoder_property_mismatch_raises PASSED [ 68%]
tests/test_retrieval_index_property_match.py::test_tier0_with_cgcnn_raises PASSED [ 71%]
tests/test_retrieval_integration.py::test_concat_retrieval_features_shape PASSED [ 75%]
tests/test_retrieval_integration.py::test_random_control_shape PASSED [ 78%]
tests/test_retrieval_integration.py::test_retrieval_vs_random_different PASSED [ 81%]
tests/test_retrieval_integration.py::test_no_nan_in_output PASSED [ 84%]
tests/test_retrieval_integration.py::test_retrieval_reproducibility PASSED [ 87%]
tests/test_split_disjointness.py::test_split_disjointness_pass PASSED [ 90%]
tests/test_split_disjointness.py::test_split_disjointness_fail PASSED [ 93%]
tests/test_uq.py::test_conformal_coverage PASSED [ 96%]
tests/test_uq.py::test_mc_dropout_variance PASSED [100%]

======================== 32 passed, 8 warnings in 4.83s ========================
```
