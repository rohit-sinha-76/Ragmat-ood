"""Test retrieval index property integrity rules (spec integrity rules 1 and 2).

Uses from_yaml (the real entry point) to ensure _validate() is enforced
at the point of use -- not manually called after construction.
"""
import pytest
import tempfile, os, textwrap
from ragmat.config import ExperimentConfig, ConfigIntegrityError


def _write_yaml(tmp_dir, content):
    p = os.path.join(tmp_dir, "test_cfg.yaml")
    with open(p, "w") as f:
        f.write(textwrap.dedent(content))
    return p


VALID_BASE = """
    experiment_name: "test_valid"
    tier: 0
    target_property: "band_gap"
    split_type: "iid"
    representation: "matminer"
    retrieval_mode: "none"
    encoder_property: "band_gap"
    retrieval_index_property: "band_gap"
"""


def test_valid_config_loads():
    """Valid config with matching properties must load without error."""
    with tempfile.TemporaryDirectory() as d:
        p = _write_yaml(d, VALID_BASE)
        cfg = ExperimentConfig.from_yaml(p)
        assert cfg.target_property == cfg.retrieval_index_property == cfg.encoder_property


def test_retrieval_index_mismatch_raises():
    """Rule 2: retrieval_index_property != target_property must raise via from_yaml."""
    bad = VALID_BASE.replace('retrieval_index_property: "band_gap"', 'retrieval_index_property: "formation_energy"')
    with tempfile.TemporaryDirectory() as d:
        p = _write_yaml(d, bad)
        with pytest.raises(ConfigIntegrityError, match="retrieval_index_property"):
            ExperimentConfig.from_yaml(p)


def test_encoder_property_mismatch_raises():
    """Rule 1: encoder_property != target_property must raise via from_yaml."""
    bad = VALID_BASE.replace('encoder_property: "band_gap"', 'encoder_property: "formation_energy"')
    with tempfile.TemporaryDirectory() as d:
        p = _write_yaml(d, bad)
        with pytest.raises(ConfigIntegrityError, match="encoder_property"):
            ExperimentConfig.from_yaml(p)


def test_tier0_with_cgcnn_raises():
    """Rule 3: Tier 0 must use matminer representation, not cgcnn."""
    bad = """
experiment_name: "test_bad_tier0"
tier: 0
target_property: "band_gap"
split_type: "iid"
representation: "cgcnn"
retrieval_mode: "none"
encoder_property: "band_gap"
retrieval_index_property: "band_gap"
"""
    with tempfile.TemporaryDirectory() as d:
        p = _write_yaml(d, bad)
        with pytest.raises(ConfigIntegrityError):
            ExperimentConfig.from_yaml(p)
