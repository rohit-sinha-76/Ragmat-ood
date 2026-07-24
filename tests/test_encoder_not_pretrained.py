"""Test encoder not pretrained (spec rule R1).

Verifies:
1. No from_pretrained / load_pretrained / load_from_checkpoint methods exist
2. Parameters are randomly initialised (not all-zeros)
3. Two fresh instances differ (random init, not fixed pretrained checkpoint)
"""
import inspect
import torch
from ragmat.encoders.cgcnn import CGCNNEncoder


def test_no_from_pretrained_method():
    """R1: No method that loads external weights."""
    encoder = CGCNNEncoder()
    for name in ["from_pretrained", "load_pretrained", "load_from_checkpoint", "load_weights"]:
        assert not hasattr(encoder, name), f"Encoder must NOT have '{name}' method."


def test_no_torch_load_in_class_source():
    """R1: CGCNNEncoder class source must not call torch.load()."""
    src = inspect.getsource(CGCNNEncoder)
    assert "torch.load" not in src, "CGCNNEncoder source contains torch.load -- forbidden per R1."


def test_parameters_are_random_not_all_zero():
    """Fresh encoder params must be non-zero (real random init)."""
    encoder = CGCNNEncoder()
    all_params = torch.cat([p.data.flatten() for p in encoder.parameters()])
    assert all_params.numel() > 0, "No parameters found."
    assert not (all_params == 0).all(), "All parameters are zero -- suspicious init."


def test_two_fresh_instances_differ():
    """Two fresh encoders must differ (random init, not fixed pretrained)."""
    enc1 = CGCNNEncoder()
    enc2 = CGCNNEncoder()
    all_same = all(torch.allclose(p1, p2) for p1, p2 in zip(enc1.parameters(), enc2.parameters()))
    assert not all_same, "Two fresh encoders are identical -- suggests fixed pretrained weights."
