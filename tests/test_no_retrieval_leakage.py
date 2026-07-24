"""Test retrieval leakage."""
import pytest
from ragmat.retrieval.leakage_check import LeakageChecker

def test_no_leakage():
    LeakageChecker.assert_no_leakage(["A", "B"], ["C", "D"], "test", "prop")

def test_leakage_detected():
    with pytest.raises(AssertionError, match="DATA LEAKAGE DETECTED"):
        LeakageChecker.assert_no_leakage(["A", "B", "C"], ["C", "D"], "test", "prop")
