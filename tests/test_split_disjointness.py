"""Test split disjointness."""
import pytest
from ragmat.retrieval.leakage_check import LeakageChecker

def test_split_disjointness_pass():
    LeakageChecker.assert_split_disjoint(["A", "B"], ["C"], ["D", "E"], "test_split")

def test_split_disjointness_fail():
    with pytest.raises(AssertionError, match="train∩test"):
        LeakageChecker.assert_split_disjoint(["A", "B"], ["C"], ["B", "D"], "test_split")
