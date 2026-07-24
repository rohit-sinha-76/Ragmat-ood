"""Leakage checker: verify zero overlap between FAISS index and test set.

This is a CRITICAL integrity check. CI must fail if this test fails.
The LeakageChecker asserts that no test material ID appears in the FAISS
index, which would constitute train/test data leakage.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


class LeakageChecker:
    """Verifies zero material ID overlap between FAISS index and test partition.

    Usage::

        checker = LeakageChecker()
        checker.assert_no_leakage(
            index_material_ids=train_ids,
            test_material_ids=test_ids,
            split_name="iid",
            property_name="formation_energy",
        )
    """

    @staticmethod
    def assert_no_leakage(
        index_material_ids: Sequence[str],
        test_material_ids: Sequence[str],
        split_name: str,
        property_name: str,
    ) -> None:
        """Assert that no test material appears in the FAISS index.

        Computes the set intersection of index IDs and test IDs. If any
        overlap is found, raises ``AssertionError`` with the offending IDs.

        Args:
            index_material_ids: Material IDs stored in the FAISS index
                (must be train-partition only).
            test_material_ids: Material IDs in the test partition.
            split_name: Name of the split (for error messages).
            property_name: Target property name (for error messages).

        Raises:
            AssertionError: If any test ID is found in the index.
        """
        index_set = set(index_material_ids)
        test_set = set(test_material_ids)
        overlap = index_set & test_set

        if overlap:
            n = len(overlap)
            sample = sorted(overlap)[:10]
            raise AssertionError(
                f"DATA LEAKAGE DETECTED in split='{split_name}', "
                f"property='{property_name}':\n"
                f"  {n} test material(s) found in the FAISS index.\n"
                f"  Sample offending IDs: {sample}"
                + (" ..." if n > 10 else "")
            )

        logger.info(
            "Leakage check PASSED: split=%s, property=%s, "
            "index_size=%d, test_size=%d, overlap=0",
            split_name,
            property_name,
            len(index_set),
            len(test_set),
        )

    @staticmethod
    def assert_split_disjoint(
        train_ids: Sequence[str],
        val_ids: Sequence[str],
        test_ids: Sequence[str],
        split_name: str,
    ) -> None:
        """Assert zero overlap across all three split partitions.

        Args:
            train_ids: Training material IDs.
            val_ids: Validation material IDs.
            test_ids: Test material IDs.
            split_name: Name of the split (for error messages).

        Raises:
            AssertionError: If any two partitions share a material ID.
        """
        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)

        violations = []
        tv = train_set & val_set
        tt = train_set & test_set
        vt = val_set & test_set

        if tv:
            violations.append(f"train∩val: {len(tv)} IDs, sample={sorted(tv)[:5]}")
        if tt:
            violations.append(f"train∩test: {len(tt)} IDs, sample={sorted(tt)[:5]}")
        if vt:
            violations.append(f"val∩test: {len(vt)} IDs, sample={sorted(vt)[:5]}")

        if violations:
            raise AssertionError(
                f"Split disjointness VIOLATED in split='{split_name}':\n"
                + "\n".join(f"  {v}" for v in violations)
            )

        logger.info(
            "Split disjointness PASSED: split=%s, "
            "train=%d, val=%d, test=%d",
            split_name,
            len(train_set),
            len(val_set),
            len(test_set),
        )
