"""Data splitting for RAGMat-OOD: IID, family-out, and element-out splits.

All split index files are saved to ``data/splits/`` as JSON and checksummed.
Checksums are appended to ``data/checksums.txt``.

Critical rules enforced here:
- Scaler is NEVER fit on full dataset — only on train partition.
- Split indices are generated deterministically from a seed.
- Zero material ID overlap between train/val/test is guaranteed by construction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SPLITS_DIR = Path(__file__).parent.parent.parent / "data" / "splits"
_DEFAULT_CHECKSUMS = Path(__file__).parent.parent.parent / "data" / "checksums.txt"

# Crystal system families from spacegroup number
_CRYSTAL_FAMILIES = {
    "triclinic": range(1, 3),
    "monoclinic": range(3, 16),
    "orthorhombic": range(16, 75),
    "tetragonal": range(75, 143),
    "trigonal": range(143, 168),
    "hexagonal": range(168, 195),
    "cubic": range(195, 231),
}

# Elements withheld in element-out split (Te is default primary held-out element per spec)
_ELEMENT_OUT_SET = {"Te", "Hf", "Sc", "Ga", "Se", "Y", "In", "Nb", "Ta", "Bi"}


class DataSplitter:
    """Generate IID, family-out, and element-out data splits.

    Args:
        splits_dir: Directory to write split JSON index files.
        checksums_file: Path to the checksums ledger.
        seed: Random seed for reproducibility.
        val_fraction: Fraction of training data held for validation.
        test_fraction: Fraction of total data held for testing.
    """

    def __init__(
        self,
        splits_dir: str | Path = _DEFAULT_SPLITS_DIR,
        checksums_file: str | Path = _DEFAULT_CHECKSUMS,
        seed: int = 42,
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
    ) -> None:
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.checksums_file = Path(checksums_file)
        self.seed = seed
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self._rng = np.random.default_rng(seed)

    def split(
        self,
        material_ids: list[str],
        labels: list[float],
        structures,
        split_type: Literal["iid", "family_out", "element_out"],
        target_property: str,
    ) -> dict[str, list[str]]:
        """Generate and persist a train/val/test split.

        Args:
            material_ids: List of material JIDs.
            labels: Corresponding property values.
            structures: Corresponding pymatgen Structure objects.
            split_type: ``"iid"``, ``"family_out"``, or ``"element_out"``.
            target_property: Name of the property (used for filename).

        Returns:
            Dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping
            to a list of material IDs.
        """
        ids = np.array(material_ids)
        labels_arr = np.array(labels, dtype=float)

        if split_type == "iid":
            split_dict = self._iid_split(ids, labels_arr)
        elif split_type == "family_out":
            split_dict = self._family_out_split(ids, labels_arr, structures)
        elif split_type == "element_out":
            split_dict = self._element_out_split(ids, labels_arr, structures)
        else:
            raise ValueError(f"Unknown split_type: {split_type!r}")

        # Verify disjointness — must never overlap
        self._assert_disjoint(split_dict, split_type)

        # Persist and checksum
        filename = f"split_{split_type}_{target_property}.json"
        out_path = self.splits_dir / filename
        with open(out_path, "w") as f:
            json.dump(split_dict, f, indent=2)

        md5 = _md5_file(out_path)
        _append_checksum(self.checksums_file, filename, md5)
        logger.info(
            "Split '%s' saved: train=%d val=%d test=%d (MD5=%s)",
            split_type,
            len(split_dict["train"]),
            len(split_dict["val"]),
            len(split_dict["test"]),
            md5,
        )
        return split_dict

    def _iid_split(
        self, ids: np.ndarray, labels: np.ndarray
    ) -> dict[str, list[str]]:
        """Random stratified IID split (stratified by property quintile)."""
        n = len(ids)
        quintiles = np.digitize(labels, np.percentile(labels, [20, 40, 60, 80]))

        test_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)

        rng = np.random.default_rng(self.seed)
        for q in range(5):
            q_idx = np.where(quintiles == q)[0]
            rng.shuffle(q_idx)
            n_test = max(1, int(len(q_idx) * self.test_fraction))
            n_val = max(1, int(len(q_idx) * self.val_fraction))
            test_mask[q_idx[:n_test]] = True
            val_mask[q_idx[n_test : n_test + n_val]] = True

        train_mask = ~test_mask & ~val_mask
        return {
            "train": ids[train_mask].tolist(),
            "val": ids[val_mask].tolist(),
            "test": ids[test_mask].tolist(),
        }

    def _family_out_split(
        self, ids: np.ndarray, labels: np.ndarray, structures
    ) -> dict[str, list[str]]:
        """Hold out all materials from one crystal family (cubic held out)."""
        families = []
        for s in structures:
            try:
                sg_num = s.get_space_group_info()[1]
                fam = _spacegroup_to_family(sg_num)
            except Exception:
                fam = "unknown"
            families.append(fam)

        families_arr = np.array(families)
        holdout_family = "cubic"  # Largest OOD family
        test_mask = families_arr == holdout_family
        remaining_ids = ids[~test_mask]
        remaining_labels = labels[~test_mask]

        # Val from remaining (random)
        n_rem = len(remaining_ids)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_rem)
        n_val = max(1, int(n_rem * self.val_fraction))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        val_mask_rem = np.zeros(n_rem, dtype=bool)
        val_mask_rem[val_idx] = True

        return {
            "train": remaining_ids[~val_mask_rem].tolist(),
            "val": remaining_ids[val_mask_rem].tolist(),
            "test": ids[test_mask].tolist(),
        }

    def _element_out_split(
        self, ids: np.ndarray, labels: np.ndarray, structures
    ) -> dict[str, list[str]]:
        """Hold out all materials containing any element in ``_ELEMENT_OUT_SET``."""
        test_mask = np.zeros(len(ids), dtype=bool)
        for i, s in enumerate(structures):
            syms = {str(el) for el in s.species}
            if syms & _ELEMENT_OUT_SET:
                test_mask[i] = True

        remaining_ids = ids[~test_mask]
        n_rem = len(remaining_ids)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_rem)
        n_val = max(1, int(n_rem * self.val_fraction))
        val_idx_rem = perm[:n_val]
        val_mask_rem = np.zeros(n_rem, dtype=bool)
        val_mask_rem[val_idx_rem] = True

        return {
            "train": remaining_ids[~val_mask_rem].tolist(),
            "val": remaining_ids[val_mask_rem].tolist(),
            "test": ids[test_mask].tolist(),
        }

    @staticmethod
    def _assert_disjoint(split_dict: dict[str, list[str]], split_type: str) -> None:
        """Assert zero overlap between all partition pairs."""
        train_set = set(split_dict["train"])
        val_set = set(split_dict["val"])
        test_set = set(split_dict["test"])

        tv = train_set & val_set
        tt = train_set & test_set
        vt = val_set & test_set

        violations = []
        if tv:
            violations.append(f"train∩val={len(tv)}")
        if tt:
            violations.append(f"train∩test={len(tt)}")
        if vt:
            violations.append(f"val∩test={len(vt)}")
        if violations:
            raise AssertionError(
                f"Split '{split_type}' has overlapping partitions: "
                + ", ".join(violations)
            )

    @staticmethod
    def load_split(
        split_type: str,
        target_property: str,
        splits_dir: str | Path = _DEFAULT_SPLITS_DIR,
    ) -> dict[str, list[str]]:
        """Load a previously saved split index from disk.

        Args:
            split_type: ``"iid"``, ``"family_out"``, or ``"element_out"``.
            target_property: Property name.
            splits_dir: Directory containing split JSON files.

        Returns:
            Dict with ``"train"``, ``"val"``, ``"test"`` ID lists.
        """
        path = Path(splits_dir) / f"split_{split_type}_{target_property}.json"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        with open(path) as f:
            return json.load(f)


def _spacegroup_to_family(sg_num: int) -> str:
    """Map a spacegroup number to its crystal family name."""
    for fam, rng in _CRYSTAL_FAMILIES.items():
        if sg_num in rng:
            return fam
    return "unknown"


def _md5_file(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_checksum(checksums_file: Path, filename: str, md5: str) -> None:
    """Append or update a checksum entry in the checksums ledger."""
    checksums_file.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    if checksums_file.exists():
        with open(checksums_file) as f:
            for line in f:
                line = line.strip()
                if "  " in line:
                    m, fn = line.split("  ", 1)
                    entries[fn] = m
    entries[filename] = md5
    with open(checksums_file, "w") as f:
        for fn, m in sorted(entries.items()):
            f.write(f"{m}  {fn}\n")
