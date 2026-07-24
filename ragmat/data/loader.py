"""JARVIS-DFT data loader for RAGMat-OOD.

Downloads JARVIS-DFT dataset via jarvis-tools, caches raw JSON to data/raw/,
and converts entries to (pymatgen Structure, float label, material_id) tuples.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default paths relative to project root
_DEFAULT_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


class JARVISLoader:
    """Downloads and caches JARVIS-DFT structures and property labels.

    Args:
        raw_dir: Directory to cache downloaded raw JSON files.
        dataset_name: JARVIS dataset name passed to ``jarvis.db.figshare.data()``.
    """

    # Supported target properties and their JARVIS field names
    PROPERTY_FIELDS = {
        "formation_energy": "formation_energy_peratom",
        "band_gap": "optb88vdw_bandgap",
    }

    def __init__(
        self,
        raw_dir: str | Path = _DEFAULT_RAW_DIR,
        dataset_name: str = "dft_3d",
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name

    def load(
        self,
        target_property: str,
        max_samples: Optional[int] = None,
    ) -> list[tuple]:
        """Load JARVIS-DFT structures and property labels.

        Returns entries as ``(pymatgen.core.Structure, float, str)`` tuples:
        ``(structure, property_value, jid)``.

        Args:
            target_property: ``"formation_energy"`` or ``"band_gap"``.
            max_samples: If set, return only the first N valid samples (for
                smoke tests and debugging).

        Returns:
            List of ``(Structure, float, str)`` tuples, one per material.

        Raises:
            ValueError: If ``target_property`` is not supported.
        """
        from pymatgen.core import Structure

        if target_property not in self.PROPERTY_FIELDS:
            raise ValueError(
                f"Unsupported target_property '{target_property}'. "
                f"Choose from: {list(self.PROPERTY_FIELDS)}"
            )
        field = self.PROPERTY_FIELDS[target_property]

        cache_path = self.raw_dir / f"{self.dataset_name}_{target_property}.json"
        raw_data = self._load_or_download(cache_path)

        results = []
        skipped = 0
        for entry in raw_data:
            jid = entry.get("jid", "")
            val = entry.get(field)
            if val is None or val == "na" or (isinstance(val, float) and np.isnan(val)):
                skipped += 1
                continue
            try:
                atoms = entry.get("atoms")
                if atoms is None:
                    skipped += 1
                    continue
                structure = self._atoms_to_structure(atoms)
            except Exception as exc:
                logger.warning("Skipping %s — structure conversion failed: %s", jid, exc)
                skipped += 1
                continue

            results.append((structure, float(val), jid))
            if max_samples is not None and len(results) >= max_samples:
                break

        logger.info(
            "Loaded %d materials (%d skipped) for property '%s'",
            len(results),
            skipped,
            target_property,
        )
        return results

    def _load_or_download(self, cache_path: Path) -> list[dict]:
        """Return cached data or download from JARVIS figshare."""
        if cache_path.exists():
            logger.info("Loading cached data from %s", cache_path)
            with open(cache_path) as f:
                return json.load(f)

        logger.info("Downloading JARVIS dataset '%s' ...", self.dataset_name)
        from jarvis.db.figshare import data as jarvis_data

        raw = jarvis_data(self.dataset_name)
        with open(cache_path, "w") as f:
            json.dump(raw, f)
        logger.info("Cached %d entries to %s", len(raw), cache_path)
        return raw

    @staticmethod
    def _atoms_to_structure(atoms: dict):
        """Convert a JARVIS atoms dict to a pymatgen Structure.

        Args:
            atoms: Dict with keys ``"lattice_mat"``, ``"elements"``,
                ``"coords"``, ``"cartesian"``.

        Returns:
            ``pymatgen.core.Structure`` instance.
        """
        from pymatgen.core import Lattice, Structure

        lattice = Lattice(atoms["lattice_mat"])
        species = atoms["elements"]
        coords = atoms["coords"]
        cart = atoms.get("cartesian", False)
        return Structure(
            lattice,
            species,
            coords,
            coords_are_cartesian=cart,
        )
